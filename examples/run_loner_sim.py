#!/usr/bin/env python3

"""
File: examples/run_loner.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

import argparse
import datetime
import os
import sys
import time

import cv2
import pandas as pd
import yaml
import ros_numpy
import rosbag
import rospy
import tf2_py
import torch
from attrdict import AttrDict
from cv_bridge import CvBridge
from pathlib import Path
from sensor_msgs.msg import Image, PointCloud2
import torch.multiprocessing as mp

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")


from src.loner import Loner
from src.common.pose import Pose
from src.common.sensors import Image, LidarScan
from src.common.settings import Settings
from src.common.pose_utils import build_poses_from_df
from src.common.pose_utils import build_poses_from_df_2D

from examples.utils import *

LIDAR_MIN_RANGE = 0.3 #http://www.oxts.com/wp-content/uploads/2021/01/Ouster-datasheet-revc-v2p0-os0.pdf

bridge = CvBridge()

WARN_MOCOMP_ONCE = True
WARN_LIDAR_TIMES_ONCE = True

def readTxt(filepath: str = ''):
    if not filepath:
        print("The filepath is None.")

    with open(filepath,"r") as f:
        file = f.read()
    data_str = file.split()
    data = [float(i) for i in data_str]
    data = np.array(data)
    data = data.reshape(-1,12)
    # arr = np.array([1,1,1,1,1,1,1,1,1,1,1,1],dtype=float)
    # data = data / arr
    return data

def build_scan_from_msg(lidar_scan, fov: dict = None) -> LidarScan:

    num_points = lidar_scan.shape[0]

    xyz = torch.zeros((num_points, 3,), dtype=torch.float32)
    xyz[:,0] = torch.from_numpy(lidar_scan[:,2].copy().reshape(-1,))
    xyz[:,1] = torch.from_numpy(lidar_scan[:,3].copy().reshape(-1,))
    xyz[:,2] = torch.from_numpy(lidar_scan[:,4].copy().reshape(-1,)) 

    dists = xyz.norm(dim=1)

    valid_ranges = dists > LIDAR_MIN_RANGE

    xyz = xyz[valid_ranges].T
    
    global WARN_MOCOMP_ONCE

    timestamps = torch.zeros((num_points), dtype=torch.float32)
    timestamps = torch.from_numpy(lidar_scan[:,5].copy().reshape(-1,))
    timestamps = timestamps[valid_ranges]
    timestamps = timestamps.float()

    dists = dists[valid_ranges].float()
    directions = (xyz / dists).float()

    timestamps, indices = torch.sort(timestamps)
    
    dists = dists[indices]
    directions = directions[:,indices]

    return LidarScan(directions.float().cpu(), dists.float().cpu(), timestamps.float().cpu())


def run_trial(config, settings, settings_description = None, config_idx = None, trial_idx = None, dryrun: bool = False):
    im_scale_factor = settings.system.image_scale_factor

    txt_path = Path(os.path.expanduser(config["dataset"]))

    init_clock = time.time()
    
    # calibration = load_calibration(config["dataset_family"], config["calibration"])
    # calibration = None
    camera_to_lidar = None
    image_size = None

    ray_range = settings.mapper.optimizer.model_config.data.ray_range

    settings["experiment_name"] = config["experiment_name"]

    settings["run_config"] = config

    loner = Loner(settings)

    # Get ground truth trajectory. This is only used to construct the world cube.
    if config["groundtruth_traj"] is not None:
        ground_truth_file = os.path.expanduser(config["groundtruth_traj"])
        ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")
        lidar_poses, timestamps = build_poses_from_df(ground_truth_df, True)
        tf_buffer, timestamps = build_buffer_from_poses(lidar_poses, timestamps)
    else:
        tf_buffer = None
        lidar_poses = None

    if config_idx is None and trial_idx is None:
        ablation_name = None
    else:
        ablation_name = config["experiment_name"]

    if settings.system.world_cube.compute_from_groundtruth:
        assert lidar_poses is not None, "Must provide groundtruth file, or set system.world_cube.compute_from_groundtruth=False"
        traj_bounding_box = None
        lidar_poses_init = lidar_poses
    else:
        lidar_poses_init = None
        traj_bounding_box = settings.system.world_cube.trajectory_bounding_box
      
    msgs = readTxt(txt_path)
    lidar_poses = build_poses_from_df_2D(msgs[:,[6,7,10,11]],True)
    lidar_poses_init = torch.unique(lidar_poses, sorted = False, dim = 0)
    loner.initialize(camera_to_lidar, lidar_poses_init, settings.calibration.camera_intrinsic.k,
                            ray_range, image_size, txt_path.as_posix(), ablation_name, config_idx, trial_idx,
                            traj_bounding_box)
    logdir = loner._log_directory

    if settings_description is not None and config_idx is not None:
        if trial_idx == 0:
            with open(f"{logdir}/../configuration.txt", 'w+') as desc_file:
                desc_file.write(settings_description)
        elif trial_idx is None:
            with open(f"{logdir}/configuration.txt", 'w+') as desc_file:
                desc_file.write(settings_description)
    
    if dryrun:
        return
     
    loner.start()
    start_clock = time.time()
    start_lidar_pose = None
    lidar_prev = msgs[0,[6,7,10,11]]
    start = 0
    end = 0
    while end<msgs.shape[0]-1:
        end = end+1
        if not np.sum(abs(msgs[end,[6,7,10,11]]-lidar_prev))<1:
            lidar_scan = build_scan_from_msg(msgs[start:end,:])           
            T_lidar = msg_to_transformation_mat_2D(lidar_poses[start,:])
            if start_lidar_pose is None:
                start_lidar_pose = T_lidar     
            gt_lidar_pose = start_lidar_pose.inverse() @ T_lidar
            loner.process_lidar(lidar_scan, Pose(gt_lidar_pose))
            start = end
            lidar_prev = msgs[end,[6,7,10,11]]
        # 存在一个bug，start-28940
    if start<msgs.shape[0]-1:
        lidar_scan = build_scan_from_msg(msgs[start:end,:])           
        T_lidar = msg_to_transformation_mat_2D(lidar_poses[start,:])
        if start_lidar_pose is None:
            start_lidar_pose = T_lidar     
        gt_lidar_pose = start_lidar_pose.inverse() @ T_lidar
        loner.process_lidar(lidar_scan, Pose(gt_lidar_pose))

    loner.stop()
    end_clock = time.time()

    with open(f"{loner._log_directory}/runtime.txt", 'w+') as runtime_f:
        runtime_f.write(f"Execution Time (With Overhead): {end_clock - init_clock}\n")
        runtime_f.write(f"Execution Time (Without Overhead): {end_clock - start_clock}\n")

# Implements a single worker in a thread-pool model.
def _gpu_worker(config, gpu_id: int, job_queue: mp.Queue, dryrun: bool) -> None:

    while not job_queue.empty():
        data = job_queue.get()
        if data is None:
            return

        settings, description, config_idx, trial_idx = data
        run_trial(config, settings, description, config_idx, trial_idx, dryrun)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run Loner SLAM on RosBag")
    parser.add_argument("configuration_path")
    parser.add_argument("experiment_name", nargs="?", default=None)
    parser.add_argument("--duration", help="How long to run for (in input data time, sec)", type=float, default=None)
    parser.add_argument("--gpu_ids", nargs="*", required=False, default = None, help="Which GPUs to use. Defaults to parallel if set")
    parser.add_argument("--num_repeats", type=int, required=False, default=1, help="How many times to run the experiment")
    parser.add_argument("--run_all_combos", action="store_true",default=False, help="If set, all combinations of overrides will be run. Otherwise, one changed at a time.")
    parser.add_argument("--overrides", type=str, default=None, help="File specifying parameters to vary for ablation study or testing")
    parser.add_argument("--dryrun", action="store_true",default=False, help="If set, generates output dirs and settings files but doesn't run anything.")

    args = parser.parse_args()


    with open(args.configuration_path) as config_file:
        config = yaml.full_load(config_file)

    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name

    config["duration"] = args.duration

    baseline_settings_path = os.path.expanduser(f"~/LonerSLAM/cfg/{config['baseline']}")

    if args.overrides is not None:
        settings_options, settings_descriptions = \
            Settings.generate_options(baseline_settings_path,
                                      args.overrides,
                                      args.run_all_combos,
                                      [config["changes"]])
        
    else:
        settings_descriptions = [None]
        settings_options = [Settings.load_from_file(baseline_settings_path)]            

        if config["changes"] is not None:
                settings_options[0].augment(config["changes"])


    if len(settings_options) > 1 or args.num_repeats > 1:
        now = datetime.datetime.now()
        now_str = now.strftime("%m%d%y_%H%M%S")
        config["experiment_name"] += f"_{now_str}"

    if args.gpu_ids is not None and len(args.gpu_ids) > 1:
        mp.set_start_method('spawn')
        
        if len(settings_descriptions) > 1:
            config_idxs = range(len(settings_descriptions))
        else:
            config_idxs = [None]

        job_queue_data = zip(settings_options, settings_descriptions, config_idxs)

        job_queue = mp.Queue()
        for element in job_queue_data:
            if args.num_repeats == 1:
                job_queue.put(element + (None,))
            else:
                for trial_idx in range(args.num_repeats):
                    job_queue.put(element + (trial_idx,))
        
        for _ in args.gpu_ids:
            job_queue.put(None)

        # Create the workers
        gpu_worker_processes = []
        for gpu_id in args.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            gpu_worker_processes.append(mp.Process(target = _gpu_worker, args=(config,gpu_id,job_queue,args.dryrun)))
            gpu_worker_processes[-1].start()

        # Sync
        for process in gpu_worker_processes:
            process.join()
        
    else:
        if args.gpu_ids is not None:
            gpu_id = str(args.gpu_ids[0])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        for config_idx, (settings, description) in enumerate(zip(settings_options, settings_descriptions)):
            if len(settings_options) == 1:
                config_idx = None
            for trial_idx in range(args.num_repeats):
                if args.num_repeats == 1:
                    trial_idx = None
                run_trial(config, settings, description, config_idx, trial_idx, args.dryrun)
