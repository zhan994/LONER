ARG BASE_IMAGE=sethgi/loner:base_1.0

FROM ${BASE_IMAGE}

ARG USER_NAME=loner
ARG USER_ID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

ENV TZ=America
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN useradd -m -l -u 1000 -s /bin/bash loner \
    && usermod -aG video loner \
    && export PATH=$PATH:/home/loner/.local/bin

# Give them passwordless sudo
RUN echo "loner ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to user to run user-space commands
USER loner
WORKDIR /home/loner

RUN sudo chown -R loner /home/loner
# RUN sudo rosdep init && rosdep update

# finish ROS setup
COPY .bashrc /home/loner/.bashrc

COPY ./entrypoint.sh /entrypoint.sh
RUN sudo chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]
