# syntax=docker/dockerfile:1
# docker build -t robosuite .
# FROM nvidia/cudagl:11.4.0-base-ubuntu20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

WORKDIR /srv

# https://askubuntu.com/a/1013396
ARG DEBIAN_FRONTEND=noninteractive

# TODO: make it work with egl rendering backend
ENV MUJOCO_GL=glx
# ENV MUJOCO_GL=egl

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3 \
    python3-pip \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

COPY robosuite/requirements-extra.txt /srv/robosuite/requirements-extra.txt

RUN pip install --upgrade pip
RUN pip install -r /srv/robosuite/requirements-extra.txt
RUN pip install \
    numpy \
    scipy \
    matplotlib \
    mujoco \
    pyrender \
    ipdb \
    opencv-python \
    trimesh \
    pyopengl \
    omegaconf \
    pygame \
    torch \
    ruamel.yaml

WORKDIR robosuite

COPY . /srv

RUN pip install -r requirements.txt

# CMD ["nvidia-smi"]
# CMD ["python3", "robosuite/demos/demo_renderers.py"]
# CMD ["python3", "robosuite/demos/demo_control.py"]
# CMD ["python3", "robosuite/demos/demo_random_action.py"]