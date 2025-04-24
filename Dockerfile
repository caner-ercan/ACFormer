# Created by Caner Ercan on 23.08.2024
# cercan@mdanderson.org
# Purpose: This Dockerfile sets up a container for AC-Former.

FROM nvidia/cuda:11.1.1-devel

ENV DEBIAN_FRONTEND noninteractive


RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y build-essential ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    sudo curl wget htop git ca-certificates python3-openslide python3.10 python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
# RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools

RUN apt-get install python3-pip

# WORKDIR /App
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
# RUN python3 --version
# RUN python --version
# RUN python3 get-pip.py
RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html

WORKDIR /App
RUN chmod 777 /App
# Install MMDetection
RUN git clone https://github.com/LL3RD/ACFormer.git 
WORKDIR /App/ACFormer/thirdparty/mmdetection
RUN pip install -e .
WORKDIR /App/ACFormer/
RUN pip install -e .

RUN pip install pandas matplotlib opencv-python
RUN apt-get update && apt-get install -y libtiff5-dev
RUN pip install libtiff
RUN sudo chmod -R 777 /usr/local/lib/python3.8/dist-packages/libtiff

RUN pip install yapf==0.40.1
RUN pip install albumentations>=0.3.2 --no-binary imgaug,albumentations

WORKDIR /rsrch5/home/trans_mol_path/cercan/