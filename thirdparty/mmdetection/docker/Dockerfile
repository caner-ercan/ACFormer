FROM nvidia/cuda:11.1.1-devel

ENV DEBIAN_FRONTEND noninteractive


RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y build-essential ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    sudo curl wget htop git ca-certificates python3-openslide python3.8 python3.8-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools
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