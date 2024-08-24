FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# set environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV HDF5_USE_FILE_LOCKING FALSE
ENV NUMBA_CACHE_DIR /tmp

# install libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential libgl1-mesa-glx libglib2.0-0 libgeos-dev libvips-tools \
  sudo curl wget htop git vim ca-certificates python3-openslide python \
  && rm -rf /var/lib/apt/lists/*

# install Miniconda
WORKDIR /App
RUN chmod 777 /App
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh
RUN bash Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -p /App/miniconda -b 
RUN rm Miniconda3-py310_24.1.2-0-Linux-x86_64.sh
ENV PATH=/App/miniconda/bin:$PATH

## create a conda python 3.10 environment
RUN /App/miniconda/bin/conda install conda-build \
 && /App/miniconda/bin/conda create -y --name py310 python=3.10.13 \
 && /App/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py310
ENV CONDA_PREFIX=/App/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# install python packages
RUN conda install -y pytorch=1.13.1 torchvision=0.14.1 cudatoolkit=11.8 -c pytorch -c nvidia
RUN pip install tensorflow==2.11 
RUN pip install tensorflow-hub==0.12.0 tensorboard==2.13.0 tensorrt
# tensorrt==8.6.1.post1 tensorflow-datasets==4.9.2
# RUN pip install gpustat==0.6.0 setuptools==61.2.0 pytz==2023.3 joblib==1.2.0 tqdm==4.66.1 docopt==0.6.2
# RUN pip install ipython==8.10.0 jupyterlab==3.6.1 notebook==6.4.11 traitlets==5.9.0 chardet==5.0.0 nbconvert==7.8.0 seaborn==0.13.0 pycm==3.5 deepdish==0.3.7 
RUN pip install pandas==2.1.2 matplotlib==3.7.2 
RUN pip install openslide-python==1.3.1 opencv-python==4.8.0.74 scikit-image==0.22.0 scikit-learn==1.3.2 albumentations==1.4.2
# RUN pip install scikit-learn==1.3.2 xgboost==2.0.3 statsmodels==0.13.5 lifelines==0.27.8 
# RUN pip install pyg-nightly==2.4.0.dev20231209 networkx==3.2 pysal==23.7 spatialentropy==0.0.4 
# RUN pip install transformers==4.36.2 pathologyfoundation==0.1.14 ultralytics==8.0.230 
RUN pip install stardist==0.8.5 csbdeep==0.7.4 

RUN apt update && apt install -y --no-install-recommends \
        libgdal-dev
RUN pip install GDAL=="$(gdal-config --version).*"
RUN pip install histomicstk --find-links https://girder.github.io/large_image_wheels
RUN pip install histomicstk==1.3.5

WORKDIR /.cache/torch/hub/checkpoints/
RUN wget https://download.pytorch.org/models/densenet121-a639ec97.pth
RUN wget https://download.pytorch.org/models/resnet50-19c8e357.pth


#segtransform
RUN pip install transformers 
RUN pip install jupyterlab
RUN pip install ultralytics==8.0.230
# configure folder permission
WORKDIR /.dgl
RUN chmod -R 777 /.dgl
WORKDIR /.local
RUN chmod -R 777 /.local
WORKDIR /.cache
RUN chmod -R 777 /.cache

# /Data folder preparation
WORKDIR /Data
RUN chmod -R 777 /Data

WORKDIR /App
RUN pip install -U openmim
RUN mim install mmcv-full==1.6.1
RUN git clone https://github.com/caner-ercan/ACFormer.git
WORKDIR /App/ACFormer/thirdparty/mmdetection
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1
RUN mim install mmcv-full
RUN pip install --no-cache-dir -e .
WORKDIR /App
RUN pip install -e ACFormer
RUN pip install pybboxes==0.1.5
WORKDIR /App/ACFormer/tools/sahi
RUN pip install -e .

# bash at /App
WORKDIR /rsrch5/home/trans_mol_path/cercan
CMD ["/bin/bash"]