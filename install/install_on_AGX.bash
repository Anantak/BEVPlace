#!/bin/bash

# Current AGX computers are on Ubuntu 18LTS and Python 3.6

# Install pytorch
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
cd ~
mkdir -p Libraries
cd Libraries
mkdir -p Torch
cd Torch

# Install pyTorch
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O  torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
python3 -m pip install 'Cython<3'
python3 -m pip install numpy torch-1.10.0-cp36-cp36m-linux_aarch64.whl
# Ends with:
# Successfully installed dataclasses-0.8 torch-1.8.0 typing-extensions-4.1.1

# Install torchvision
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
export BUILD_VERSION=0.9.0
export OPENBLAS_CORETYPE=ARMV8
cd torchvision
python3 setup.py install --user
#cd ../  # loading torchvision from build dir will result in import error
#python3 -m pip install 'pillow<7' # always needed for Python 2.7, not needed torchvision v0.5.0+ with Python 3.6

# Back to home
cd ~

# Install basics
python3 -m pip install tqdm

# Install scikit
python3 -m pip install scikit-image
python3 -m pip install scikit-learn

