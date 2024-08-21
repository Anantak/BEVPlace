#!/bin/bash

python3 --version
# Python 3.6.9

# Note that this is R32 on Jetpack 5.1. Python version is 3.6
# From NVIDIA's notes, Jetpack 5.1 should be R34 and python 3.8
# So we actually have Jetpack 4.5

# Packages that are already there
#  cv2   - 4.1.1
#  numpy - 1.13.3
#  PIL   - 8.4.0


# We first need to install scikit-image and scikit-learn 
sudo apt-get install python3-sklearn python3-sklearn-lib -y
sudo apt-get install python3-skimage python3-skimage-lib -y
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev -y
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev -y
sudo apt-get install libopenblas-dev libavcodec-dev libavformat-dev -y 
sudo apt-get install libswscale-dev -y

# Install pytorch
#  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
cd Libraries
mkdir -p Torch
cd Torch

python3 -m pip install 'Cython<3'
# Successfully installed Cython-0.29.37

# tqdm tool
python3 -m pip install tqdm
# Successfully installed importlib-resources-5.4.0 tqdm-4.64.1 zipp-3.6.0

# wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O  torch-1.10.0-cp36-cp36m-linux_aarch64.whl
# Could also use:
sudo gsutil cp gs://anantak-ups/Production/VGV/2.8.1/torch-1.10.0-cp36-cp36m-linux_aarch64.whl .
python3 -m pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Requirement already satisfied: dataclasses in /home/ubuntu/.local/lib/python3.6/site-packages (from torch==1.10.0) (0.8)
# Requirement already satisfied: numpy in /usr/lib/python3/dist-packages (from torch==1.10.0) (1.13.3)
# Installing collected packages: typing-extensions, torch
# Successfully installed torch-1.8.0 typing-extensions-4.1.1

# torch vision
# git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
# Could also use:
sudo gsutil cp gs://anantak-ups/Production/VGV/2.8.1/torch-vision-0.9.0.zip .
sudo chmod 777 ./torch-vision-0.9.0.zip
unzip ./torch-vision-0.9.0.zip
mv ./vision-0.9.0 torchvision
export BUILD_VERSION=0.9.0
export OPENBLAS_CORETYPE=ARMV8
cd torchvision
python3 setup.py install --user

# Installed /home/ubuntu/.local/lib/python3.6/site-packages/torchvision-0.9.0-py3.6-linux-aarch64.egg
# Processing dependencies for torchvision==0.9.0
# Searching for torch==1.8.0
# Best match: torch 1.8.0
# Adding torch 1.8.0 to easy-install.pth file
# Installing convert-caffe2-to-onnx script to /home/ubuntu/.local/bin
# Installing convert-onnx-to-caffe2 script to /home/ubuntu/.local/bin

# Using /home/ubuntu/.local/lib/python3.6/site-packages
# Searching for Pillow==8.4.0
# Best match: Pillow 8.4.0
# Adding Pillow 8.4.0 to easy-install.pth file

# Using /usr/local/lib/python3.6/dist-packages
# Searching for numpy==1.13.3
# Best match: numpy 1.13.3
# Adding numpy 1.13.3 to easy-install.pth file

# Using /usr/lib/python3/dist-packages
# Searching for typing-extensions==4.1.1
# Best match: typing-extensions 4.1.1
# Adding typing-extensions 4.1.1 to easy-install.pth file

# Using /home/ubuntu/.local/lib/python3.6/site-packages
# Searching for dataclasses==0.8
# Best match: dataclasses 0.8
# Adding dataclasses 0.8 to easy-install.pth file

# Using /home/ubuntu/.local/lib/python3.6/site-packages
# Finished processing dependencies for torchvision==0.9.0

cd   # loading torchvision from build dir will result in import error
#python3 -m pip install 'pillow<7' # always needed for Python 2.7, not needed torchvision v0.5.0+ with Python 3.6

# python3
# >>> import cv2
# >>> cv2.__version__
# '4.1.1'
# >>> import numpy
# >>> numpy.__version__
# '1.13.3'
# >>> import PIL
# >>> PIL.__version__
# '8.4.0'
# >>> import sklearn
# >>> sklearn.__version__
# '0.19.1'
# >>> import skimage
# >>> skimage.__version__
# '0.13.1'
# >>> import torch
# >>> torch.__version__
# '1.8.0'
# >>> import torchvision
# >>> torchvision.__version__
# '0.9.0'

# Python versions
python3 -c 'import cv2;print(f"cv2: {cv2.__version__}");import numpy;print(f"numpy: {numpy.__version__}");import PIL;print(f"PIL: {PIL.__version__}");import sklearn;print(f"sklearn: {sklearn.__version__}");import skimage;print(f"skimage: {skimage.__version__}");import torch;print(f"torch: {torch.__version__}");import torchvision;print(f"torchvision: {torchvision.__version__}");'

# How to get the system path to pytorch liraries
python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'
# /home/ubuntu/.local/lib/python3.6/site-packages/torch/share/cmake

