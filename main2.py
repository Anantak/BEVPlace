from __future__ import print_function
import argparse
from os.path import join, isfile
from os import environ
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network.bevplace import BEVPlace
from network.utils import to_cuda

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='BEVPlace')
parser.add_argument('--test_batch_size', type=int, default=8, help='Batch size for testing')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=4, help='Number of threads for each data loader to use')
parser.add_argument('--resume', type=str, default='checkpoints/checkpoint_paper_kitti.pth.tar', help='Path to load checkpoint from, for resuming training or testing.')

import seaborn as sns
import matplotlib.pyplot as plt


def evaluate2(eval_set, model):
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False, 
                pin_memory=cuda)

    model.eval()

    global_features = []
    with torch.no_grad():
        print('====> Extracting Features')
        with tqdm(total=len(test_data_loader)) as t:
            for iteration, (input, indices) in enumerate(test_data_loader, 1):
                # print(input[0][0][0][0].shape) # this is the image shape
                if cuda:
                    input = to_cuda(input)
                batch_feature = model(input)
                global_features.append(batch_feature.detach().cpu().numpy())
                t.update(1)

    global_features = np.vstack(global_features)
    # print(global_features)
    print(global_features.shape)

    corr_mat = np.dot(global_features, global_features.T)
    print(corr_mat.shape)

    # Create a heatmap using Seaborn
    sns.heatmap(corr_mat, cmap="coolwarm")

    # Add labels and title
    plt.xlabel("RigFrames")
    plt.ylabel("RigFrames")
    plt.title("Similarity Matrix")

    # Show the plot
    plt.show()

import dataset as dataset

from network import netvlad

if __name__ == "__main__":
    opt = parser.parse_args()

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")
    
    print('===> Building model')
    model = BEVPlace()
    resume_ckpt = opt.resume

    print("=> loading checkpoint '{}'".format(resume_ckpt))
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    model = model.to(device)
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(resume_ckpt, checkpoint['epoch']))


    if cuda:
        model = nn.DataParallel(model)
        # model = model.to(device)

    data_path = '/home/ubuntu/Downloads/BEV/'
    seq = '05'
    eval_set = dataset.ANADataset(data_path, seq)
    recalls = evaluate2(eval_set, model)
