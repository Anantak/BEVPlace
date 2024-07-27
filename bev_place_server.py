#!/usr/bin/env python3

# Addressing the Protobuf issue
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python" 

import sys
import signal
import argparse
import zmq
import json 
import sched, time

from termcolor import cprint
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from network.bevplace import BEVPlace
from network.utils import to_cuda

import torchvision.transforms as transforms
import torch.utils.data as data

from network.utils import TransformerCV
from network.groupnet import group_config

sys.path.append('/home/ubuntu/Anantak/Admin/messaging');
import sensor_messages_pb2


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

class SingleImageDataset(Dataset):
    def __init__(self, image):
        self.image = image
        self.input_transform = input_transform()
        self.transformer = TransformerCV(group_config)
        self.pts_step = 5

    def transformImg(self, img):
        xs, ys = np.meshgrid(np.arange(self.pts_step,img.size()[1]-self.pts_step,self.pts_step), np.arange(self.pts_step,img.size()[2]-self.pts_step,self.pts_step))
        xs = xs.reshape(-1,1)
        ys = ys.reshape(-1,1)
        pts = np.hstack((xs,ys))
        img = img.permute(1,2,0).detach().numpy()
        transformed_imgs=self.transformer.transform(img,pts)
        data = self.transformer.postprocess_transformed_imgs(transformed_imgs)
        return data

    def __getitem__(self, index):
        img = self.input_transform(self.image)
        img *= 255
        img = self.transformImg(img)
        return img, index

    def __len__(self):
        return 1

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='BEVPlace')
parser.add_argument('--listenPort', type=int, default=51705, help='port to use for listening for images')
parser.add_argument('--publishPort', type=int, default=51706, help='port to use for publishing back')
parser.add_argument("--periodic", type=bool, default=False, help="Run periodically at 2Hz with noblock")
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=4, help='Number of threads for each data loader to use')
parser.add_argument('--resume', type=str, default='checkpoints/checkpoint_paper_kitti.pth.tar', help='Path to load checkpoint from, for resuming training or testing.')

# Static BEV Place server
class BEVPlaceServer:

    # Static variables of the class
    opt = parser.parse_args()

    # How many images have been processed
    num_images_processed = 0

    # ZMQ socket objects
    zmq_context = zmq.Context()
    ZMQ_sub_PORT = opt.listenPort
    zmq_sub_read_socket = zmq_context.socket(zmq.SUB)
    zmq_sub_read_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    zmq_sub_read_socket.setsockopt(zmq.CONFLATE, 1)
    zmq_sub_read_socket.connect(f"tcp://127.0.0.1:{ZMQ_sub_PORT}")
    ZMQ_pub_PORT = opt.publishPort
    zmq_pub_socket = zmq_context.socket(zmq.PUB)
    zmq_pub_url = f"tcp://127.0.0.1:{ZMQ_pub_PORT}"
    zmq_pub_socket.bind(zmq_pub_url)

    # Initiate BEV Place model
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")
    
    print('===> Building model')
    model = BEVPlace()
    resume_ckpt = opt.resume

    print("=> loading checkpoint '{}'".format(resume_ckpt))
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage) #, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    model = model.to(device)
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(resume_ckpt, checkpoint['epoch']))

    if cuda:
        model = nn.DataParallel(model)
        # model = model.to(device)
    
    # Instance init
    def __init__(self):
        pass
        
    @staticmethod
    def encodeImage(image):

        dataset = SingleImageDataset(image)
        dataloader = DataLoader(dataset, num_workers=1, batch_size=1, shuffle=False, pin_memory=BEVPlaceServer.cuda)

        BEVPlaceServer.model.eval()

        global_features = []
        with torch.no_grad():
            print('      Extracting Features of the input image')
            for iteration, (input, indices) in enumerate(dataloader, 1):
                # print(input) # this is the image shape
                if BEVPlaceServer.cuda:
                    input = to_cuda(input)
                batch_feature = BEVPlaceServer.model(input)
                global_features.append(batch_feature.detach().cpu().numpy())

        global_features = np.vstack(global_features)
        # print(global_features)
        print(f"      Calculated a feature vector of shape: {global_features.shape}")
        return global_features

    @staticmethod
    def process_image_message(image_msg):

        # Sample data array
        img = Image.frombytes('L', (image_msg.width, image_msg.height), image_msg.image_data)
        img = img.convert('RGB')

        # Display or save the image
        # img.show()
        # img.save('my_image.png')

        encode_start_time = time.time()
        global_features = BEVPlaceServer.encodeImage(img)
        encode_end_time = time.time()
        global_features_array = global_features.flatten()
        image_msg.description.extend(global_features_array)
        image_msg_bytes_str = image_msg.SerializeToString()
        BEVPlaceServer.zmq_pub_socket.send(image_msg_bytes_str)
        print(f"      Published image message back with descriptor")
        publish_end_time = time.time()

        BEVPlaceServer.num_images_processed += 1        
        encoding_time_taken = encode_end_time - encode_start_time
        publishing_time_taken = publish_end_time - encode_end_time
        print(f"{BEVPlaceServer.num_images_processed} Encoding,publishing took {encoding_time_taken*1000:.0f}, {publishing_time_taken*1000:.0f} ms.")

        pass

    @staticmethod
    def read_image_no_block():
        
        # Start subscription, read the messages, parse them and show them

        error_msg_read = False
        try:
            image_msg_str = BEVPlaceServer.zmq_sub_read_socket.recv(zmq.NOBLOCK)
            image_msg_read = True
            # print(f'Read: {image_msg_str}')

            # Parse the input message string
            image_msg = sensor_messages_pb2.ImageMessage()
            image_msg.ParseFromString(image_msg_str)
            # print(f'Read: {reporter_msg}')

            BEVPlaceServer.process_image_message(image_msg)

        except zmq.Again as e:
            print('INFO: No image message was read')
            pass


    @staticmethod
    def read_image_forever():
        
        # Start subscription, read the messages, parse them and show them
        while True:

            error_msg_read = False
            try:
                image_msg_str = BEVPlaceServer.zmq_sub_read_socket.recv()
                image_msg_read = True
                # print(f'Read: {image_msg_str}')

                # Parse the input message string
                image_msg = sensor_messages_pb2.ImageMessage()
                image_msg.ParseFromString(image_msg_str)
                # print(f'Read: {reporter_msg}')

                BEVPlaceServer.process_image_message(image_msg)

            except Exception as e:
                print(f"ERROR: {e}")
                pass
        
        pass

    @staticmethod
    def ShutDown():

        # Exit now
        print("BEV Place model is unloading")
        del BEVPlaceServer.model
        print("Server is shutting down")

        return


# Global scheduler object
msg_read_interval_sec = 0.5
time_scheduler = sched.scheduler(time.time, time.sleep)
last_msg_read_time = time.time()

# Repeated running using a timer
def repeatedly_run(_scheduler):
    global last_msg_read_time

    currtime = time.time()
    # print 'running ', math.floor(currtime-starttime)

    if (last_msg_read_time + msg_read_interval_sec <= currtime):
        # print '  monitoring processes ', math.floor(currtime-starttime)
        BEVPlaceServer.read_image_no_block()
        last_monitor_processes_time = currtime

    # reenter
    time_scheduler.enter(msg_read_interval_sec, 1, repeatedly_run, (_scheduler,))


# SigInt handler for exiting
def signal_handler(sig, frame):
    print('SIGINT received. Exiting...')
    BEVPlaceServer.ShutDown()
    sys.exit(0)

# Main
if __name__ == "__main__":

    args=parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    if args.periodic:
        time_scheduler.enter(0.5, 1, repeatedly_run, (time_scheduler,))
        time_scheduler.run()
    else:
        BEVPlaceServer.read_image_forever()
