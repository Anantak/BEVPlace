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

import logging
from logging.handlers import TimedRotatingFileHandler

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
parser.add_argument('--listenPort', type=int, default=7776, help='port to use for listening for images')
parser.add_argument('--publishPort', type=int, default=7794, help='port to use for publishing back')
parser.add_argument("--periodic", type=bool, default=False, help="Run periodically at 2Hz with noblock")
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=4, help='Number of threads for each data loader to use')
parser.add_argument('--resume', type=str, default='checkpoints/checkpoint_paper_kitti.pth.tar', help='Path to load checkpoint from, for resuming training or testing.')

def setup_logging(log_file_path):
    """Sets up logging to both file and console with daily rotation and 15-day retention."""

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Adjust log level as needed

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create file handler
    file_handler = TimedRotatingFileHandler(log_file_path, when='midnight', interval=1, backupCount=14)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)  # Adjust log level for file

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Static BEV Place server
class BEVPlaceServer:

    # Static variables of the class
    opt = parser.parse_args()

    log_file = "./logs/bev_place_server.log"
    logger = setup_logging(log_file)

    # How many images have been processed
    num_images_processed = 0

    # ZMQ socket objects
    zmq_context = zmq.Context()
    ZMQ_sub_PORT = opt.listenPort
    zmq_sub_read_socket = zmq_context.socket(zmq.SUB)
    zmq_sub_read_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    # zmq_sub_read_socket.setsockopt(zmq.CONFLATE, 1)
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
    
    logger.info('Building model')
    model = BEVPlace()
    resume_ckpt = opt.resume

    logger.info("Loading checkpoint '{}'".format(resume_ckpt))
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage) #, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    model = model.to(device)
    logger.info("Loaded checkpoint '{}' (epoch {})"
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
            BEVPlaceServer.logger.info('      Extracting Features of the input image')
            for iteration, (input, indices) in enumerate(dataloader, 1):
                # print(input) # this is the image shape
                if BEVPlaceServer.cuda:
                    input = to_cuda(input)
                batch_feature = BEVPlaceServer.model(input)
                global_features.append(batch_feature.detach().cpu().numpy())

        global_features = np.vstack(global_features)
        # print(global_features)
        BEVPlaceServer.logger.info(f"      Calculated a feature vector of shape: {global_features.shape}")
        return global_features

    @staticmethod
    def process_image_message(sensor_msg):

        # Sample data array
        img = Image.frombytes('L', (sensor_msg.image_msg.width, sensor_msg.image_msg.height), sensor_msg.image_msg.image_data)
        img = img.convert('RGB')

        # Display or save the image
        # img.show()
        # img.save('my_image.png')

        encode_start_time = time.time()
        global_features = BEVPlaceServer.encodeImage(img)
        encode_end_time = time.time()

        global_features_array = global_features.flatten()
        
        sensor_msg.image_msg.description.extend(global_features_array)
        sensor_msg.header.type = "Descriptor"

        sensor_msg_bytes_str = sensor_msg.SerializeToString()
        BEVPlaceServer.zmq_pub_socket.send(sensor_msg_bytes_str)
        BEVPlaceServer.logger.info(f"      Published image message back with descriptor")
        publish_end_time = time.time()

        encoding_time_taken = encode_end_time - encode_start_time
        publishing_time_taken = publish_end_time - encode_end_time
        BEVPlaceServer.logger.info(f"      Encoding,publishing took {encoding_time_taken*1000:.0f}, {publishing_time_taken*1000:.0f} ms.")

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

                BEVPlaceServer.num_images_processed += 1        
                BEVPlaceServer.logger.info(f"Received message {BEVPlaceServer.num_images_processed}")
                BEVPlaceServer.logger.info(f"      Processing message as a BEVImage sensor message")

                # Interpret the message as a protocol buffer sensor message
                sensor_msg = sensor_messages_pb2.SensorMsg()
                sensor_msg.ParseFromString(image_msg_str)

                # Get the header message
                header_msg = sensor_msg.header
                if (header_msg.type != "BEVImage"):
                    BEVPlaceServer.logger.error("ERROR: message type is not BEVImage. Got {}".format(header_msg.type))
                    return

                # Check if there is a image_msg
                if (not sensor_msg.HasField('image_msg')):
                    BEVPlaceServer.logger.error("ERROR: message does not have a image_msg")
                    return

                # Get the image payload
                image_msg = sensor_msg.image_msg
                BEVPlaceServer.logger.info("      Got an image with dim {}x{}x{}".format(image_msg.width, image_msg.height, image_msg.depth))
                
                if (len(image_msg.image_data) < 1):
                    BEVPlaceServer.logger.error("ERROR: len(image_msg.image_data) < 1. So not using this message.")
                    return
                
                BEVPlaceServer.process_image_message(sensor_msg)

            except Exception as e:
                BEVPlaceServer.logger.error(f"ERROR: {e}")
                pass
        
        pass


    @staticmethod
    def ShutDown():

        # Exit now
        BEVPlaceServer.logger.info("BEV Place model is unloading")
        del BEVPlaceServer.model
        BEVPlaceServer.logger.info("Server is shutting down")

        return



# SigInt handler for exiting
def signal_handler(sig, frame):
    BEVPlaceServer.logger.info('SIGINT received. Exiting...')
    BEVPlaceServer.ShutDown()
    sys.exit(0)

# Main
if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)

    BEVPlaceServer.read_image_forever()
