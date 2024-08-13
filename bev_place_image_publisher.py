#!/usr/bin/env python3

# Addressing the Protobuf issue
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python" 

import sys
import signal
import argparse
import zmq
import datetime 
import time
import sched

from termcolor import cprint

from PIL import Image

sys.path.append('/home/ubuntu/Anantak/Admin/messaging');
import sensor_messages_pb2

# Create the publisher for image receiver.
# We use the map-editor error reporter parameters
#     name: "MapEditorErrorsReporter"
#     endpoint: "tcp://127.0.0.1:51705"
#     subject: ""

# message ImageMessage {
# required int32 camera_num = 101;
# required int32 height = 102;
# required int32 width = 103;
# required int32 depth = 104;
# required bytes image_data = 200;
# repeated float posn_data = 300;
# optional int64 timestamp_us = 400;
# }

class ImagePublisher:

    # Static variables of the class
    # ZMQ socket objects
    ZMQ_MapEditorErrorsReporter_PORT = 7776
    zmq_context = zmq.Context()
    zmq_MapEditorErrorsReporter_socket = zmq_context.socket(zmq.PUB)
    zmq_MapEditorErrorsReporter_url = f"tcp://127.0.0.1:{ZMQ_MapEditorErrorsReporter_PORT}"
    zmq_MapEditorErrorsReporter_socket.bind(zmq_MapEditorErrorsReporter_url)

    owner_names = {}
    owner_msgs = {}    
    owner_summing_msgs = {}

    # Instance init
    def __init__(self):
        pass
    
    @staticmethod
    def transmit_image(image_path):
        
        try:
          with Image.open(image_path) as img:
            # img.show()
            img_bytes = img.tobytes()
            now_ts_us = int(datetime.datetime.now().timestamp()*1000000)

            sensor_msg = sensor_messages_pb2.SensorMsg()

            sensor_msg.header.type = "BEVImage"
            sensor_msg.header.timestamp = now_ts_us
            sensor_msg.header.recieve_timestamp = now_ts_us
            sensor_msg.header.send_timestamp = now_ts_us

            image_msg = sensor_msg.image_msg
            image_msg.camera_num = -1
            image_msg.height = 151
            image_msg.width = 151
            image_msg.depth = 1
            image_msg.image_data = img_bytes
            image_msg.timestamp_us = now_ts_us

            image_msg_bytes_str = sensor_msg.SerializeToString()
            ImagePublisher.zmq_MapEditorErrorsReporter_socket.send(image_msg_bytes_str)
            print(f"published {image_path}")

        except FileNotFoundError:
          print("Image file not found:", image_path)
        except Exception as e:
          print("Error converting image to bytes:", e)        


# SigInt handler for exiting
def signal_handler(sig, frame):
    print('SIGINT received. Exiting...')
    sys.exit(0)


# Global scheduler object
msg_pub_interval_sec = 2.0
time_scheduler = sched.scheduler(time.time, time.sleep)
last_msg_read_time = time.time()

# Repeated running using a timer
def repeatedly_run(_scheduler, image_path):
    global last_msg_read_time

    currtime = time.time()
    # print 'running ', math.floor(currtime-starttime)

    if (last_msg_read_time + msg_pub_interval_sec <= currtime):
        # print '  monitoring processes ', math.floor(currtime-starttime)
        ImagePublisher.transmit_image(image_path)
        last_monitor_processes_time = currtime

    # reenter
    time_scheduler.enter(msg_pub_interval_sec, 1, repeatedly_run, (_scheduler,image_path))


if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument("--image", help="protoc filename to parse", type=str, default='')
    args=parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    if (args.image == ''):
       print(f"ERROR: Need an image to publish. Provide with --image=image/file/path")
       sys.exit(1)

    time_scheduler.enter(msg_pub_interval_sec, 1, repeatedly_run, (time_scheduler,args.image))
    time_scheduler.run()

    sys.exit(0)


