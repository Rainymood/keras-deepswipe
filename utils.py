import numpy as np
import os 
import cv2
import re
import json
from dotmap import DotMap
import time
import argparse
import importlib

def create(cls):
    '''expects a string that can be imported as with a module.class name'''
    module_name, class_name = cls.rsplit(".",1)

    try:
        print('importing '+module_name)
        somemodule = importlib.import_module(module_name)
        print('getattr '+class_name)
        cls_instance = getattr(somemodule, class_name)
        print(cls_instance)
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

    return cls_instance

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict

def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.callbacks.tensorboard_log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), config.exp.name, "logs/")
    config.callbacks.checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/",time.localtime()), config.exp.name, "checkpoints/")
    return config

def crop_frame(frame, FRAME_WIDTH, FRAME_HEIGHT, crop_size):
    """Resizes and crops a single frame to crop_size.

    This function first rescales the video and then takes a crop_size by
    crop_size centre crop. Please note that we assume that FRAME_WIDTH >
    FRAME_HEIGHT otherwise there would be an if/else statement and we would have
    to change the scale. This is why we have the assertion.

    Parameters
    ----------
    frame : np.array
        Frame of size (frame_height, frame_width, 3)
    FRAME_WIDTH : int
        Width of frame in pixels.
    FRAME_HEIGHT : int
        Height of frame in pixels.
    crop_size : int
        Size of crop to take.

    Returns
    -------
    np.array
        Cropped frame of size (crop_size, crop_size, 3)
    """
    assert FRAME_WIDTH > FRAME_HEIGHT

    # Resize image (NOTE: We assume width > height, otherwise if ... statement.)
    scale = float(crop_size) / float(FRAME_HEIGHT)
    dim = (int(FRAME_WIDTH*scale+1), crop_size)
    frame = np.array(cv2.resize(np.array(frame), dim))

    # Take center crop (224x224 by default)
    crop_x = int((frame.shape[0] - crop_size) / 2)
    crop_y = int((frame.shape[1] - crop_size) / 2)
    cropped_frame = frame[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]

    return cropped_frame

def get_best_weights(config):
    """Returns path of .hfd5 file in 'experiments/' with the lowest val acc"""
    # Create list of all hdf5 paths
    list_of_days = [day for day in os.listdir('experiments/') if not day.startswith('.')]
    list_of_hdf5_paths = []
    for day in list_of_days:
        checkpoint_dir = os.path.join('experiments/', day, config.exp.name, 'checkpoints')
        list_of_hdf5_paths.extend([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.hdf5')])

    # get validation loss with some grep magic
    list_val_loss = []
    for path in list_of_hdf5_paths:
        decimal_match = re.search('[0-9]+\.[0-9]+', path) # ".../deepswipe-01-8.98.hdf5"
        if decimal_match.group() != None:
            list_val_loss.append(float(decimal_match.group()))

    # get path with lowest val loss
    min_index = list_val_loss.index(min(list_val_loss))
    lowest_val_hdf5_path = list_of_hdf5_paths[min_index]

    return lowest_val_hdf5_path
