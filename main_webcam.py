"""This script performs the main webcam pipeline. It first loads in the
latest model, creates an OpenCV webcam object and then predicts on
every (other) frame"""

from __future__ import print_function
from data_loader.deepswipe_data_loader import DeepSwipeDataGenerator
from models.deepswipe_model import DeepSwipeModel
from trainers.deepswipe_trainer import DeepSwipeTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
import os
import numpy as np 
import cv2
import re

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

def main():
    """Runs the main deep learning pipeline."""
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print('Missing or invalid arguments.')
        exit(0)

    weights_path = get_best_weights(config)

    print('Create the model.')
    model = DeepSwipeModel(config)

    print('Loading weights.')
    model.load_weights(weights_path)
    
if __name__ == '__main__':
    main()

