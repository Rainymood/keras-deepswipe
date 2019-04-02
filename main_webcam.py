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
import datetime

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
    # model.load_weights(weights_path)

    print('Opening VideoCapture')

    cv2.namedWindow("Preview")
    cap = cv2.VideoCapture(0)

    crop_size = 224

    ACTIVE_LEN = 10
    ACTIVE_WIDTH = crop_size # todo: change to crop size
    ACTIVE_HEIGHT = crop_size # todo: change to crop size

    active_frames = np.zeros((ACTIVE_LEN, ACTIVE_HEIGHT, ACTIVE_WIDTH, 3))

    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    NUM_RGB_CHANNELS = 3
    NUM_CLASSES = 3

    while True:
        rval, frame = cap.read() # read in frame
        frame = crop_frame(frame, FRAME_WIDTH, FRAME_HEIGHT, crop_size) # crop frame
        frame_reshaped = np.expand_dims(frame, axis=0) # reshape frame

        if frame is not None:
            cv2.imshow("preview", frame) # print reshaped frame

        active_frames = np.concatenate((active_frames, frame_reshaped), axis=0) # add frame
        active_frames = active_frames[1:, :, :, :] # pop first frame

        now = datetime.datetime.now()
        pred = model.predict(np.expand_dims(active_frames, axis=0)) # add batch_size=1 dimension

        print(str(now), " | ", "Prediction: ", str(pred))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break

    # Running into this error? You forgot to release the capture device, dummy.
    # OpenCV: error in [AVCaptureDeviceInput initWithDevice:error:]
    # OpenCV: Cannot Use FaceTime HD Camera (Built-in)
    # OpenCV: camera failed to properly initialize!
    cap.release()

if __name__ == '__main__':
    main()




