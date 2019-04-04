"""This script performs the main webcam pipeline. It first loads in the
latest model, creates an OpenCV webcam object and then predicts on
every (other) frame"""

from __future__ import print_function
from data_loader.deepswipe_data_loader import DeepSwipeDataGenerator
from models.deepswipe_model import DeepSwipeModel
from trainers.deepswipe_trainer import DeepSwipeTrainer
import os
import numpy as np
import cv2
import re
import datetime
from utils2 import * 

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
    model.model.load_weights(weights_path)

    print('Opening VideoObject')

    cv2.namedWindow("Preview")
    cap = cv2.VideoCapture(0)

    crop_size = 224

    ACTIVE_LEN = 10
    ACTIVE_WIDTH = crop_size # todo: change to crop size
    ACTIVE_HEIGHT = crop_size # todo: change to crop size

    active_frames = np.zeros((ACTIVE_LEN, ACTIVE_HEIGHT, ACTIVE_WIDTH, 3))

    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        rval, frame = cap.read() # read in frame
        frame = crop_frame(frame, FRAME_WIDTH, FRAME_HEIGHT, crop_size) # crop frame
        frame_reshaped = np.expand_dims(frame, axis=0) # reshape frame

        if frame is not None:
            cv2.imshow("preview", frame) # print reshaped frame

        active_frames = np.concatenate((active_frames, frame_reshaped), axis=0) # add frame
        active_frames = active_frames[1:, :, :, :] # pop first frame

        now = datetime.datetime.now()
        input_video = np.expand_dims(active_frames, axis=0)
        pred = model.model.predict(input_video) # add batch_size=1 dimension

        print(str(now), " | ", "Prediction: ", str(pred))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break

    cap.release() # prevents error in [AVCaptureDeviceInput initWithDevice:error:]

if __name__ == '__main__':
    main()




