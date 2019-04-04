"""This script performs the main machine learning pipeline of the model. It
first reads the arguments from the .json config file, then creates the
data_loader, the model, and finally the trainer. It then trains the model
according to the specifications in the .json config file."""

from __future__ import print_function
from data_loader.deepswipe_data_loader import DeepSwipeDataGenerator
from models.deepswipe_model import DeepSwipeModel
from trainers.deepswipe_trainer import DeepSwipeTrainer
import os
import numpy as np
from utils import get_args, process_config, create_dirs

def main():
    """Runs the main deep learning pipeline."""
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print('Missing or invalid arguments.')
        exit(0)

    print('Create experiment directories.')
    create_dirs([
        config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir
    ])

    # TODO Refactor this
    print('Create partitions and labels.')
    partition = {}
    all_ids = [filename.split('.')[0] for filename in os.listdir('data') if filename.endswith('.npy')]
    partition['train'] = all_ids[50:]
    partition['validation'] = all_ids[:50]

    labels_ids = [filename.split('.')[0] for filename in os.listdir('data') if filename.endswith('.npy')]
    labels_values = [1 if 'swipe_positive_right' in filename \
                     else -1 if 'swipe_positive_left' in filename \
                     else 0 for filename in os.listdir('data') if filename.endswith('.npy')]
    labels = dict(zip(labels_ids, labels_values))

    print('Create the training and validation data generators.')
    training_generator = DeepSwipeDataGenerator(config, partition['train'],labels)
    validation_generator = DeepSwipeDataGenerator(config, partition['validation'], labels)
    data_generator = (training_generator, validation_generator)

    print('Create the model.')
    model = DeepSwipeModel(config)

    print('Create the trainer')
    trainer = DeepSwipeTrainer(model.model, data_generator, config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()

