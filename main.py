"""This script performs the main machine learning pipeline of the model. It
first reads the arguments from the .json config file, then creates the
data_loader, the model, and finally the trainer. It then trains the model
according to the specifications in the .json config file."""

from __future__ import print_function
from data_loader.simple_mnist_data_loader import SimpleMnistDataLoader
from models.simple_mnist_model import SimpleMnistModel
from trainers.simple_mnist_trainer import SimpleMnistModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

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

    print('Create the data generator.')
    data_loader = SimpleMnistDataLoader(config)

    print('Create the model.')
    model = SimpleMnistModel(config)

    print('Create the trainer')
    trainer = SimpleMnistModelTrainer(model.model,
                                      data_loader.get_train_data(), config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
