from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist
import keras
import numpy as np
import os

class DeepSwipeDataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, config, shuffle=True):
        # 'Initialization'
        self.config = config # because we don't super(...).__init__(config)
        self.dim = tuple(self.config.trainer.dim)
        self.batch_size = self.config.trainer.batch_size
        self.n_channels = self.config.trainer.n_channels
        self.n_classes = self.config.trainer.n_classes
        self.shuffle = shuffle

        self.list_IDs = {} # partition
        self.labels = {}
        self.__make_partition()
        self.on_epoch_end()

    def __make_partition(self):
        """Sets list_IDs/partition and labels"""
        all_ids = [filename.split('.')[0] for filename in os.listdir('./data') if filename.endswith('.npy')]
        self.list_IDs['train'] = all_ids[50:]
        self.list_IDs['validation'] = all_ids[:50]

        labels_ids = [filename.split('.')[0] for filename in os.listdir('./data') if filename.endswith('.npy')]
        labels_values = [1 if 'swipe_positive_right' in filename else -1 if 'swipe_positive_left' in filename else 0 for filename in os.listdir('data') if filename.endswith('.npy')]
        self.labels = dict(zip(labels_ids, labels_values))

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # print("indexes:", indexes)

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            tmp = np.load('data/' + ID + '.npy')
            tmp = tmp[:, :10, :, :, :] # TODO: Fix. Make all data 10 frames long
            X[i,] = tmp

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
