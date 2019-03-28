from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist
import keras 
import numpy as np 

class DeepSwipeDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DeepSwipeDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.X_train = self.X_train.reshape((-1, 28 * 28))
        self.X_test = self.X_test.reshape((-1, 28 * 28))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

class DeepSwipeDataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self,
                 config,
                 list_IDs,
                 labels,
                 shuffle=True):
        # 'Initialization'
        self.config = config # because we don't super(...).__init__(config)
        self.dim = tuple(self.config.trainer.dim)
        self.batch_size = self.config.trainer.batch_size
        self.n_channels = self.config.trainer.n_channels
        self.n_classes = self.config.trainer.n_classes
        self.shuffle = shuffle

        self.labels = labels
        self.list_IDs = list_IDs
        self.on_epoch_end()

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

        # print("Shape of X:", X.shape)
        # print("Shape of y:", y.shape)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # print("Loading in {}.npy".format(ID))
            tmp = np.load('data/' + ID + '.npy')
            # print("Shape of {}.npy: {}".format(ID, tmp.shape))
            tmp = tmp[:, :10, :, :, :] # TODO: HACK! How to fix: make all data 10 frames long 

            # print("New shape of {}.npy: {}".format(ID, tmp.shape))

            # X[i,] = np.load('data/' + ID + '.npy')
            X[i,] = tmp

            # print("ID:", ID)
            # print("i:", i)

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
