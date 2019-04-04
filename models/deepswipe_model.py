from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from models.i3d_inception import Inception_Inflated3d

class DeepSwipeModel(BaseModel):
    def __init__(self, config):
        super(DeepSwipeModel, self).__init__(config)

        # read in config
        self.input_shape = tuple(self.config.trainer.dim) + (self.config.trainer.n_channels,) # add two tuples, json doesnt support tuples
        self.n_classes = self.config.trainer.n_channels

        print('Shape of input_size:', self.input_shape)

        # build model
        print('Building the model.')
        self.build_model()

    def build_model(self):
        print('Creating Inception_Inflated3d model.')
        rgb_model = Inception_Inflated3d(
                        include_top=False, # no prediction layer
                        weights='rgb_kinetics_only',
                        input_shape=self.input_shape,
                        classes=self.n_classes)

        # Load in old model
        print('Loading in old model.')
        self.model = Sequential()
        self.model.add(rgb_model)

        for layer in self.model.layers:
            layer.trainable = False

        print('Extending old model with new layers on top.')
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(3, activation='softmax'))

        print('Compiling model. ')
        self.model.compile(optimizer=self.config.model.optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                     )

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

