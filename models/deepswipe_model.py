from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from models.i3d_inception import Inception_Inflated3d

class DeepSwipeModel(BaseModel):
    def __init__(self, config):
        super(DeepSwipeModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        # TODO: add this to config json
        NUM_FRAMES = 10
        FRAME_HEIGHT = 224
        FRAME_WIDTH = 224
        NUM_RGB_CHANNELS = 3
        NUM_CLASSES = 3

        print('Creating Inception_Inflated3d model.')
        rgb_model = Inception_Inflated3d(
                        include_top=False, # no prediction layer
                        weights='rgb_kinetics_only',
                        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                        classes=NUM_CLASSES)

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
