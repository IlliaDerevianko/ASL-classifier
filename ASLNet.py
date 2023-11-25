# from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
import keras.backend as K


class ASLNet:
    activation = 'relu'
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first": 
            inputShape = (depth, height, width)
            chanDim = 1
        
        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation(ASLNet.activation))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation(ASLNet.activation))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((3, 3)))


        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation(ASLNet.activation))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation(ASLNet.activation))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(ASLNet.activation))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes)) 
        model.add(Activation("softmax"))

        return model


