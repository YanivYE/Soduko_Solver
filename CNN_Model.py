# import the necessary packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


def build(width, height, depth, classes):
    # initialize the model
    model = Sequential()
    input_shape = (height, width, depth)
    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(32, (5, 5), padding="same",
                     input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # first set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # second set of FC => RELU layers
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    # return the constructed network architecture
    return model

