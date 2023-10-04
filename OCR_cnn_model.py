#!/usr/bin/env python

# TensorFlow's Keras API
from keras import layers, models
from keras.preprocessing import image
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

DATA_DIR = 'data/'
TEST_DATA_FILENAME = DATA_DIR + 't10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + 'train-labels.idx1-ubyte'
