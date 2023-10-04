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

N_TRAIN = 10000
N_TEST = 20


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')


def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = int.from_bytes(f.read(1), 'big')  # Convert bytes to integer
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels


def train_model():
    # Read and convert image data
    x_train = np.array(read_images(TRAIN_DATA_FILENAME, N_TRAIN))
    x_test = np.array(read_images(TEST_DATA_FILENAME, N_TEST))

    # Read labels
    y_train = read_labels(TRAIN_LABELS_FILENAME, N_TRAIN)
    y_test = read_labels(TEST_LABELS_FILENAME, N_TEST)


def main():
    train_model()


if __name__ == '__main__':
    main()
