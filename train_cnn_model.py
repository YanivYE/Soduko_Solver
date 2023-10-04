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
RESOLUTION = 28

EVALUATION = False


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


def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model():
    # Read and convert image data
    x_train = np.array(read_images(TRAIN_DATA_FILENAME, N_TRAIN))
    x_test = np.array(read_images(TEST_DATA_FILENAME, N_TEST))

    # Read labels
    y_train = read_labels(TRAIN_LABELS_FILENAME, N_TRAIN)
    y_test = read_labels(TEST_LABELS_FILENAME, N_TEST)

    # Assuming images are 28x28 pixels
    input_shape = (RESOLUTION, RESOLUTION, 1)
    num_classes = 10  # Number of digits

    # Preprocess data for CNN
    x_train = x_train.reshape(-1, RESOLUTION, RESOLUTION, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, RESOLUTION, RESOLUTION, 1).astype('float32') / 255.0

    # Convert labels to one-hot encoded vectors - binary class matrix
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Build the CNN model
    model = build_cnn_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model, epochs - dataset iterations
    fit_model = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    if EVALUATION:
        evaluate_model(fit_model)

    # Evaluate the model, returns the loss value & metrics values for the model in test mode.
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_accuracy)

    # Save the trained model
    model.save('my_model.keras')
    print("Model trained and saved.")


def evaluate_model(fit_model):
    plt.figure(1)
    plt.plot(fit_model.history['loss'])
    plt.plot(fit_model.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.figure(2)
    plt.plot(fit_model.history['accuracy'])
    plt.plot(fit_model.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.show()


def main():
    train_model()


if __name__ == '__main__':
    main()
