#!/usr/bin/env python

from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

RESOLUTION = 28
EVALUATION = True  # Set this to True to evaluate the model


def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(x_train.reshape(-1, RESOLUTION, RESOLUTION, 1))

    # Data Preprocessing
    x_train = x_train.reshape((-1, RESOLUTION, RESOLUTION, 1)).astype('float32') / 255
    x_test = x_test.reshape((-1, RESOLUTION, RESOLUTION, 1)).astype('float32') / 255

    # One-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Model
    input_shape = (RESOLUTION, RESOLUTION, 1)
    num_classes = 10
    model = build_cnn_model(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training
    history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                        epochs=10, validation_data=(x_test, y_test))

    if EVALUATION:
        evaluate_model(history)

    # Evaluate the model, returns the loss value & metrics values for the model in test mode.
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_accuracy)

    # Save the trained model
    model.save('my_model.keras')
    print("Model trained and saved.")


def evaluate_model(history):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.show()


def main():
    train_model()


if __name__ == '__main__':
    main()
