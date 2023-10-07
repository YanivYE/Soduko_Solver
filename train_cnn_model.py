#!/usr/bin/env python

# TensorFlow's Keras API
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

RESOLUTION = 28
EVALUATION = False

def build_cnn_model(input_shape, num_classes):
    num_of_filters = 60
    size_of_filter_1 = (5, 5)
    size_of_filter_2 = (3, 3)
    size_of_pool = (2, 2)
    num_of_nodes = 500
    model = models.Sequential([
        layers.Conv2D(num_of_filters, size_of_filter_1, activation='relu', input_shape=input_shape),
        layers.Conv2D(num_of_filters, size_of_filter_1, activation='relu'),
        layers.MaxPooling2D(pool_size=size_of_pool),
        layers.Conv2D(num_of_filters // 2, size_of_filter_2, activation='relu'),
        layers.Conv2D(num_of_filters // 2, size_of_filter_2, activation='relu'),
        layers.MaxPooling2D(pool_size=size_of_pool),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(num_of_nodes, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Read and convert image data
    x_train = x_train.reshape((60000, RESOLUTION, RESOLUTION, 1))
    x_train = x_train.astype("float32") / 255

    x_test = x_test.reshape((10000, RESOLUTION, RESOLUTION, 1))
    x_test = x_test.astype('float32') / 255

    # Assuming images are 28x28 pixels
    input_shape = (RESOLUTION, RESOLUTION, 1)
    num_classes = 10  # Number of digits

    # Convert labels to one-hot encoded vectors - binary class matrix
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Build the CNN model
    model = build_cnn_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model, epochs - dataset iterations
    fit_model = model.fit(x_train, y_train, epochs=10, batch_size=32,
                          validation_data=(x_test, y_test))

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
