# import the necessary packages
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt

import CNN_Model
from keras.optimizers import Adam
from keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer

EVALUATE = False

# initialize the initial learning rate, number of epochs to train
# for, and batch size
INIT_LR = 1e-3
EPOCHS = 10
BS = 128

# grab the MNIST dataset
print("[INFO] accessing MNIST...")
((X_train, y_train), (X_test, y_test)) = mnist.load_data()

# add a channel (i.e., grayscale) dimension to the digits
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

le = LabelBinarizer()

# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)
model = CNN_Model.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=BS,
    epochs=EPOCHS,
    verbose=1)

if EVALUATE:
    # evaluate the network
    print("[INFO] evaluating network...")

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

# Evaluate the model, returns the loss value & metrics values for the model in test mode.
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)

# Save the trained model
print("[INFO] saving model..")
model.save('my_model.keras')
