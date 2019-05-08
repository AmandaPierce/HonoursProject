import numpy as nump
import cv2
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import pandas as pd
K.set_image_dim_ordering('th')


def train_cnn():

    # Initialize random number generator to constant to be able to reproduce results
    seed = 7
    nump.random.seed(seed)

    raw_data_train = pd.read_csv("./emnist-balanced-train.csv")
    raw_data_test = pd.read_csv("./emnist-balanced-test.csv")

    # load the train and test set
    # x is images
    # y is labels
    X_train = raw_data_train.values[:, 1:]
    y_train = raw_data_train.values[:, 0]

    X_test = raw_data_test.values[:, 1:]
    y_test = raw_data_test.values[:, 0]
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Data sets are structured as 3D arrays with instance, image width and height
    # For a multi-layer perceptron model we must reduce the images down into a vector of pixels.
    # In this case the 28×28 sized images will be 784 pixel input values.
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype(
        'float32')  # reducing memory requirements
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

    # Scale values to range 0 - 1
    X_train = X_train / 255
    X_test = X_test / 255

    # Use one hot encoding for multi-class classification.
    # Transforms the vector of class integers into a binary matrix.
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    model = baseline_model(num_classes)

    # Fit the model
    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=1, batch_size=200)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    accuracy = 100*score[1]
    print('Test accuracy: %.4f%%' % accuracy)
    model.save('my_model.h5')

    # img = cv2.imread('images/cropped/training_1.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = (nump.expand_dims(img, 0))

    # predictions_single = model.predict(img)

    # print(predictions_single)


def baseline_model(num_classes):
    model = Sequential()
    # Convolution layer
    # 32 feature functions
    # 5 by 5
    # rectifier activation function
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    # pooling layer
    # 2 by 2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution layer
    model.add(Conv2D(15, (3, 3), activation='relu'))
    # pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # dropout layer
    model.add(Dropout(0.4))
    # Transforms image from 2D to 1D array
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # Fully connected layer with 128 neurons
    model.add(Dense(128, activation='relu'))
    # Fully connected layer with 50 neurons
    model.add(Dense(50, activation='relu'))
    # Softmax activation function
    # Is used on the output layer to turn the outputs
    # into probability-like values and allow one class to be selected as the model’s output prediction.
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    # Logarithmic loss
    # ADAM gradient descent algorithm is used to learn the weights
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model
