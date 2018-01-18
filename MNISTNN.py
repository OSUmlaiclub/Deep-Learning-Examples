from __future__ import print_function

import random

import time

random.seed(time.time)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

print("here")
# Load the data, shuffled and split between train and test sets (x_train and y_rain)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_array = x_test

## For our purposes, these images are just a vector of 784 inputs, so let's convert
x_train = x_train.reshape(len(x_train), 28*28)
x_test = x_test.reshape(len(x_test), 28*28)

## Keras works with floats, so we must cast the numbers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

## Normalize the inputs so they are between 0 and 1
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)

### Keras model

model_2 = Sequential()
model_2.add(Dense(400, activation='relu', input_shape=(784,)))
model_2.add(Dropout(0.4))
model_2.add(Dense(300, activation='relu'))
model_2.add(Dropout(0.4))
model_2.add(Dense(10, activation='softmax'))


#Compile the model
learning_rate = 0.001
model_2.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['accuracy'])
#model_2.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

batch_size = 128
epochs = 20
history = model_2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model_2.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#plot_loss_accuracy(history)


#Show the number and the prediction
def Make_Preds():
    fig=plt.figure(figsize=(10, 10))
    rows = 5
    offset = random.randint(1,1000)
    for i in range(1, rows +1):
        single_test = np.array([x_test[i + offset]])
        output = model_2.predict(single_test).round()
        mystring = "Prediction:" + str(np.nonzero(output[0])[0][0])
        #print("   Real output:", np.nonzero(y_test[i])[0][0])
        axi = fig.add_subplot(rows, 1, i)
        axi.text(1.5,0.5, mystring, size=12, ha="center", transform=axi.transAxes)
        axi.xaxis.set_visible(False)
        axi.yaxis.set_visible(False)
        plt.imshow(image_array[i+offset])
    plt.show()

#while loop creates popups with 5 images
while(1):
    input("Press enter to get a new set of predictions")
    Make_Preds()
