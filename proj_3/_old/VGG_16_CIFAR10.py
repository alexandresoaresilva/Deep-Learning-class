#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import matplotlib.pyplot as plt
import numpy as np
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import pickle
import pandas as pd
from keras.datasets import cifar10


batch_size = 32 
num_classes = 10
epochs = 100

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255

def VGG_16(weights_path=None):
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=x_train.shape[1:]))
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,(3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,(3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,(3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3, 3)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    if weights_path:
        model.load_weights(weights_path, by_name=True)
    
    sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    

    return model


# In[ ]:


VGG_NN = VGG_16(weights_path='vgg16_weights.h5') #'vgg16_weights.h5')
VGG_NN.summary()
cnn = VGG_NN.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test),shuffle=True)

keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

# In[ ]:


plt.figure(0)
plt.plot(cnn.history['acc'],'r')
plt.plot(cnn.history['val_acc'],'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])
 
plt.figure(1)
plt.plot(cnn.history['loss'],'r')
plt.plot(cnn.history['val_loss'],'g')
plt.xticks(np.arange(0, 101, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])
 
plt.show()

