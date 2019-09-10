# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:27:37 2019

@author: shael
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np

from tensorflow.keras.callbacks import TensorBoard
NAME = "DogvCatCNN"

X= pickle.load(open("X.pickle","rb"))
y= pickle.load(open("y.pickle","rb"))
    
y= np.array(y)#1 = greyscale, 3 = colour
    
X=X/255.0
model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))# use either activation layer or pooling layer, using rectify linear
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))# use either activation layer or pooling layer, using rectify linear
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))


tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss="binary_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X,y,batch_size=32,epochs=5,validation_split=0.3,
          callbacks=[tensorboard])

model.save("Puss-or-Ruff-CNN2.model")
