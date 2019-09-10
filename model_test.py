# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 22:37:09 2019

@author: shael
"""

import cv2
import tensorflow as tf
CATEGORIES =["Dog","Cat"]
def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

model = tf.keras.models.load_model("DogvCatCNN2.model")
img_to_guess=([prepare('dog.jpg')])
prediction = model.predict(img_to_guess)
print("It's a " + CATEGORIES[int(prediction[0][0])])
