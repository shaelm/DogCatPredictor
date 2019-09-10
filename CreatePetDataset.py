# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 08:47:40 2019

@author: shael
"""



import tensorflow as tf
import random 
import numpy as np 
import matplotlib.pyplot as plt
import pathlib
import IPython.display as display
import os
import pathlib
import pickle

import cv2

DATADIR = ".../PetImages"
IMG_SIZE=50
CATEGORIES=["Dog","Cat"]
training_data=[]

def create_training_data():
    for counter,category in enumerate(CATEGORIES):
        path=os.path.join(DATADIR,category)
        class_num=counter
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                resized_photos=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([resized_photos,class_num])
                
            except Exception as e:
                pass
            

create_training_data()
random.shuffle(training_data)# to make sure images are not dog dog dog cat cat etc

X=[]
y=[]

for features,label in training_data:
    X.append(features)
    y.append(label)
    
X= np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)#1 = greyscale, 3 = colour

pickle_out=open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in= open("X.pickle","rb")
X= pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)






