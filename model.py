
#importing libraries
import tensorflow as tf

import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers,Model,losses


#loading data
directory="data"
dirs=['Normal 59','RP 121']
x=[]
y=[]
for i in range(2):
  directory1=directory+"/"+dirs[i]
  entries=os.listdir(directory1)
  for entry in entries:
    img_path=os.path.join(directory1,entry)
    image=cv2.imread(img_path)
    image=cv2.resize(image,(224,224))
    x.append(image)
    y.append(i)

x=np.array(x)
y=np.array(y)


#shuffling data
rng = np.random.default_rng(seed=42)
shuffled_indices = rng.permutation(len(x))
x_shuffled = x[shuffled_indices]
y_shuffled = y[shuffled_indices]


#splitting data into train and test data
n1=int(len(x_shuffled)*0.8)
x_train=x_shuffled[:n1]
x_test=x_shuffled[n1:]
y_train=y_shuffled[:n1]
y_test=y_shuffled[n1:]


#Getting base model
base_model=tf.keras.applications.ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
for layer in base_model.layers:
  layer.trainable = False

#add top layers
x = layers.Flatten()(base_model.output)
x = layers.Dense(1000, activation='relu')(x)
predictions = layers.Dense(2, activation = 'softmax')(x)

#combine base model and top layers
head_model = Model(inputs = base_model.input, outputs = predictions)
head_model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

history = head_model.fit(x_train, y_train, batch_size=64, epochs=50)
print(head_model.evaluate(x_test, y_test))

head_model.save('head_model.h5')