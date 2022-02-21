from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import struct
import os
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#optimize GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
      tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6024)])
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')  
      print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
      print(e)

# number of possible label values
np.random.seed(5)
image_size=(48, 48)
num_classes=9
batch_size=128
#image argumentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.2, # Randomly zoom image 
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

#image preprocessing
test_it= datagen.flow_from_directory("color_dataset/test/", class_mode="categorical", batch_size=128, target_size=image_size, shuffle=True, seed=42)
train_it=datagen.flow_from_directory("color_dataset/train/", class_mode="categorical", batch_size=128, target_size=image_size, shuffle=True, seed=42)
valid_it=datagen.flow_from_directory("color_dataset/valid/", class_mode="categorical", batch_size=128, target_size=image_size, shuffle=True, seed=42)
# Initialising the CNN
model = Sequential()

# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Fully connected layer 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

epochs=1000
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history=model.fit_generator(train_it, steps_per_epoch=74, epochs=epochs,validation_data=test_it,validation_steps=5, callbacks=callbacks_list)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

def draw(history):

  plt.figure(figsize=(20,10))
  plt.subplot(1, 2, 1)
  plt.suptitle('Optimizer : Adam', fontsize=10)
  plt.ylabel('Loss', fontsize=16)
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.legend(loc='upper right')

  plt.subplot(1, 2, 2)
  plt.ylabel('Accuracy', fontsize=16)
  plt.plot(history.history['acc'], label='Training Accuracy')
  plt.plot(history.history['val_acc'], label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.show()

draw(history)