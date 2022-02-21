import numpy as np
import pandas as pd
import keras
import struct
import cv2
import os
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, ZeroPadding2D, Lambda, Flatten, Dropout, Activation, AveragePooling2D, Input, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from tensorflow.keras.applications import MobileNet
import traceback
import contextlib
from PIL import Image

#에러출력을 위한 헬퍼 함수
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('기대하는 예외 발생 \n  {}:'.format(error_class))
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('{}를 기대했지만 아무런 에러도 발생되지 않았습니다!'.format(
        error_class))

#optimise GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
      tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
  except RuntimeError as e:
      print(e)

np.random.seed(5)
image_size=(160,160)
num_classes=9

#class list:
class_list=sorted(["black", "blue", "brown", "green", "pink", "red", "silver", "white", "yellow"])

#image augmentation:

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.2, # Randomly zoom image 
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

test_it= datagen.flow_from_directory("color_dataset/test/", class_mode="categorical", batch_size=128, target_size=image_size)
train_it=datagen.flow_from_directory("color_dataset/train/", class_mode="categorical", batch_size=128, target_size=image_size)
valid_it=datagen.flow_from_directory("color_dataset/valid/", class_mode="categorical", batch_size=128, target_size=image_size)

def trained_model():
    #base model
    base_model=tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet', pooling=max)
    base_model.trainable=False
    #base_model.summary()

    #model
    model=Sequential([
        base_model,
        Flatten(),
        Dense(units=4096, activation='relu'),
        Dropout(0.5),
        Dense(units=4096, activation='relu'),
        Dropout(0.5),
        Dense(units=num_classes, activation='softmax')
    ]
    )

    epochs=1
    batch_size=128
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    filepath = 'color_weights.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    #model.fit_generator(train_it, steps_per_epoch=23, validation_data=test_it, callbacks=callbacks_list, verbose=1, epochs=epochs)
    #model.save('color_model_mnet.h5')
    #tf.saved_model.save(model, "D:\weblivestream")
    #loaded=tf.saved_model.load("D:\weblivestream")

    #model1=tf.keras.models.load_model('color_model_mnet.h5')
    #predict=model1.predict(image, verbose=1)
    return model

    #print("Evaluate")
    #scores= model.evaluate_generator(valid_it)
    #print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
    #print(model.metrics_names)

def real_model(img):
    model1=load_model("color_model_mnet.h5")
    predict=model1.predict(img)
    return predict

#@tf.function(experimental_follow_type_hints=True)
def preprocess(img):
    image=cv2.resize(img, image_size)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = image.astype("float") / 255.0
    #image = np.expand_dims(image, axis=0)
    rgb_tensor=tf.convert_to_tensor(image, dtype=tf.float32)
    #rgb_tensor=tf.cast(image, tf.float32)
    #rgb_tensor=tf.divide(image, 255)
    rgb_tensor=tf.expand_dims(rgb_tensor, 0)
    return rgb_tensor

def postprocess(result):
    class_prob=result
    class_index=np.argmax(class_prob)
    class_label=class_list[class_index]
    return class_label

def get_args():
    parser=argparse.ArgumentParser("train new model or pretrained model.")
    parser.add_argument('-weightfile', type=str, default=None)
    parser.add_argument('-modelversion', type=int, default=None)
    args=parser.parse_args()
    return args

def webcam():
    cap=cv2.VideoCapture(0)
    window_width=1600
    window_height=900
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)
    assert cap.isOpened(), 'Cannot capture source'

    while  cap.isOpened():
      ret, frame=cap.read()
      if ret:
        image=preprocess(frame)
        result=real_model(image)
        label=postprocess(result)
        #print("label:{}, type:{}".format(label, type(label)))
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
        cv2.imshow('color', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

if __name__ == '__main__':
  args=get_args()
  if args.modelversion is not None:
    webcam()



