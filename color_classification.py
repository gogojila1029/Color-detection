import numpy as np
import pandas as pd
from tensorflow import keras
import struct
import cv2
import os
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
#from keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, ZeroPadding2D, Lambda, Flatten, Dropout, Activation, AveragePooling2D, Input, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from tensorflow.keras.layers import add, Concatenate
from tensorflow.keras.models import Model

#optimise GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
      tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
  except RuntimeError as e:
      print(e)

#image preprocessing
np.random.seed(5)
image_size=(100,100)
num_classes=6

#class list:
class_list=np.array(["black", "blue", "brown", "green", "red", "yellow"])
#image augmentation:
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    brightness_range=(0.1, -0.1),
    rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.2, # Randomly zoom image 
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True  # randomly flip images
  )

y_label=[]
test_it= datagen.flow_from_directory("color_dataset/test/", class_mode="categorical", batch_size=32, target_size=image_size)
train_it=datagen.flow_from_directory("color_dataset/train/", class_mode="categorical", batch_size=32, target_size=image_size)
valid_it=datagen.flow_from_directory("color_dataset/valid/", class_mode="categorical", batch_size=32, target_size=image_size)

class_indices=train_it.class_indices
for i in list(class_indices.keys()):
    y_label.append(class_indices.get(i))

print(y_label)
#color classification model:

def beer_net(num_classes):
    # placeholder for input image
    input_image = Input(shape=(100, 100, 3))
    # ============================================= TOP BRANCH ===================================================
    # first top convolution layer
    top_conv1 = Conv2D(filters=48,kernel_size=(11,11),strides=(4,4),
                            input_shape=(100,100,3),activation='relu')(input_image)
    top_conv1 = BatchNormalization()(top_conv1)
    top_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_conv1)

    # second top convolution layer
    # split feature map by half
    top_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(top_conv1)
    top_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(top_conv1)

    top_top_conv2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(top_top_conv2)
    top_top_conv2 = BatchNormalization()(top_top_conv2)
    top_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv2)

    top_bot_conv2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(top_bot_conv2)
    top_bot_conv2 = BatchNormalization()(top_bot_conv2)
    top_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv2)

    # third top convolution layer
    # concat 2 feature map
    top_conv3 = Concatenate()([top_top_conv2,top_bot_conv2])
    top_conv3 = Conv2D(filters=192,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(top_conv3)

    # fourth top convolution layer
    # split feature map by half
    top_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(top_conv3)
    top_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(top_conv3)

    top_top_conv4 = Conv2D(filters=96,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(top_top_conv4)
    top_bot_conv4 = Conv2D(filters=96,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(top_bot_conv4)

    # fifth top convolution layer
    top_top_conv5 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(top_top_conv4)
    top_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_top_conv5) 

    top_bot_conv5 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(top_bot_conv4)
    top_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(top_bot_conv5)

    # ============================================= TOP BOTTOM ===================================================
    # first bottom convolution layer
    bottom_conv1 = Conv2D(filters=48,kernel_size=(11,11),strides=(4,4),
                            input_shape=(100,100,3),activation="relu")(input_image)
    bottom_conv1 = BatchNormalization()(bottom_conv1)
    bottom_conv1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_conv1)

    # second bottom convolution layer
    # split feature map by half
    bottom_top_conv2 = Lambda(lambda x : x[:,:,:,:24])(bottom_conv1)
    bottom_bot_conv2 = Lambda(lambda x : x[:,:,:,24:])(bottom_conv1)

    bottom_top_conv2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(bottom_top_conv2)
    bottom_top_conv2 = BatchNormalization()(bottom_top_conv2)
    bottom_top_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv2)

    bottom_bot_conv2 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(bottom_bot_conv2)
    bottom_bot_conv2 = BatchNormalization()(bottom_bot_conv2)
    bottom_bot_conv2 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv2)

    # third bottom convolution layer
    # concat 2 feature map
    bottom_conv3 = Concatenate()([bottom_top_conv2,bottom_bot_conv2])
    bottom_conv3 = Conv2D(filters=192,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(bottom_conv3)

    # fourth bottom convolution layer
    # split feature map by half
    bottom_top_conv4 = Lambda(lambda x : x[:,:,:,:96])(bottom_conv3)
    bottom_bot_conv4 = Lambda(lambda x : x[:,:,:,96:])(bottom_conv3)

    bottom_top_conv4 = Conv2D(filters=96,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(bottom_top_conv4)
    bottom_bot_conv4 = Conv2D(filters=96,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(bottom_bot_conv4)

    # fifth bottom convolution layer
    bottom_top_conv5 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(bottom_top_conv4)
    bottom_top_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_top_conv5) 

    bottom_bot_conv5 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation="relu",padding='same')(bottom_bot_conv4)
    bottom_bot_conv5 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(bottom_bot_conv5)

    # ======================================== CONCATENATE TOP AND BOTTOM BRANCH =================================
    conv_output = Concatenate()([top_top_conv5,top_bot_conv5,bottom_top_conv5,bottom_bot_conv5])

    # Flatten
    flatten = Flatten()(conv_output)

    # Fully-connected layer
    FC_1 = Dense(units=1619, activation="tanh")(flatten)
    FC_1 = Dropout(0.5)(FC_1)
    FC_2 = Dense(units=743, activation="selu")(FC_1)
    FC_2 = Dropout(0.5)(FC_2)
    output = Dense(units=num_classes, activation='softmax')(FC_2)
    
    model = Model(inputs=input_image,outputs=output)
    #model.summary()
    #sgd = tf.keras.optimizers.SGD(learning_rate=0.004057)
    # sgd = SGD(lr=0.01, momentum=0.9, decay=0.0005, nesterov=True)
    sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    #return model
    

    #model training
    epochs=500
    batch_size=64

    filepath = 'color_model3.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history=model.fit_generator(train_it, steps_per_epoch=14, validation_data=valid_it, callbacks=callbacks_list, verbose=1, epochs=epochs)
    model_json = model.to_json()
    with open("model2.json", "w") as json_file:
      json_file.write(model_json)
    
    #return prediction
    #print(prediction.argmax(axis=-1))
    #evaluation:
    print("Evaluate")
    #scores= model.evaluate_generator(test_it)
    scores=model.evaluate(test_it)
    print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
    print(model.metrics_names)

    print("Lenght of the model layers:%d" %len(model.layers))

def preprocess(img):
    image=cv2.resize(img, image_size, interpolation = cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = image.astype("float") / 255.0
    #image = np.expand_dims(image, axis=0)
    rgb_tensor=tf.convert_to_tensor(image, dtype=tf.float32)
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

    while cap.isOpened():
      ret, frame=cap.read()
      if ret:
        image=preprocess(frame)
        result=beer_net(num_classes, image)
        label=postprocess(result)
        #print("label:{}, type:{}".format(label, type(label)))
        cv2.putText(frame, str(label), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
        cv2.imshow('color', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
          
def link(frame):
    image=preprocess(frame)
    result=beer_net(num_classes, image)
    label=postprocess(result)
    return label

if __name__ == '__main__':
  args=get_args()
  if args.modelversion is not None:
    beer_net(num_classes)
  elif args.modelversion is None:
    train_again(args.weightfile)











