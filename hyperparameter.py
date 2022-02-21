import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.layers import Input
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, ZeroPadding2D, Lambda, Flatten, Dropout, Activation, AveragePooling2D, Input, Reshape
from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from keras.layers.merge import add, Concatenate
from keras.models import Model
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

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
image_size=(80,80)
num_classes=9
image_shape=(80, 80, 3)

#class list:
class_list=np.array(["black", "blue", "brown", "green", "pink", "red", "silver", "white", "yellow"])
#image augmentation:
datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.2, # Randomly zoom image 
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

y_label=[]
test_it= datagen.flow_from_directory("color_dataset/test/", class_mode="categorical", batch_size=128, target_size=image_size)
train_it=datagen.flow_from_directory("color_dataset/train/", class_mode="categorical", batch_size=128, target_size=image_size)
valid_it=datagen.flow_from_directory("color_dataset/valid/", class_mode="categorical", batch_size=128, target_size=image_size)

print(train_it.labels)

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape=input_shape
        self.num_classes=num_classes

    def build(self, hp):
        # placeholder for input image
        input_image = Input(shape=self.input_shape)
        # ============================================= TOP BRANCH ===================================================
        # first top convolution layer
        top_conv1 = Conv2D(filters=48,kernel_size=(11,11),strides=(4,4),
                                input_shape=(80,80,3),activation=hp.Choice(
                                    'conv_activation1',
                                    values=['relu', 'elu', 'selu', 'swish', 'tanh'],
                                    default="relu"
                                ))(input_image)
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
                                input_shape=(80,80,3),activation="relu")(input_image)
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
        FC_1 = Dense(units=hp.Int('units1', min_value=32, max_value=2048), activation=hp.Choice('dense_activation', values=['relu', 'elu', 'selu', 'swish', 'tanh'], default="relu"))(flatten)
        FC_1 = Dropout(0.5)(FC_1)
        FC_2 = Dense(units=hp.Int('units2', min_value=32, max_value=2048), activation=hp.Choice('dense_activation1', values=['relu', 'elu', 'selu', 'swish', 'tanh'], default="relu"))(FC_1)
        FC_2 = Dropout(0.5)(FC_2)
        output = Dense(units=self.num_classes, activation='softmax')(FC_2)

        model = Model(inputs=input_image,outputs=output)
        #sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=keras.optimizers.SGD(
            hp.Float(
                'learning_rate',
                min_value=1e-4,
                max_value=1e-2,
                sampling="LOG",
                default=1e-3
            )
        ), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

hypermodel=CNNHyperModel(input_shape=image_shape, num_classes=num_classes)

tuner=RandomSearch(
    hypermodel,
    objective="val_accuracy",
    seed=1,
    max_trials=30,
    executions_per_trial=2,
    directory="./test_dir"
)

tuner.search_space_summary()
N_EPOCH_SEARCH=100

#Show a summary of the search
tuner.search(train_it, epochs=N_EPOCH_SEARCH, steps_per_epoch=74,validation_data=test_it)
tuner.results_summary()

#Retrieve the best model
best_model=tuner.get_best_models(num_models=1)[0]

#Evaluate the best model.
print("Evaluate")
scores= best_model.evaluate_generator(valid_it)
print("%s: %.2f%%" %(best_model.metrics_names[1], scores[1]*100))
print(best_model.metrics_names)
