# CC BY-NC-SA
# Copyright 2022. Suhwan Lim. all rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

from PIL import Image
import os
import cv2
import numpy as np
from scipy.stats import itemfreq

class color_util:
    dic={}
    def __init__(self, test_src_image=None, label=None):
        self.test_src_image=test_src_image
        self.label=label

    def color_histogram_of_test_image(self):

        # load the image
        dic=color_util.dic
        image = self.test_src_image
        mask=image.copy()
        mask=255
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        chans = cv2.split(image)
        colors = ('b', 'g', 'r')
        features = []
        feature_data = ''
        counter = 0

        for (chan, color) in zip(chans, colors):
            counter = counter + 1

            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            # find the peak pixel values for R, G, and B
            elem = np.argmax(hist)
            features.append(elem)  
        dic[self.label]=features
        return dic, self.label

    def color_histogram_of_training_image(self, img_name):

        # detect image color by using image file name to label training data
        if 'red' in img_name:
            data_source = 'red'
        elif 'yellow' in img_name:
            data_source = 'yellow'
        elif 'green' in img_name:
            data_source = 'green'
        elif 'orange' in img_name:
            data_source = 'orange'
        elif 'white' in img_name:
            data_source = 'white'
        elif 'black' in img_name:
            data_source = 'black'
        elif 'blue' in img_name:
            data_source = 'blue'
        elif 'pink' in img_name:
            data_source = 'pink'
        elif 'brown' in img_name:
            data_source = "brown"    

        # load the image
        image = cv2.imread(img_name)

        chans = cv2.split(image)
        colors = ('b', 'g', 'r')
        features = []
        feature_data = ''
        counter = 0
        for (chan, color) in zip(chans, colors):
            counter = counter + 1

            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist)

            # find the peak pixel values for R, G, and B
            elem = np.argmax(hist)

            if counter == 1:
                blue = str(elem)
            elif counter == 2:
                green = str(elem)
            elif counter == 3:
                red = str(elem)
                feature_data = red + ',' + green + ',' + blue

        with open('training3.data', 'a') as myfile:
            myfile.write(feature_data + ',' + data_source + '\n')


    def training(self):
        a=color_util()
        # red color training images
        for f in os.listdir('./train/red'):
            a.color_histogram_of_training_image('./train/red/' + f)

        # yellow color training images
        for f in os.listdir('./train/yellow'):
            a.color_histogram_of_training_image('./train/yellow/' + f)

        # green color training images
        for f in os.listdir('./train/green'):
            a.color_histogram_of_training_image('./train/green/' + f)

        # black color training images
        for f in os.listdir('./train/black'):
            a.color_histogram_of_training_image('./train/black/' + f)

        # blue color training images
        for f in os.listdir('./train/blue'):
            a.color_histogram_of_training_image('./train/blue/' + f)	
            	
        for f in os.listdir('./train/brown'):
            a.color_histogram_of_training_image('./train/brown/' + f)

            
if __name__=='__main__':
    col=color_util()
    col.training()