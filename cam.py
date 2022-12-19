# CC BY-NC-SA
# Copyright 2022. Suhwan Lim. all rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

from __future__ import division
import torch
import torch.nn as nn
from flask import Flask, render_template, Response
from color_histogram_feature_extraction import color_util
import knn_classifier
import time
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image

import pandas as pd
import random 
import argparse
import pickle as pkl
import tensorflow as tf
import os

app=Flask(__name__) #initialize the Flask app
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
      for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus=tf.config.list_logical_devices("GPU")
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")    
  except RuntimeError as e:
      print(e)

#optimize opencv2
cv2.useOptimized()

def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()#transpose axis of the matrix
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, model):
    count=0
    num_classes=9
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    #print("x_max:{}, y_max:{}, x_min:{}, y_min:{}".format(str(c1[0].item()), str(c1[1].item()), str(c2[0].item()), str(c2[1].item())))
    ref_point=[(c1[0].item(), c1[1].item()), (c2[0].item(), c2[1].item())]
    pred="None"
    crop_img=img[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]

    #cv2.imshow("croped", crop_img)
    #resize the frame of crop_img to input in keras color classfication model
    #crop_img=cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    try:
        crop_img=cv2.resize(crop_img, (1600, 800), interpolation=cv2.INTER_NEAREST)#원래는 (80, 80)
        if label:
            pred=model.main("training3.data", crop_img, label)
           

    except cv2.error:
        pass

    #get the color label of image:
    #color_label=color_classification.link(crop_img)

    cv2.rectangle(img, c1, c2,(253, 4, 3), 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 3 , 3)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, (253,4,3), -1)
    cv2.putText(img, "{}:{}".format(label, pred), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 4, [225,255,0], 2)

    
    #print(crop_img)
    #if label=="toothbrush":
        #print("yay!")
    #color = random.choice(colors)
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 based color detection')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.4)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "192", type = str)
    return parser.parse_args()

def gen_frames():
    #model1=ColorModel("model.json", "color_model1.h5")
    frames=0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img, orig_im, dim = prep_image(frame, inp_dim)

            #im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            if CUDA:
                #im_dim = im_dim.cuda()
                img = img.cuda()
            
            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                #cv2.imshow("frame", orig_im) ret, buffer=cv2.imencode('.jpg', orig_im)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show resul
                #key = cv2.waitKey(1)
                #if key & 0xFF == ord('q'):
                    #break
                #continue
            
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            
#            im_dim = im_dim.repeat(output.size(0), 1)
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]
            
            prediction=knn_classifier
            list(map(lambda x: write(x, orig_im, prediction), output))
            #cv2.imshow("frame", orig_im)
            ret, buffer=cv2.imencode('.jpg', orig_im)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            #key = cv2.waitKey(1)
            #if key & 0xFF == ord('q'):
                #break
            #frames += 1
            #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))  
        else:
            break 
         

if __name__ == '__main__':  
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "custom/yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    torch.set_num_threads(1)
    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
            
    model.eval()
    
    videofile = 'video.avi'
    
    #cap = cv2.VideoCapture("rtsp://192.168.0.114/live/ch00_0")
    cap=cv2.VideoCapture(0)
    window_width=1600
    window_height=900
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)
    print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    assert cap.isOpened(), 'Cannot capture source'

    
    start = time.time()   
    classes = load_classes('data/coco.names')
    #colors = pkl.load(open("pallete", "rb"))
    @app.route('/video_feed')
    def video_feed():
        #Video streaming route. Put this in the src attribute of an img tag
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/')
    def index():
        """Video streaming home page."""
        return render_template('index.html')   
    app.run(debug=True, threaded=True, port=8000)     

    

    
    

