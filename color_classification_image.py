import cv2
from color_histogram_feature_extraction import color_util
import knn_classifier
import os
import os.path
import sys

# read the test image
try:
    source_image = cv2.imread(sys.argv[1])
except:
    source_image = cv2.imread('f7919e023f.jpg')
prediction = 'n.a.'

# checking whether the training data is ready
PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    print ('training data is being created...')
    open('training.data', 'w')
    color_util=color_util(source_image)
    color_util.training()
    print ('training data is ready, classifier is loading...')

# get the prediction
color_util.color_histogram_of_test_image(source_image)
prediction = knn_classifier.main('training.data')
print('Detected color is:', prediction)
cv2.putText(
    source_image,
    'Prediction: ' + prediction,
    (15, 45),
    cv2.FONT_HERSHEY_PLAIN,
    3,
    200,
    )

# Display the resulting frame
cv2.imshow('color classifier', source_image)
cv2.waitKey(0)		
