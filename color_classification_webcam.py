import cv2
from color_histogram_feature_extraction import color_util
import knn_classifier
import os
import os.path


cap = cv2.VideoCapture(0)
window_width=1600
window_height=900
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)
assert cap.isOpened(), 'Cannot capture source'
#ret, frame = cap.read()
prediction = 'n.a.'

# checking whether the training data is ready
PATH = './training.data'

#if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    #print ('training data is ready, classifier is loading...')

print ('training data is being created...')
open('training.data', 'w')
color_u=color_util()
color_u.training()
print ('training data is ready, classifier is loading...')

while True:
    # Capture frame-by-frame
    ret, f = cap.read()
    cv2.putText(f,'Prediction: ' + prediction,(15, 45),cv2.FONT_HERSHEY_PLAIN, 4, [225,255,255], 4)

    # Display the resulting frame
    cv2.imshow('color', f)

    color_u.color_histogram_of_test_image()

    prediction = knn_classifier.main('training.data')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()		
