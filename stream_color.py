import cv2
from load_color import ColorModel
import cv2
import numpy as np
import os
import tensorflow as tf

#optimize GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
      tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
  except RuntimeError as e:
      print(e)

model=ColorModel("model1.json", "color_model2.h5")

#optimize opencv2
cv2.useOptimized()
font=cv2.FONT_HERSHEY_SIMPLEX

#video feed
video=cv2.VideoCapture("rtsp://192.168.0.114/live/ch00_0")
while video.isOpened():
    _, frame=video.read()
    color=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi=cv2.resize(color, (80, 80))
    rgb_tensor=tf.convert_to_tensor(roi/255.0, dtype=tf.float32)
    rgb_tensor=tf.expand_dims(rgb_tensor, 0)
    pred=model.predict_color(rgb_tensor)
    #print(rgb_tensor.numpy().shape)
    cv2.putText(frame, pred, (50, 50), font, 1, (255, 255, 0), 2)
    cv2.imshow('color', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()



