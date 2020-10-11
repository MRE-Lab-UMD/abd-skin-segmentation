# Importing bunch of libraries
import os
import sys
import time
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from skimage.io import imread, imshow
from skimage.transform import resize

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import backend as K
import cv2

# Specify Image Dimensions
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed

# Define mean IOU metric, if needed
def mean_iou(y_true, y_pred):
   prec = []
   for t in np.arange(0.5, 1.0, 0.05):
       y_pred_ = tf.to_int32(y_pred > t)
       score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
       K.get_session().run(tf.local_variables_initializer())
       with tf.control_dependencies([up_opt]):
           score = tf.identity(score)
       prec.append(score)
   return K.mean(K.stack(prec), axis=0)

# Load the trained model, you can use yours or the model we provide in our code
# Make sure to set up the path correctly
model = load_model('../Models/UNET.h5', custom_objects={'mean_iou': mean_iou})

# Open window to visualize the segmentation
cv2.namedWindow("preview")
cv2.namedWindow("normal")
vc = cv2.VideoCapture(0)

# Try to get first frame
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

# Loop until user presses ESC key
while rval:
    rval, frame = vc.read()
    ima1 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img = resize(ima1, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True)
    newframe= np.expand_dims(img, axis=0)
    preds = model.predict(newframe, verbose=0)
    preds_img = (preds > 0.5).astype(np.uint8)
    cv2.imshow("preview", np.squeeze(preds_img*255, axis=0))
    cv2.imshow("normal",frame)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

# Close window
cv2.destroyWindow("preview")
cv2.destroyWindow("normal")
