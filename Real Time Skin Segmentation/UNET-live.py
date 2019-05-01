#!/usr/bin/env python
# coding: utf-8

# In[9]:


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



model = load_model('../models/UNET.h5', custom_objects={'mean_iou': mean_iou})
# model = load_model('./model-withouvdm-abd-3Mar2019.h5', custom_objects={'mean_iou': mean_iou})


# ## Single Image
# cv2.namedWindow("preview")
# cv2.namedWindow("normal")

# frame= imread('./wound3.jpg')
# # frame= cv2.imread('./128.jpg')
# cv2.imshow("normal", frame)

# img = resize(frame, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True)
# newframe= np.expand_dims(img, axis=0)
# # preds = model.predict((newframe), verbose=0)    
# preds = model.predict(newframe[:int(newframe.shape[0])], verbose=0)    
# # preds_img = preds.astype(np.uint8)
# preds_img = (preds > 0.5).astype(np.uint8)
# cv2.imshow("preview", np.squeeze(preds_img*255, axis=0))
# flag = True
# while (flag):
# 	key = cv2.waitKey(2)
# 	if key == 27: # exit on ESC
# 		flag = False
# cv2.destroyWindow("preview")
# cv2.destroyWindow("normal")


cv2.namedWindow("preview")
cv2.namedWindow("normal")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    ima1 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img = resize(ima1, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True)
    newframe= np.expand_dims(img, axis=0)
    preds = model.predict(newframe, verbose=0)    
    # preds_img = preds.astype(np.uint8)
    preds_img = (preds > 0.5).astype(np.uint8)
    # np.squeeze(preds_img*255, axis=0)
    # print(preds_img.shape)
    cv2.imshow("preview", np.squeeze(preds_img*255, axis=0))
    cv2.imshow("normal",frame)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
cv2.destroyWindow("normal")
