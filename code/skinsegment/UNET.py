#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import sys
import time
import random
import warnings

import cv2 as cv
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



model = load_model('lydia.h5', custom_objects={'mean_iou': mean_iou})

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    preds = model.predict(frame, verbose=1)    
    preds_img = (preds > 0.5).astype(np.uint8)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")

#




#
## H) See predicted masks for training samples
#
## In[ ]:
#
#
#ix = random.randint(0, len(preds_train_t))
#imshow(X_train[ix])
#plt.show()
#imshow(np.squeeze(Y_train[ix]))
#plt.show()
#imshow(np.squeeze(preds_train_t[ix]))
#plt.show()
#
#
## I) See predicted masks for validation data
#
## In[ ]:
#
#
#ix = random.randint(0, len(preds_val_t))
#imshow(X_train[int(X_train.shape[0]*0.7):][ix])
#plt.show()
#imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.7):][ix]))
#plt.show()
#imshow(np.squeeze(preds_val_t[ix]))
#plt.show()
#
#
## J) Load model, testing data and check against trained network (if masks exist)
#
## In[ ]:
#
#
#model = load_model('your_model_name.h5', custom_objects={'mean_iou': mean_iou})
#ABD_PATH = '/home/lalzogbi/Documents/Skin_Datasets/allabdomen/val/skin_val2019/'
#MSK_PATH = '/home/lalzogbi/Documents/Skin_Datasets/allabdomen/val/annotations/'
#
#abd_ids = next(os.walk(ABD_PATH))[2]
#msk_ids = next(os.walk(MSK_PATH))[2]
#abd_ids.sort()
#msk_ids.sort()
#
#abd = np.zeros((len(abd_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
#msk = np.zeros((len(msk_ids), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
#
#sys.stdout.flush()
#for n, id_ in tqdm(enumerate(abd_ids), total=len(abd_ids)):
#    path = ABD_PATH + id_
#    img = imread(path)[:,:,:IMG_CHANNELS]
#    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#    abd[n] = img
#    
#for n, id_ in tqdm(enumerate(msk_ids), total=len(msk_ids)):
#    path = MSK_PATH + id_
#    img = imread(path)
#    
#    if img.ndim == 3:
#        img = img[:,:,1]
#        
#    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
#                                      preserve_range=True)
#    if (np.unique(img).size) > 2:
#        img = img > 30           # Important, Needed to make labels 0's and 1's only   
#    else:   
#        img = img > 0
#    img = img.astype(np.uint8)
#    msk[n] = img
#    
## Actual Predictions
#preds_test = model.predict(abd[:int(abd.shape[0])], verbose=1)
#
## Threshold predictions
#preds_test_t = (preds_test > 0.5).astype(np.uint8)
#
## Overall accuracy on abdomen pictures
#answer = acc_comp(msk, preds_test_t);
#
## # Save results in a .npy file
## a = np.reshape(answer[2],(100,1))
## b = np.reshape(answer[3],(100,1))
## c = np.reshape(answer[4],(100,1))
## d = np.reshape(answer[5],(100,1))
## g = np.concatenate([a,b,c,d],axis = 1)
## np.save('your_file_name.npy',g)
#
#
## K) Visualize results
#
## In[ ]:
#
#
#for j in range(len(abd_ids)):
#    print(j)
#    plt.show()
#    imshow(abd[j])
#    plt.show()
#    imshow(np.squeeze(preds_test_t[j]*255))
#    plt.show()
#    imshow(np.squeeze(msk[j]))
#
#
## J') Load model, testing data and check against trained network (if masks do NOT exist)
#
## In[5]:
#
#
#model = load_model('lydia.h5', custom_objects={'mean_iou': mean_iou})
#ABD_PATH = '/home/lalzogbi/Documents/Umbilicus_Skin_Detection/code/wounds/'
#
#abd_ids = next(os.walk(ABD_PATH))[2]
#abd_ids.sort()
#
#abd = np.zeros((len(abd_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
#
#sys.stdout.flush()
#for n, id_ in tqdm(enumerate(abd_ids), total=len(abd_ids)):
#    path = ABD_PATH + id_
#    img = imread(path)[:,:,:IMG_CHANNELS]
#    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#    abd[n] = img
#    
## Actual Predictions
#preds_test = model.predict(abd[:int(abd.shape[0])], verbose=1)
#
## Threshold predictions
#preds_test_t = (preds_test > 0.5).astype(np.uint8)
#
#
## K') Visualize results
#
## In[6]:
#
#
#for j in range(len(abd_ids)):
#    print(j)
#    plt.show()
#    imshow(abd[j])
#    plt.show()
#    imshow(np.squeeze(preds_test_t[j]*255))
#    plt.show()
#
#
## In[ ]:




