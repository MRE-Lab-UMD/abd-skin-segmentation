# Importing bunch of libraries
import os
import random
import time

import cv2
import numpy as np
import torch
from skimage.io import imread, imshow
from skimage.transform import resize

from model import UNet

# Specify Image Dimensions
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

seed = 42
random.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

# Load the trained model, you can use yours or the model we provide in our code
# Make sure to set up the path correctly
PATH = './Models/final_unet_pytorch.pth'
model = UNet(input_channels=3)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

# Open window to visualize the segmentation
cv2.namedWindow("preview")
cv2.namedWindow("normal")
vc = cv2.VideoCapture(0)

# Try to get first frame
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

start_time = time.time()
frame_count = 1
# Loop until user presses ESC key
while rval:
    rval, frame = vc.read()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_img = np.asarray(
        resize(rgb_image, (IMG_HEIGHT, IMG_WIDTH), mode='constant'))
    channel_first_img = np.transpose(resized_img, (2, 0, 1))
    img_added_axis = np.expand_dims(channel_first_img, axis=0)
    input_tensor = torch.from_numpy(img_added_axis).float()
    input_tensor.to(device=device)
    preds = model(input_tensor)
    prediction = preds[0].cpu().detach().numpy()
    prediction = np.transpose(prediction, (1, 2, 0))
    # To see a binary mask, uncomment the below line.
    # prediction = np.uint8((prediction > 0.7) * 255)
    cv2.imshow("preview", prediction)
    cv2.imshow("normal", frame)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    if frame_count % 30 == 0:
        print("Frame Per second: {} fps.".format(
            (time.time() - start_time) / frame_count))
    frame_count = frame_count + 1

# Close window
cv2.destroyWindow("preview")
cv2.destroyWindow("normal")
