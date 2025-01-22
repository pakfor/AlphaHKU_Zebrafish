#%% Import Library

# Import tensorflow
import tensorflow as tf

# Import
from skimage.io import imread, imsave
import skimage.util
from skimage import color
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

#%%
def preprocess_input(image):
    # Convert the input to tf.float32 type and add batch dimension
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    return image

#%% Loading Data

# File path
model_path = "./Model/1736748801/model_ckpt"
image_path = "G:/Zebrafish Blood Analysis/YOLOV8/20250109_Analysis/All Images"
mask_path = f"{image_path}/prediction_mask_v2"
montage_path = f"{image_path}/prediction_montage_v2"
# Create the directories as mentioned above
if not os.path.exists(mask_path):
    os.mkdir(mask_path)
    if not os.path.exists(montage_path):
        os.mkdir(montage_path)

# Load UNet model
model = tf.saved_model.load(model_path)

file_name = []
for file in os.listdir(image_path):
    if ".png" in file:
        file_name.append(file)

#%% Running model
for i in range(len(file_name)):
    # Load testing image
    # Rick
    #image = imread(f"{image_path}/{file_name[i]}")
    #image = np.double(image[:, :, 0:3])
    #image = image/np.max(image)
    #img_plot = color.rgb2gray(image)

    # PF
    image = cv2.imread(f"{image_path}/{file_name[i]}", cv2.IMREAD_GRAYSCALE)
    image = skimage.exposure.equalize_hist(image)
    img_plot = image.copy()
    image = np.expand_dims(image, axis=-1)
    

    # Process to UNet required shape
    image = preprocess_input(image)
    
    # Model predicting mask
    pred_mask = model(image)
    pred_mask = np.squeeze(pred_mask)
    
    # Saving file
    np.save(f"{mask_path}/{file_name[i][:-4]}", pred_mask)
    imsave(f"{mask_path}/{file_name[i][:-4]}.tif", pred_mask)
    mon = np.zeros([2, 128, 128], dtype = np.float32)
    mon[0, :, :] = img_plot
    mon[1, :, :] = pred_mask
    mon_img = skimage.util.montage(mon)
    imsave(f"{montage_path}/{file_name[i][:-4]}.tif", mon_img)