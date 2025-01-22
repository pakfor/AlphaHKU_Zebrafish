import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Aggregate dataset
data_dir = "G:/Zebrafish Blood Analysis/YOLOV8/20241107_Analysis/ZBF_4_Cropped_Cells"
list_of_data = os.listdir(data_dir)
size_of_img = 128
x_train = np.zeros((len(list_of_data), size_of_img, size_of_img, 1), dtype="float32")

for i, data_name in enumerate(list_of_data):
    #img_temp = plt.imread(f"{data_dir}/{data_name}")[:,:,0:-1]
    img_temp = cv2.imread(f"{data_dir}/{data_name}", 0)
    img_temp = np.reshape(img_temp, (size_of_img, size_of_img, 1))
    img_temp = img_temp / 255.0
    x_train[i, :, :, :] = img_temp

x_train = np.reshape(x_train, (len(list_of_data), size_of_img ** 2))

pca = PCA(n_components=2)
pca_components = pca.fit_transform(x_train)

plt.scatter(list(pca_components[:,0]), list(pca_components[:,1]))



