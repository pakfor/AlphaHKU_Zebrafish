import os
import cv2
import random
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from skimage.transform import resize

frame_skip = 1
conf_threshold = 0.8
height_width_ratio_limit = 0.8
resize_img = False
force_resize = True
resize_to = 128
pad_img = False

trained_model_dir = "G:/Zebrafish Blood Analysis/YOLOV8/20241215_Self_Training/Round 0/runs/detect/train/weights/best.pt"
image_to_predict_dir = "G:/Zebrafish Blood Analysis/YOLOV8/All Labeled Data/images"
output_dir = "G:/Zebrafish Blood Analysis/YOLOV8/20250109_Analysis/All Images"

list_of_img_to_predict = os.listdir(image_to_predict_dir)
list_of_img_to_predict = [list_of_img_to_predict[i] for i in range(0, len(list_of_img_to_predict), frame_skip)]

model = YOLO(trained_model_dir)

for image_name in tqdm(list_of_img_to_predict):
    img_to_pred = cv2.imread(f"{image_to_predict_dir}/{image_name}")
    prediction = model(img_to_pred)
    predicted_b_boxes = prediction[0].boxes.xyxy.cpu().numpy().astype("int")
    predicted_b_boxes_confidence = list(prediction[0].boxes.conf.cpu().numpy())
    num_b_boxes = predicted_b_boxes.shape[0]
    for i in range(0, num_b_boxes):
        if predicted_b_boxes_confidence[i] >= conf_threshold:
            x1 = predicted_b_boxes[i, 0]
            x2 = predicted_b_boxes[i, 2]
            y1 = predicted_b_boxes[i, 1]
            y2 = predicted_b_boxes[i, 3]
            x_diff = abs(x1 - x2)
            y_diff = abs(y1 - y2)
            ratio = np.min([x_diff, y_diff]) / np.max([x_diff, y_diff])
            if ratio >= height_width_ratio_limit:
                cropped_img = img_to_pred[y1:y2, x1:x2]

                # Resize
                if resize_img:
                    if cropped_img.shape[0] > cropped_img.shape[1]:
                        new_size = (resize_to, int(cropped_img.shape[1] * (resize_to / cropped_img.shape[0])), cropped_img.shape[2])
                        cropped_img = resize(cropped_img, new_size)
                    else:
                        new_size = (int(cropped_img.shape[0] * (resize_to / cropped_img.shape[1])), resize_to, cropped_img.shape[2])
                    cropped_img = resize(cropped_img, new_size)
                else:
                    pass

                if force_resize:
                    new_size = (resize_to, resize_to, cropped_img.shape[2])
                    cropped_img = resize(cropped_img, new_size)
                else:
                    pass

                # Pad
                if pad_img:
                    long_edge = np.max([cropped_img.shape[0], cropped_img.shape[1]])
                    diff = abs(cropped_img.shape[0] - cropped_img.shape[1])
                    half_diff_1 = int(diff / 2)
                    half_diff_2 = diff - half_diff_1
                    img_unfilled = np.zeros((long_edge, long_edge, cropped_img.shape[2]), dtype="float32")
                    # Pad horizontally
                    if cropped_img.shape[0] == long_edge:
                        img_unfilled[:,0+half_diff_1:0+half_diff_1+cropped_img.shape[1],:] = cropped_img
                    else:
                        img_unfilled[0+half_diff_1:0+half_diff_1+cropped_img.shape[0], :, :] = cropped_img
                    cropped_img = img_unfilled
                else:
                    pass
                img_save_name = f"{image_name}_Box_{str(i).zfill(2)}.png"
                plt.imsave(f"{output_dir}/{img_save_name}", cropped_img)
        else:
            pass