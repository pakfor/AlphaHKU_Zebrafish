# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:06:49 2024

@author: ALPHAHKU_HTC1
"""

import albumentations as A
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import cv2

def parse_text_file_to_bboxes_list(path):
    with open(path, "r") as f:
        bboxes = []
        for line in f:
            data = line.replace("\n", "")
            bbox = data.split(" ")[1:]
            label = int(data.split(" ")[0])
            bbox = [float(i) for i in bbox] + [label]
            bboxes.append(bbox)
    return bboxes

def write_bboxes_list_to_text_file(path, bboxes_list):
    string_to_write = ""
    for bbox in bboxes_list:
        bbox_copy = list(bbox)
        bbox_label = bbox_copy.pop()
        bbox_str = [str(round(i, 6)) for i in bbox_copy]
        bbox_str = " ".join(bbox_str)
        bbox_str = str(int(bbox_label)) + " " + bbox_str + "\n"
        string_to_write += bbox_str
    string_to_write = string_to_write[0:-1]
    with open(path, "w+") as f:
        f.write(string_to_write)

def yolo_format_to_xyxy(image, width, height, bbox):
    x_center_norm = bbox[0]
    x_center_abs = int(x_center_norm * width)
    y_center_norm = bbox[1]
    y_center_abs = int(y_center_norm * height)
    width_norm = bbox[2]
    width_abs = int(width_norm * width)
    height_norm = bbox[3]
    height_abs = int(height_norm * height)

    x1 = int(x_center_abs - width_abs / 2)
    if x1 < 0:
        x1 = 0
    x2 = int(x_center_abs + width_abs / 2)
    if x2 > width:
        x2 = width
    y1 = int(y_center_abs - height_abs / 2)
    if y1 < 0:
        y1 = 0
    y2 = int(y_center_abs + height_abs / 2)
    if y2 > height:
        y2 = height

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    return image

VIDEO_NAME = "20240830_005_cropped"
IMAGE_INPUT_PATH = f"G:/Zebrafish Blood Analysis/YOLOV8/Raw Data/Frames/{VIDEO_NAME}/All Images"
LABEL_INPUT_PATH = f"G:/Zebrafish Blood Analysis/YOLOV8/Raw Data/Frames/{VIDEO_NAME}/Label_YOLO"
IMAGE_OUTPUT_PATH = f"G:/Zebrafish Blood Analysis/YOLOV8/Raw Data/Frames/{VIDEO_NAME}_data_augmentation/Images"
LABEL_OUTPUT_PATH = f"G:/Zebrafish Blood Analysis/YOLOV8/Raw Data/Frames/{VIDEO_NAME}_data_augmentation/Labels"
# For verification
LABELED_IMAGE_OUTPUT_PATH = f"G:/Zebrafish Blood Analysis/YOLOV8/Raw Data/Frames/{VIDEO_NAME}_data_augmentation/Labeled_Images"

label_list = os.listdir(LABEL_INPUT_PATH)
label_list.remove("classes.txt")
label_list = [i.replace(".txt", "") for i in label_list]

transformation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=(-90, 90), p=1.0)
    ], bbox_params=A.BboxParams(format='yolo'))
transformation_repetition = 10

for orig_sample in tqdm(label_list):
    orig_img = plt.imread(f"{IMAGE_INPUT_PATH}/{orig_sample}.jpg")
    orig_label_path = f"{LABEL_INPUT_PATH}/{orig_sample}.txt"
    orig_bboxes = parse_text_file_to_bboxes_list(orig_label_path)
    for rep in range(1, transformation_repetition + 1):
        transformed_pairs = transformation(image=orig_img, bboxes=orig_bboxes)
        transformed_img = transformed_pairs['image']
        transformed_bboxes = transformed_pairs['bboxes']

        transformed_img_save_path = f"{IMAGE_OUTPUT_PATH}/{VIDEO_NAME}_{orig_sample}-{rep}.jpg"
        transformed_label_save_path = f"{LABEL_OUTPUT_PATH}/{VIDEO_NAME}_{orig_sample}-{rep}.txt"

        plt.imsave(transformed_img_save_path, transformed_img)
        write_bboxes_list_to_text_file(transformed_label_save_path, transformed_bboxes)

        # For verification
        img_to_plot_save_path = f"{LABELED_IMAGE_OUTPUT_PATH}/{VIDEO_NAME}_{orig_sample}-{rep}.jpg"
        image_to_plot = transformed_img.copy()
        bboxes_to_plot = transformed_bboxes.copy()
        for j in range(0, len(bboxes_to_plot)):
            image_to_plot = yolo_format_to_xyxy(image_to_plot, 320, 320, bboxes_to_plot[j][0:-1])
        plt.imsave(img_to_plot_save_path, image_to_plot)
