# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:33:02 2024

@author: ALPHAHKU_HTC1
"""

import os
import cv2
import random
import shutil
from ultralytics import YOLO

random.seed(1)

NUM_ROUND = 1
ALL_LABELED_IMG_DIR = "G:/Zebrafish Blood Analysis/YOLOV8/All Labeled Data/images"
ALL_LABELS_DIR = "G:/Zebrafish Blood Analysis/YOLOV8/All Labeled Data/labels"
ALL_IMG_DIR = "G:/Zebrafish Blood Analysis/YOLOV8/All Labeled Data/images"
SELF_TRAIN_MASTER_DIR = "G:/Zebrafish Blood Analysis/YOLOV8/20241215_Self_Training"
if not os.path.exists(SELF_TRAIN_MASTER_DIR):
    os.mkdir(SELF_TRAIN_MASTER_DIR)

INITIAL_TRAIN_TEST_SPLIT = 0.7
DESIRED_CONF = 0.5
EXTRA_IMAGE_PER_ROUND = 400

def create_directories_recursive(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

def prepare_fully_labeled_data(all_labeled_img_dir, all_labels_dir, train_dir, test_dir, train_test_split):
    train_img_dir = f"{train_dir}/images"
    train_lb_dir = f"{train_dir}/labels"
    test_img_dir = f"{test_dir}/images"
    test_lb_dir = f"{test_dir}/labels"
    create_directories_recursive([train_img_dir, train_lb_dir, test_img_dir, test_lb_dir])

    labels_list = os.listdir(all_labels_dir)
    labeled_data_size = len(labels_list)
    train_size = int(train_test_split * labeled_data_size)
    random.shuffle(labels_list)
    for i in range(0, labeled_data_size):
        datum_name = labels_list[i].replace(".txt", "")
        if i < train_size:
            shutil.copy(f"{all_labeled_img_dir}/{datum_name}.png",
                        f"{train_img_dir}/{datum_name}.png")
            shutil.copy(f"{all_labels_dir}/{datum_name}.txt",
                        f"{train_lb_dir}/{datum_name}.txt")
        else:
            shutil.copy(f"{all_labeled_img_dir}/{datum_name}.png",
                        f"{test_img_dir}/{datum_name}.png")
            shutil.copy(f"{all_labels_dir}/{datum_name}.txt",
                        f"{test_lb_dir}/{datum_name}.txt")

def prepare_pseudo_labeled_data(prev_rd_model_dir, all_img_dir, prev_data_dir, curr_data_dir, desired_conf_level, num_pseudo_labeled_data=None):
    # Copy existing labeled data
    shutil.copytree(prev_data_dir, curr_data_dir)

    # Sample new images for generating pseudo labels
    all_img_list = os.listdir(all_img_dir)
    prev_rd_train_labeled_img_list = os.listdir(f"{prev_data_dir}/Train/images")
    prev_rd_test_labeled_img_list = os.listdir(f"{prev_data_dir}/Test/images")

    chosen_new_img = []
    if num_pseudo_labeled_data is None:
        num_pseudo_labeled_data = len(prev_rd_train_labeled_img_list)
    while len(chosen_new_img) < num_pseudo_labeled_data:
        choose_temp = random.choice(all_img_list)
        if (choose_temp not in prev_rd_train_labeled_img_list) and (choose_temp not in prev_rd_test_labeled_img_list):
            chosen_new_img.append(choose_temp)

    # Make pseudo labels using the model trained in the last round
    model = YOLO(prev_rd_model_dir)
    for img_to_pred_name in chosen_new_img:
        # Load and predict
        img_to_pred = cv2.imread(f"{all_img_dir}/{img_to_pred_name}")
        prediction = model(img_to_pred)
        prediction = prediction[0]
        pred_boxes = prediction.boxes.xywhn.cpu().numpy()
        pred_boxes_conf = list(prediction.boxes.conf.cpu().numpy())

        write_str = ""
        num_boxes = pred_boxes.shape[0]
        for i in range(0, num_boxes):
            if pred_boxes_conf[i] >= desired_conf_level:
                write_str = write_str + "0" + " " + str(round(pred_boxes[i, 0], 6)) + " " + str(round(pred_boxes[i, 1], 6)) + " " + str(round(pred_boxes[i, 2], 6)) + " " + str(round(pred_boxes[i, 3], 6)) + "\n"
            else:
                pass

        # Copy pseudo labeled image
        shutil.copy(f"{all_img_dir}/{img_to_pred_name}", f"{curr_data_dir}/Train/images/{img_to_pred_name}")
        # Copy pseudo labels
        labels_txt_file_dir = f"{curr_data_dir}/Train/labels/{img_to_pred_name.replace('png', 'txt')}"
        with open(labels_txt_file_dir, "w+") as f:
            f.write(write_str)

def main():
    for i in range(0, NUM_ROUND):
        print(f"Round {i} begins")

        # Copy data for training
        if i == 0:
            prepare_fully_labeled_data(ALL_LABELED_IMG_DIR,
                                       ALL_LABELS_DIR,
                                       f"{SELF_TRAIN_MASTER_DIR}/Round 0/Data/Train",
                                       f"{SELF_TRAIN_MASTER_DIR}/Round 0/Data/Test",
                                       INITIAL_TRAIN_TEST_SPLIT)
        else:
            pass
            prepare_pseudo_labeled_data(f"{SELF_TRAIN_MASTER_DIR}/Round {i - 1}/runs/detect/train/weights/best.pt",
                                        ALL_IMG_DIR,
                                        f"{SELF_TRAIN_MASTER_DIR}/Round {i - 1}/Data",
                                        f"{SELF_TRAIN_MASTER_DIR}/Round {i}/Data",
                                        DESIRED_CONF,
                                        100)

        yaml_save_dir = f"{SELF_TRAIN_MASTER_DIR}/Round {i}/config.yaml"
        with open(yaml_save_dir, "w+") as f:
            f.write(f"train: {SELF_TRAIN_MASTER_DIR}/Round {i}/Data/Train\nval: {SELF_TRAIN_MASTER_DIR}/Round {i}/Data/Test\n\n# Classes\nnames:\n  0: Cell")
        os.chdir(f"{SELF_TRAIN_MASTER_DIR}/Round {i}")
        model = YOLO("G:/Zebrafish Blood Analysis/YOLOV8/Codes/ultralytics/yolov8n.pt")
        results = model.train(data="config.yaml", epochs=500, batch=16, imgsz=500, device=0, workers=2)

if __name__ == '__main__':
    main()