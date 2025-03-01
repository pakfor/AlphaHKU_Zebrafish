import os
import cv2
import random
import shutil
import argparse
import numpy as np
import skimage
import matplotlib.pyplot as plt
from ultralytics import YOLO

random.seed(1)

def create_directories_recursive(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

def prepare_fully_labeled_data(all_labeled_img_dir, all_labels_dir, train_dir, test_dir, train_test_split, preprocess_method, hist_ref_path):
    train_img_dir = f"{train_dir}/images"
    train_lb_dir = f"{train_dir}/labels"
    test_img_dir = f"{test_dir}/images"
    test_lb_dir = f"{test_dir}/labels"
    create_directories_recursive([train_img_dir, train_lb_dir, test_img_dir, test_lb_dir])

    labels_list = os.listdir(all_labels_dir)
    labeled_data_size = len(labels_list)
    train_size = int(train_test_split * labeled_data_size)
    random.shuffle(labels_list)

    if hist_ref_path is not None:
        ref_img = cv2.imread(hist_ref_path, cv2.IMREAD_GRAYSCALE)
        ref_img = np.expand_dims(ref_img, axis=-1)

    for i in range(0, labeled_data_size):
        datum_name = labels_list[i].replace(".txt", "")
        if preprocess_method == "hist_match":
            img_tmp = cv2.imread(f"{all_labeled_img_dir}/{datum_name}.png", cv2.IMREAD_GRAYSCALE)
            img_tmp = np.expand_dims(img_tmp, axis=-1)
            img_tmp = skimage.exposure.match_histograms(img_tmp, ref_img, channel_axis=-1)
            img_tmp = np.squeeze(img_tmp, axis=-1)
            img_tmp = (img_tmp - np.min(img_tmp)) / (np.max(img_tmp) - np.min(img_tmp))

            if i < train_size:
                plt.imsave(f"{train_img_dir}/{datum_name}.png", img_tmp, cmap='gray')
                shutil.copy(f"{all_labels_dir}/{datum_name}.txt",
                            f"{train_lb_dir}/{datum_name}.txt")
            else:
                plt.imsave(f"{test_img_dir}/{datum_name}.png", img_tmp, cmap='gray')
                shutil.copy(f"{all_labels_dir}/{datum_name}.txt",
                            f"{test_lb_dir}/{datum_name}.txt")

        else:
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

def prepare_pseudo_labeled_data(prev_rd_model_dir, all_img_dir, prev_data_dir, curr_data_dir, desired_conf_level, num_pseudo_labeled_data):
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

def main(labeled_img_path, labels_path, num_cell_type, all_img_path, master_path, train_round, train_test_split, conf, extra_sample_per_rd, epoch, batch_size, image_dim, preprocess_method, hist_ref_path):
    for i in range(0, train_round):
        print(f"Round {i} begins")

        # Copy data for training
        if i == 0:
            prepare_fully_labeled_data(labeled_img_path,
                                       labels_path,
                                       f"{master_path}/Round 0/Data/Train",
                                       f"{master_path}/Round 0/Data/Test",
                                       train_test_split,
                                       preprocess_method,
                                       hist_ref_path)
        else:
            prepare_pseudo_labeled_data(f"{master_path}/Round {i - 1}/runs/detect/train/weights/best.pt",
                                        all_img_path,
                                        f"{master_path}/Round {i - 1}/Data",
                                        f"{master_path}/Round {i}/Data",
                                        conf,
                                        100)

        yaml_save_dir = f"{master_path}/Round {i}/config.yaml"
        with open(yaml_save_dir, "w+") as f:
            cell_type_str = "".join([f"  {i}: Cell_{i}\n" for i in range(0, num_cell_type)])
            # f.write(f"train: {master_path}/Round {i}/Data/Train\nval: {master_path}/Round {i}/Data/Test\n\n# Classes\nnames:\n  0: Cell_0\n  1: Cell_1")
            f.write(f"train: {master_path}/Round {i}/Data/Train\nval: {master_path}/Round {i}/Data/Test\n\n# Classes\nnames:\n" + cell_type_str)
        os.chdir(f"{master_path}/Round {i}")
        model = YOLO("G:/Zebrafish Blood Analysis/YOLOV8/Codes/ultralytics/yolov8n.pt")
        results = model.train(data="config.yaml", epochs=epoch, batch=batch_size, imgsz=image_dim, device=0, workers=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_img_path", help="Full path where all the labeled images are stored", type=str)
    parser.add_argument("--labels_path", help="Full path where all labels are stored", type=str)
    parser.add_argument("--num_cell_type", help="Number of cell types", type=int)
    parser.add_argument("--all_img_path", help="Full path where all images are stored (for unsupervised learning)", type=str)
    parser.add_argument("--master_path", help="Full path where the training will be conducted at (results will be saved at)", type=str)
    parser.add_argument("--train_round", help="Number of training rounds (1 for supervised; > 1 for semi-supervised", type=int)
    parser.add_argument("--train_test_split", help="Train-test split for the initial round (fully supervised)", type=float)
    parser.add_argument("--conf", help="Minimum confidence of detection to be considered as true detection (for unsupervised learning)", type=float)
    parser.add_argument("--extra_sample_per_rd", help="Extra samples added to the training set per round", type=int)
    parser.add_argument("--epoch", help="Epoch per round", type=int)
    parser.add_argument("--batch_size", help="Batch size", type=int)
    parser.add_argument("--image_dim", help="Image dimension", type=int)
    parser.add_argument("--preprocess", help="Pre-processing technique", type=str)
    parser.add_argument("--hist_ref", help="Histogram reference image (Only applicable for using preprocessing method hist_match", type=str)
    args = parser.parse_args()

    labeled_img_path = args.labeled_img_path  # "G:/Zebrafish Blood Analysis/YOLOV8/20250206 - Labeled Training/YOLO Data/images"
    labels_path = args.labels_path  # "G:/Zebrafish Blood Analysis/YOLOV8/20250206 - Labeled Training/YOLO Data/labels"
    num_cell_type = args.num_cell_type  # 1
    all_img_path = args.all_img_path  # "G:/Zebrafish Blood Analysis/YOLOV8/20250206 - Labeled Training/YOLO Data/images"
    master_path = args.master_path  # "G:/Zebrafish Blood Analysis/YOLOV8/20250206 - Labeled Training/YOLO Train"
    train_round = args.train_round  # 1
    train_test_split = args.train_test_split  #  0.7
    conf = args.conf  # 0.5
    extra_sample_per_rd = args.extra_sample_per_rd  # 400
    epoch = args.epoch  # 500
    batch_size = args.batch_size  # 16
    image_dim = args.image_dim  # 500
    preprocess_method = args.preprocess  # hist_eq/hist_match
    hist_ref_path = args.hist_ref  # G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation/Data/image/zbf_4_img_0990_png_Box_04.png

    print("Labeled images path", labeled_img_path)
    print("Labels path", labels_path)
    print("Number of cell types", num_cell_type)
    print("All images path", all_img_path)
    print("Training folder", master_path)
    print("Round of training", train_round)
    print("Train-test split", train_test_split)
    print("Confidence level for true detection", conf)
    print("Extra samples per round", extra_sample_per_rd)
    print("Epoch per round", epoch)
    print("Batch size", batch_size)
    print("Image dimension", image_dim)
    print("Preprocessing", preprocess_method)
    print("Histogram reference image", hist_ref_path)

    os.makedirs(master_path, exist_ok=True)

    if preprocess_method is None:
        preprocess_method = " "

    main(labeled_img_path, labels_path, num_cell_type, all_img_path, master_path, train_round, train_test_split, conf, extra_sample_per_rd, epoch, batch_size, image_dim, preprocess_method, hist_ref_path)