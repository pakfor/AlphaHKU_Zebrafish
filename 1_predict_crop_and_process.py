import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from skimage.transform import resize
import argparse


def main(frame_skip, conf_threshold, height_width_ratio_limit, crop_as_square, resize_img, force_resize, resize_to, pad_img, yolo_path, img_in_path, img_out_path):
    frame_skip = frame_skip
    conf_threshold = conf_threshold
    height_width_ratio_limit = height_width_ratio_limit
    crop_as_square = crop_as_square
    resize_img = resize_img
    force_resize = force_resize
    resize_to = resize_to
    pad_img = pad_img

    trained_model_dir = yolo_path # "G:/Zebrafish Blood Analysis/YOLOV8/20250216 - Demo/Model/YOLOV8/20241215_RD0/best.pt"
    image_to_predict_dir = img_in_path # "G:/Zebrafish Blood Analysis/YOLOV8/20250216 - Demo/ZBF 2/Frame"
    output_dir = img_out_path # "G:/Zebrafish Blood Analysis/YOLOV8/20250216 - Demo/ZBF 2/Cell"

    os.makedirs(output_dir, exist_ok=True)

    list_of_img_to_predict = os.listdir(image_to_predict_dir)
    list_of_img_to_predict = [list_of_img_to_predict[i] for i in range(0, len(list_of_img_to_predict), frame_skip)]
    
    model = YOLO(trained_model_dir)
    
    for image_name in tqdm(list_of_img_to_predict):
        # Load image
        img_to_pred = cv2.imread(f"{image_to_predict_dir}/{image_name}")
    
        # Basic information - Dimension
        y_length = img_to_pred.shape[0]
        x_length = img_to_pred.shape[1]
    
        # Prediction
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
    
                if crop_as_square:
                    # Crop as square with bigger bounding box
                    long_edge = max(x_diff, y_diff)
                    # Make y edge longer
                    if y_diff < long_edge:
                        add_to_each_side = int((long_edge - y_diff + 1) / 2)
                        distance_to_upper_bound = y_length - (y2 + add_to_each_side)
                        distance_to_lower_bound = y1 - add_to_each_side
                        if distance_to_upper_bound >= 0 and distance_to_lower_bound >= 0:
                            y1 = y1 - add_to_each_side
                            y2 = y2 + add_to_each_side
                        elif distance_to_upper_bound >= 0 and distance_to_lower_bound < 0:
                            if abs(distance_to_upper_bound) >= abs(distance_to_lower_bound):
                                y1 = 0
                                y2 = y2 + add_to_each_side + abs(distance_to_lower_bound)
                            else:
                                print("Bounding box is too big, skipping...")
                                continue
                        elif distance_to_upper_bound < 0 and distance_to_lower_bound >= 0:
                            if abs(distance_to_lower_bound) >= abs(distance_to_upper_bound):
                                y1 = y1 - add_to_each_side - abs(distance_to_upper_bound)
                                y2 = y_length
                            else:
                                print("Bounding box is too big, skipping...")
                                continue
                        else:
                            print("Bounding box is too big, skipping...")
                            continue
                    # Make x edge longer
                    if x_diff < long_edge:
                        add_to_each_side = int((long_edge - x_diff + 1) / 2)
                        distance_to_upper_bound = x_length - (x2 + add_to_each_side)
                        distance_to_lower_bound = x1 - add_to_each_side
                        if distance_to_upper_bound >= 0 and distance_to_lower_bound >= 0:
                            x1 = x1 - add_to_each_side
                            x2 = x2 + add_to_each_side
                        elif distance_to_upper_bound >= 0 and distance_to_lower_bound < 0:
                            if abs(distance_to_upper_bound) >= abs(distance_to_lower_bound):
                                x1 = 0
                                x2 = x2 + add_to_each_side + abs(distance_to_lower_bound)
                            else:
                                print("Bounding box is too big, skipping...")
                                continue
                        elif distance_to_upper_bound < 0 and distance_to_lower_bound >= 0:
                            if abs(distance_to_lower_bound) >= abs(distance_to_upper_bound):
                                x1 = x1 - add_to_each_side - abs(distance_to_upper_bound)
                                x2 = x_length
                            else:
                                print("Bounding box is too big, skipping...")
                                continue
                        else:
                            print("Bounding box is too big, skipping...")
                            continue
    
                x_diff = abs(x1 - x2)
                y_diff = abs(y1 - y2)
    
                ratio = np.min([x_diff, y_diff]) / np.max([x_diff, y_diff])
                if ratio >= height_width_ratio_limit:
                    cropped_img = img_to_pred[y1:y2, x1:x2]
                    orig_shape_0 = cropped_img.shape[0]
    
                    # Resize
                    if resize_img:
                        if cropped_img.shape[0] > cropped_img.shape[1]:
                            new_size = (resize_to, int(cropped_img.shape[1] * (resize_to / cropped_img.shape[0])), cropped_img.shape[2])
                            cropped_img = resize(cropped_img, new_size)
                        else:
                            new_size = (int(cropped_img.shape[0] * (resize_to / cropped_img.shape[1])), resize_to, cropped_img.shape[2])
                        cropped_img = resize(cropped_img, new_size)
                        # For calculating exact area in cell feature extraction
                        # resize_ratio = cropped_img.shape[0] / resize_to
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
                        # half_diff_2 = diff - half_diff_1
                        img_unfilled = np.zeros((long_edge, long_edge, cropped_img.shape[2]), dtype="float32")
                        # Pad horizontally
                        if cropped_img.shape[0] == long_edge:
                            img_unfilled[:,0+half_diff_1:0+half_diff_1+cropped_img.shape[1],:] = cropped_img
                        else:
                            img_unfilled[0+half_diff_1:0+half_diff_1+cropped_img.shape[0], :, :] = cropped_img
                        cropped_img = img_unfilled
                    else:
                        pass
                    img_save_name = f"{image_name.replace('.png', '')}_Box_{str(i).zfill(2)}_{orig_shape_0}_{resize_to}.png"
                    plt.imsave(f"{output_dir}/{img_save_name}", cropped_img)
            else:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_skip", help="Number of frame skipped between processing", type=int)
    parser.add_argument("--conf", help="Minimum confidence level to be considered as valid detection", type=float)
    parser.add_argument("--crop_square", help="Crop cell as square", action='store_true')
    parser.add_argument("--resize_to", help="Dimension of resizing, only applicable when cropping as square", type=int)
    parser.add_argument("--trained_yolo_path", help="Full path where trained YOLO is stored", type=str)
    parser.add_argument("--image_input_path", help="Full path where frame to be predicted is stored", type=str)
    parser.add_argument("--image_output_path", help="Full path where output (cropped cells) will be stored", type=str)
    args = parser.parse_args()

    print("Frame skip", args.frame_skip)
    print("Confidence level", args.conf)
    print("Crop as square", args.crop_square)
    print("Resize to", args.resize_to)
    print("YOLO model", args.trained_yolo_path)
    print("INPUT", args.image_input_path)
    print("OUTPUT", args.image_output_path)

    frame_skip = args.frame_skip
    conf_threshold = args.conf
    height_width_ratio_limit = 0.0
    crop_as_square = args.crop_square
    resize_img = False
    pad_img = False
    if args.resize_to is None:
        force_resize = False
        resize_to = 128
    else:
        force_resize = True
        resize_to = args.resize_to

    yolo_path = args.trained_yolo_path
    img_in_path = args.image_input_path
    img_out_path = args.image_output_path

    main(frame_skip, conf_threshold, height_width_ratio_limit, crop_as_square, resize_img, force_resize, resize_to, pad_img, yolo_path, img_in_path, img_out_path)