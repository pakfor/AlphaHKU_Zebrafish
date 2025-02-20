import os
import cv2
import pickle
import sklearn
import skimage
import numpy as np
import scipy
import argparse
import tensorflow as tf
from tqdm import tqdm
from skimage.transform import resize


MASK_PROPERTIES = ['area', 'centroid', 'centroid_norm', 'area_convex', 'eccentricity', 'equivalent_diameter_area', 'extent', 'axis_major_length', 'axis_minor_length', 'orientation', 'perimeter', 'solidity']
CELL_PROPERTIES = ['absorption_density', 'variance', 'skewness', 'kurtosis', 'amplitude_range', 'amplitude_max']


# Mask features extraction
def measure_mask_properties(mask, mask_prop, threshold, pix_ratio=1):
    mask_y_dim, mask_x_dim = mask.shape
    mask[mask >= threshold] = 1.0 ; mask[mask < threshold] = 0.0
    labeled = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(labeled)
    label_num_with_largest_area = np.argmax([i.area for i in regions])
    feature_table = dict()
    for prop in mask_prop:
        if prop == 'centroid':
            cx, cy = regions[label_num_with_largest_area].centroid
            feature_table['centroid_x'] = cx
            feature_table['centroid_y'] = cy
        elif prop == 'centroid_norm':
            cx, cy = regions[label_num_with_largest_area].centroid
            feature_table['centroid_x_norm'] = cx / mask_x_dim
            feature_table['centroid_y_norm'] = cy / mask_y_dim
        elif prop == 'area':
            feature_table[prop] = regions[label_num_with_largest_area][prop] * (pix_ratio ** 2)
        else:
            feature_table[prop] = regions[label_num_with_largest_area][prop]
    return list(feature_table.values())

# Cell features extraction
def measure_cell_properties(image, mask, cell_prop):
    mask_tf = mask == 1
    mask_area = np.sum(mask)
    cell_only = np.ma.masked_array(image, mask=mask_tf)
    averaged_absorbance = (np.sum(np.max(cell_only) - cell_only)) / mask_area
    variance = np.var(cell_only)
    amp_range = np.max(cell_only) - np.min(cell_only)
    amp_max = np.max(cell_only)
    cell_only_flatten = [image.flatten()[i] for i in range(0, len(image.flatten())) if mask_tf.flatten()[i]]
    skewness = scipy.stats.skew(cell_only_flatten)
    kurtosis = scipy.stats.kurtosis(cell_only_flatten)
    # amp_min = np.min(cell_only)
    feature_vec = [averaged_absorbance, variance, skewness, kurtosis, amp_range, amp_max]
    return feature_vec


def convert_bbox_to_square(x1, x2, y1, y2, x_dim, y_dim):
    y_length = y_dim
    x_length = x_dim
    x_diff = abs(x1 - x2)
    y_diff = abs(y1 - y2)
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
        elif distance_to_upper_bound < 0 and distance_to_lower_bound >= 0:
            if abs(distance_to_lower_bound) >= abs(distance_to_upper_bound):
                y1 = y1 - add_to_each_side - abs(distance_to_upper_bound)
                y2 = y_length
            else:
                print("Bounding box is too big, skipping...")
        else:
            print("Bounding box is too big, skipping...")
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
        elif distance_to_upper_bound < 0 and distance_to_lower_bound >= 0:
            if abs(distance_to_lower_bound) >= abs(distance_to_upper_bound):
                x1 = x1 - add_to_each_side - abs(distance_to_upper_bound)
                x2 = x_length
            else:
                print("Bounding box is too big, skipping...")
        else:
            print("Bounding box is too big, skipping...")
    return x1, x2, y1, y2

def segment_image(image, segmentor):
    image = np.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    mask = segmentor.predict(image, verbose=0)
    mask = np.squeeze(mask)
    mask = np.expand_dims(mask, axis=-1)
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    labeled = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(labeled)
    mask_with_largest_area = (labeled == (1 + np.argmax([i.area for i in regions]))).astype("float32")
    mask_with_largest_area = np.squeeze(mask_with_largest_area)
    # masked_img = np.multiply(image, mask_with_largest_area)
    return mask_with_largest_area

def pca_transform(feature_vec, pca_model):
    feature_vec = np.expand_dims(np.array(feature_vec, dtype=np.double), 0)
    pca_components = pca_model.transform(feature_vec)
    pca_components = pca_components.astype("float32")
    return pca_components

def km_transform(pca_components, km_model):
    km_pred = km_model.predict(pca_components)
    return km_pred

def main(labeled_img_dir, labels_dir, segmentor_saved_dir, pca_saved_dir, km_saved_dir, new_labels_save_dir):
    # Directories
    # all_labeled_data_dir = all_labeled_data_dir # "G:/Zebrafish Blood Analysis/YOLOV8/20241016_Self_Training/All Labeled Data"
    all_labeled_images_dir = labeled_img_dir # f"{all_labeled_data_dir}/images"
    all_labeled_labels_dir = labels_dir # f"{all_labeled_data_dir}/labels"
    segmentor_saved_dir = segmentor_saved_dir # "G:/Zebrafish Blood Analysis/UNET/Single Cell Segmentation/Single Cell Segmentation/Model/1737636964/model_ckpt/epoch_60_loss_0.0881"
    pca_saved_dir = pca_saved_dir # "G:/Zebrafish Blood Analysis/YOLOV8/20250206 - Labeled Training/20250213_PCA.pkl"
    km_saved_dir = km_saved_dir # "G:/Zebrafish Blood Analysis/YOLOV8/20250206 - Labeled Training/20250213_KM.pkl"
    new_labels_save_dir = new_labels_save_dir # "G:/Zebrafish Blood Analysis/YOLOV8/20250206 - Labeled Training/YOLO Data/labels"

    # Parameters
    resize_to = 128

    # Preparation
    # List of labels
    list_of_labels = os.listdir(all_labeled_labels_dir)
    # PCA & KM models
    with open(pca_saved_dir, 'rb') as f:
        pca_model = pickle.load(f)
    with open(km_saved_dir, 'rb') as f:
        km_model = pickle.load(f)
    segmentor = tf.keras.models.load_model(segmentor_saved_dir)

    # idx = 100
    for idx in tqdm(range(0, len(list_of_labels))):
        image = cv2.imread(f"{all_labeled_images_dir}/{list_of_labels[idx].replace('txt', 'png')}", cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, -1)
        x_dim = image.shape[1]
        y_dim = image.shape[0]
        
        with open(f"{all_labeled_labels_dir}/{list_of_labels[idx]}", "r") as f:
            line = f.readlines()
    
        bbox_str = ""
        new_labels_save_dir_for_image = f"{new_labels_save_dir}/{list_of_labels[idx]}"
        for i, bbox_info in enumerate(line):
            bbox_info = bbox_info.replace("\n", "").split(" ")
            # label_orig = int(bbox_info[0])
        
            bbox_cent_x_eq = float(bbox_info[1])
            bbox_cent_y_eq = float(bbox_info[2])
            bbox_w_eq = float(bbox_info[3])
            bbox_h_eq = float(bbox_info[4])

            x1 = max(int(bbox_cent_x_eq * x_dim - bbox_w_eq * x_dim / 2), 0)
            x2 = min(int(bbox_cent_x_eq * x_dim + bbox_w_eq * x_dim / 2), x_dim)
            y1 = max(int(bbox_cent_y_eq * y_dim - bbox_h_eq * y_dim / 2), 0)
            y2 = min(int(bbox_cent_y_eq * y_dim + bbox_h_eq * y_dim / 2), y_dim)
            # x_diff = abs(x1 - x2)
            # y_diff = abs(y1 - y2)

            # Crop as square
            x1, x2, y1, y2 = convert_bbox_to_square(x1, x2, y1, y2, x_dim, y_dim)
            squared_bbox_img_cast_orig = image[y1:y2, x1:x2, :]
            # y_dim_orig = squared_bbox_img_cast.shape[0]
            # x_dim_orig = squared_bbox_img_cast.shape[1]
            pix_ratio = squared_bbox_img_cast_orig.shape[0] / resize_to

            # Histogram equalization
            squared_bbox_img_cast = skimage.exposure.equalize_hist(squared_bbox_img_cast_orig)

            # Resize to fit into the segmentation model
            new_size = (resize_to, resize_to, squared_bbox_img_cast.shape[2])
            squared_bbox_img_cast = resize(squared_bbox_img_cast, new_size)
            squared_bbox_img_cast_orig = resize(squared_bbox_img_cast_orig, new_size)

            # Predict the mask
            mask = segment_image(squared_bbox_img_cast, segmentor)

            # Convert to features
            mask_feature = measure_mask_properties(mask=mask, mask_prop=MASK_PROPERTIES, threshold=0.5, pix_ratio=pix_ratio)
            cell_feature = measure_cell_properties(image=squared_bbox_img_cast_orig, mask=mask, cell_prop=CELL_PROPERTIES)
            all_feature = mask_feature + cell_feature

            # PCA & KM
            predicted_label = km_transform(pca_transform(all_feature, pca_model), km_model)
            predicted_label = predicted_label[0]
            # print(predicted_label)

            # Label replacement
            if i != len(line) - 1:
                bbox_str += f"{str(predicted_label)} {str(bbox_info[1])} {str(bbox_info[2])} {str(bbox_info[3])} {str(bbox_info[4])}\n"
            else:
                bbox_str += f"{str(predicted_label)} {str(bbox_info[1])} {str(bbox_info[2])} {str(bbox_info[3])} {str(bbox_info[4])}"
        
        with open(new_labels_save_dir_for_image, "w") as f:
            f.write(bbox_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_img_path", help="Full path where the labeled images is stored", type=str)
    parser.add_argument("--labels_path", help="Full path where the labels are stored", type=str)
    parser.add_argument("--seg_model", help="Full path where the segmentation model is stored", type=str)
    parser.add_argument("--pca_model", help="Full path where the trained PCA model is stored", type=str)
    parser.add_argument("--km_model", help="Full path where the trained K-means clustering model is stored", type=str)
    parser.add_argument("--new_labels_save_to", help="Full path where the new labels will be saved to", type=str)
    args = parser.parse_args()

    labeled_img_dir = args.labeled_img_path
    labels_dir = args.labels_path
    segmentor_saved_dir = args.seg_model
    pca_saved_dir = args.pca_model
    km_saved_dir = args.km_model
    new_labels_save_dir = args.new_labels_save_to

    print("Labeled images path", labeled_img_dir)
    print("Original labels path", labels_dir)
    print("Segmentation model", segmentor_saved_dir)
    print("PCA model", pca_saved_dir)
    print("K-means clustering model", km_saved_dir)
    print("New labels save path", new_labels_save_dir)

    os.makedirs(new_labels_save_dir, exist_ok=True)

    main(labeled_img_dir, labels_dir, segmentor_saved_dir, pca_saved_dir, km_saved_dir, new_labels_save_dir)