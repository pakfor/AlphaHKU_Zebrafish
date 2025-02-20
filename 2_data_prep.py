import os
import cv2
import glob2
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import skimage
import argparse


'''
From SCA_FeatExtrV4.m

Mask features:
Area, Centroid, ConvexArea, Eccentricity, EquivDiameter, Extent, MajorAxisLength, MinorAxisLength, Orientation, Perimeter, Solidity

Cell (BF) features:
Absorption density, Amplitude variance, Amplitude skewness, Amplitude kurtosis, Amplitude range, Maximum amplitude, Maximum absorption

'''

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
    return feature_table

def update_properties_table(table_to_be_updated, update_tables):
    dict_combined = dict()
    for update_table in update_tables:
        dict_combined.update(update_table)

    if table_to_be_updated is None:
        table_to_be_updated = dict.fromkeys(dict_combined.keys())

    for key in table_to_be_updated.keys():
        if table_to_be_updated[key] is None:
            table_to_be_updated[key] = [dict_combined[key]]
        else:
            table_to_be_updated[key] += [dict_combined[key]]
    return table_to_be_updated

# Cell features extraction
def measure_cell_properties(image, mask, cell_prop):
    feature_table = dict()
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
    feature_table['absorption_density'] = averaged_absorbance
    feature_table['variance'] = variance
    feature_table['skewness'] = skewness
    feature_table['kurtosis'] = kurtosis
    feature_table['amplitude_range'] = amp_range
    feature_table['amplitude_max'] = amp_max
    return feature_table

def main(orig_cell_img_master_dir, segmentor_dir, prop_save_dir, mask_save_dir, masked_cell_save_dir):
    orig_cell_img_master_dir = orig_cell_img_master_dir # "G:/Zebrafish Blood Analysis/YOLOV8/20250216 - Demo/ZBF 2/Cell"
    segmentor_dir = segmentor_dir # "G:/Zebrafish Blood Analysis/YOLOV8/20250216 - Demo/Model/U-Net/1737636964/epoch_60_loss_0.0881"
    prop_save_dir = prop_save_dir # "G:/Zebrafish Blood Analysis/YOLOV8/20250216 - Demo/ZBF 2/Properties - ZBF2.csv"
    mask_save_dir = mask_save_dir
    masked_cell_save_dir = masked_cell_save_dir

    if mask_save_dir is not None:
        os.makedirs(mask_save_dir, exist_ok=True)
    if masked_cell_save_dir is not None:
        os.makedirs(masked_cell_save_dir, exist_ok=True)

    orig_cell_img_dir_list = glob2.glob(f"{orig_cell_img_master_dir}/*.png")
    orig_cell_img_dir_list = [x.replace("\\", "/") for x in orig_cell_img_dir_list]
    segmentor = tf.keras.models.load_model(segmentor_dir)
    all_features = None

    for orig_cell_img_dir in tqdm(orig_cell_img_dir_list):
        img_orig = cv2.imread(orig_cell_img_dir, cv2.IMREAD_GRAYSCALE)
        img = skimage.exposure.equalize_hist(img_orig)

        img_to_pred = np.expand_dims(img, axis=-1)
        img_to_pred = tf.cast(img_to_pred, tf.float32)
        img_to_pred = tf.expand_dims(img_to_pred, axis=0)

        mask = segmentor.predict(img_to_pred, verbose=0)
        mask = np.squeeze(mask)

        # Binarize and extract the largest area within the mask
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        labeled = skimage.measure.label(mask)
        regions = skimage.measure.regionprops(labeled)
        mask_with_largest_area = (labeled == (1 + np.argmax([i.area for i in regions]))).astype("float32")
        masked_img = np.multiply(img_orig, mask_with_largest_area)

        # Feature extraction
        # Mask features extraction
        pix_ratio = float(orig_cell_img_dir.split(".")[0].split("_")[-2]) / float(orig_cell_img_dir.split(".")[0].split("_")[-1])
        mask_feature_table = measure_mask_properties(mask=mask, mask_prop=MASK_PROPERTIES, threshold=0.5, pix_ratio=pix_ratio)
        # Cell features extraction
        cell_feature_table = measure_cell_properties(image=img_orig, mask=mask_with_largest_area, cell_prop=CELL_PROPERTIES)
        all_features = update_properties_table(all_features, [mask_feature_table, cell_feature_table])

        # Save mask/masked cell as PNG
        if masked_cell_save_dir:
            plt.imsave(f"{masked_cell_save_dir}/{orig_cell_img_dir.split('/')[-1].replace('.png', '_masked_cell.png')}", masked_img, cmap='gray')
        if mask_save_dir:
            plt.imsave(f"{mask_save_dir}/{orig_cell_img_dir.split('/')[-1].replace('.png', '_mask.png')}", mask_with_largest_area, cmap='gray')

    # Save properties as CSV
    if prop_save_dir:
        all_features_df = pd.DataFrame(data=all_features)
        all_features_df.to_csv(prop_save_dir, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_input_path", help="Full path where the cell images are stored", type=str)
    parser.add_argument("--seg_model", help="Full path where the segmentation model is stored", type=str)
    parser.add_argument("--save_prop_to", help="Full path where the extracted properties will be saved to", type=str)
    parser.add_argument("--save_mask_to", help="Full path where the predicted mask will be saved to", type=str)
    parser.add_argument("--save_masked_cell_to", help="Full path where the masked cell image will be saved to", type=str)
    args = parser.parse_args()

    orig_cell_img_master_dir = args.image_input_path
    segmentor_dir = args.seg_model
    prop_save_dir = args.save_prop_to
    mask_save_dir = args.save_mask_to
    masked_cell_save_dir = args.save_masked_cell_to

    print("Image input path", orig_cell_img_master_dir)
    print("Segmentation model", segmentor_dir)
    print("Properties save to", prop_save_dir)
    print("Mask save to", mask_save_dir)
    print("Masked cell save to", masked_cell_save_dir)

    main(orig_cell_img_master_dir, segmentor_dir, prop_save_dir, mask_save_dir, masked_cell_save_dir)