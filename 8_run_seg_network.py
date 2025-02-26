import os
import cv2
import numpy as np
import skimage
import tensorflow as tf
from tqdm import tqdm
import argparse


def preprocess_from_path(image_path, preprocess_method, hist_ref=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_orig = img.copy()
    img_orig = (img_orig - np.min(img_orig)) / (np.max(img_orig) - np.min(img_orig))
    if preprocess_method == "hist_eq":
        img = skimage.exposure.equalize_hist(img)
    elif preprocess_method == "hist_match":
        ref_img = cv2.imread(hist_ref, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, axis=-1)
        ref_img = np.expand_dims(ref_img, axis=-1)
        img = skimage.exposure.match_histograms(img, ref_img, channel_axis=-1)
        img = np.squeeze(img, axis=-1)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    else:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img, img_orig

def main(model_path, image_to_pred_path, pred_montage_save_path, preprocess_method, hist_ref):
    model_path = model_path
    image_to_pred_path = image_to_pred_path
    # pred_mask_save_path = pred_mask_save_path
    pred_montage_save_path = pred_montage_save_path
    preprocess_method = preprocess_method
    hist_ref = hist_ref

    # os.makedirs(pred_mask_save_path, exist_ok=True)
    os.makedirs(pred_montage_save_path, exist_ok=True)

    model = tf.saved_model.load(model_path)

    img_to_pred_list = os.listdir(image_to_pred_path)

    for img_name in tqdm(img_to_pred_list):
        img_to_pred, img_orig = preprocess_from_path(f"{image_to_pred_path}/{img_name}", preprocess_method, hist_ref)
        img_to_pred = np.expand_dims(img_to_pred, axis=0)
        img_to_pred = np.expand_dims(img_to_pred, axis=-1)
        img_to_pred = tf.cast(img_to_pred, tf.float32)
        pred_mask = model(img_to_pred)
        pred_mask = np.squeeze(pred_mask)

        montage = np.zeros([2, 128, 128], dtype = np.float32)
        montage[0, :, :] = img_orig
        montage[1, :, :] = pred_mask
        montage_img = skimage.util.montage(montage)
        skimage.io.imsave(f"{pred_montage_save_path}/{img_name.replace('png', 'tif')}", montage_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Full path to the trained segmentation model", type=str)
    parser.add_argument("--image_to_pred_path", help="Full path to where all the images to be predicted are saved", type=str)
    parser.add_argument("--pred_montage_save_path", help="Full path to where all the prediction results will be saved", type=str)
    parser.add_argument("--preprocess_method", help="Preprocessing method", type=str)
    parser.add_argument("--hist_ref", help="Reference image for histogram matching", type=str)
    args = parser.parse_args()

    model_path = args.model_path
    image_to_pred_path = args.image_to_pred_path
    pred_montage_save_path = args.pred_montage_save_path
    preprocess_method = args.preprocess_method
    hist_ref = args.hist_ref

    print("Segmentation model", model_path)
    print("Image to be predicted", image_to_pred_path)
    print("Prediction result (montage) to be saved", pred_montage_save_path)
    print("Preprocessing method", preprocess_method)
    print("Histogram reference", hist_ref)

    main(model_path, image_to_pred_path, pred_montage_save_path, preprocess_method, hist_ref)