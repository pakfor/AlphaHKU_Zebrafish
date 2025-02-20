import os
import shutil
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def main(mask_prop_csv_dir, trained_pca_save_dir, km_model_save_dir, km_cluster_plot_save_dir):
    mask_prop_csv_dir = mask_prop_csv_dir
    trained_pca_save_dir = trained_pca_save_dir
    km_model_save_dir = km_model_save_dir  # "G:/Zebrafish Blood Analysis/YOLOV8/20250206 - Labeled Training/20250213_KM.pkl"
    km_cluster_plot_save_dir = km_cluster_plot_save_dir

    # Load properties
    mask_prop_csv_dir = mask_prop_csv_dir
    mask_prop_df = pd.read_csv(mask_prop_csv_dir)
    mask_prop = mask_prop_df.to_numpy().astype('float32')

    # Load PCA
    with open(trained_pca_save_dir, 'rb') as f:
        trained_pca = pickle.load(f)

    # Use PCA
    pca_components = trained_pca.transform(mask_prop)

    # Train KM
    km_model = KMeans(n_clusters=2, random_state=0)
    km_model.fit(pca_components)

    # Use KM
    y_pred = km_model.predict(pca_components) # KMeans(n_clusters=2, random_state=9).fit_predict(pca_components)

    # Save plot
    if km_cluster_plot_save_dir is not None:
        plt.figure(dpi=1000)
        plt.scatter(list(pca_components[:,0]), list(pca_components[:,1]), marker='x', s=0.5, c=y_pred)
        plt.savefig(km_cluster_plot_save_dir)

    # Save trained KM
    if km_model_save_dir is not None:
        with open(km_model_save_dir, "wb") as f:
            pickle.dump(km_model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prop_csv_path", help="Full path to where the properties CSV is saved", type=str)
    parser.add_argument("--trained_pca_path", help="Full path to where the trained PCA model is saved", type=str)
    parser.add_argument("--save_km_model_to", help="Full path to where the K-means clustering model will be saved to", type=str)
    parser.add_argument("--save_cluster_plot_to", help="Full path to where the K-means clustering plot will be saved to", type=str)
    args = parser.parse_args()

    mask_prop_csv_dir = args.prop_csv_path
    trained_pca_save_dir = args.trained_pca_path
    km_model_save_dir = args.save_km_model_to
    km_cluster_plot_save_dir = args.save_cluster_plot_to

    print("Properties CSV input path", mask_prop_csv_dir)
    print("Trained PCA", trained_pca_save_dir)
    print("KM model save path", km_model_save_dir)
    print("KM plot save path", km_cluster_plot_save_dir)

    main(mask_prop_csv_dir, trained_pca_save_dir, km_model_save_dir, km_cluster_plot_save_dir)