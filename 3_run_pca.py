import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import argparse


def main(mask_prop_csv_dir, pca_model_save_dir, pca_scatter_plot_save_dir):
    # Cell features import
    mask_prop_csv_dir = mask_prop_csv_dir # "G:/Zebrafish Blood Analysis/YOLOV8/20250216 - Demo/ZBF 2/Properties - ZBF2.csv"
    mask_prop_df = pd.read_csv(mask_prop_csv_dir)
    mask_prop = mask_prop_df.to_numpy().astype('float32')

    # Save PCA
    pca_model_save_dir = pca_model_save_dir # "G:/Zebrafish Blood Analysis/YOLOV8/20250206 - Labeled Training/20250213_PCA.pkl"

    # Train PCA
    pca_model = PCA(n_components=2)
    pca_model.fit(mask_prop)

    # Run with trained PCA
    pca_components = pca_model.transform(mask_prop)

    # Save PCA plot
    if pca_scatter_plot_save_dir is not None:
        plt.figure(dpi=1000)
        plt.scatter(list(pca_components[:,0]), list(pca_components[:,1]), marker='x', s=0.5)
        plt.savefig(pca_scatter_plot_save_dir)

    # Save trained PCA
    if pca_model_save_dir is not None:
        with open(pca_model_save_dir, "wb") as f:
            pickle.dump(pca_model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prop_csv_path", help="Full path to where the properties CSV is saved", type=str)
    parser.add_argument("--save_pca_model_to", help="Full path to where the PCA model will be saved", type=str)
    parser.add_argument("--save_pca_plot_to", help="Full path to where the PCA scatter plot will be saved", type=str)
    args = parser.parse_args()

    mask_prop_csv_dir = args.prop_csv_path
    pca_model_save_dir = args.save_pca_model_to
    pca_scatter_plot_save_dir = args.save_pca_plot_to

    print("Properties CSV input path", mask_prop_csv_dir)
    print("PCA model save path", pca_model_save_dir)
    print("PCA plot save path", pca_scatter_plot_save_dir)

    main(mask_prop_csv_dir, pca_model_save_dir, pca_scatter_plot_save_dir)