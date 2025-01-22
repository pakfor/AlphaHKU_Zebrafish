from skimage import io
import matplotlib.pyplot as plt
import glob2
import os
import numpy as np
from tqdm import tqdm

tif_list = glob2.glob("G:/Zebrafish Blood Analysis/20241119 Videos/zebrafish_20241119/*/*/*.tif")

for vid in tqdm(tif_list):
    video_dir = vid.replace("\\", "/")
    tif = io.imread(video_dir)
    video_name = f"{video_dir.split('/')[-3]}_{video_dir.split('/')[-2]}"
    save_to = f"G:/Zebrafish Blood Analysis/20241119 Videos/Frames/{video_name}"
    if not os.path.exists(save_to):
        os.makedirs(save_to, exist_ok=True)

    for i in tqdm(range(0, tif.shape[0])):
        frame = tif[i, :, :]
        frame = frame / 65535
        frame = frame.astype("float32")
        plt.imsave(f"{save_to}/{video_name}_frame_{str(i).zfill(4)}.png", frame, vmin=0.0, vmax=1, cmap='gray')

#%% Random select for label

import random
import os
import shutil

select_n = 20
copy_to = "G:/Zebrafish Blood Analysis/20241119 Videos/Frames to Label"
main_dir = "G:/Zebrafish Blood Analysis/20241119 Videos/Frames"
video_dirs = os.listdir(main_dir)

for vid_folder in video_dirs:
    frames_list = os.listdir(f"{main_dir}/{vid_folder}")
    selected_frames_list = random.sample(frames_list, select_n)
    for selected_frame in selected_frames_list:
        shutil.copy(f"{main_dir}/{vid_folder}/{selected_frame}",
                    f"{copy_to}/{selected_frame}")