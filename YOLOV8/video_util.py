# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 09:31:58 2024

@author: ALPHAHKU_HTC1
"""

import os
import shutil
import cv2
import glob2
import math
from tqdm import tqdm
import moviepy.editor as mpy
from moviepy.video.fx.all import crop
from moviepy.editor import ImageClip, concatenate_videoclips

class VideoUtil():
    def extract_frames(self, input_path, output_path):
        vidcap = cv2.VideoCapture(input_path)

        #print(f"Video: {input_path} ({self.args.fps} fps)")
        #print(f"Extracting frames to {output_path}")

        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(
                os.path.join(output_path, f"frame-{count:04}.jpg"),
                image,
            )
            success, image = vidcap.read()
            count += 1

    def crop_video(self, input_path, output_path, x_center, y_center, width, height):
        clip = mpy.VideoFileClip(input_path)
        cropped_clip = crop(clip, x_center=x_center,
                            y_center=y_center,
                            width=width,
                            height=height)
        cropped_clip.write_videofile(output_path)

    def frames_to_video(self, frame_folder, output_vid_path, frame_duration, fps):
        list_of_frame = glob2.glob(f"{frame_folder}/*")
        clips = [ImageClip(m.replace("\\", "/")).set_duration(frame_duration) for m in list_of_frame]
        concat_clip = concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(output_vid_path, fps=fps)

    def crop_video_in_batch(self, input_path, output_path, crop_width, crop_height):
        clip = mpy.VideoFileClip(input_path)
        clip_name = (input_path.split("/")[-1]).split(".")[0]
        (width, height) = clip.size
        crop_horizontal_count = math.ceil(width / crop_width)
        crop_vertical_count = math.ceil(height / crop_height)
        for i in tqdm(range(0, crop_horizontal_count)):
            for j in range(0, crop_vertical_count):
                if i < crop_horizontal_count - 1:
                    crop_center_x = i * crop_width + int(crop_width / 2)
                else:
                    crop_center_x = width - int(crop_width / 2)

                if j < crop_vertical_count - 1:
                    crop_center_y = j * crop_height + int(crop_height / 2)
                else:
                    crop_center_y = height - int(crop_height / 2)

                cropped_clip = crop(clip,
                                    x_center=crop_center_x, y_center=crop_center_y,
                                    width=crop_width, height=crop_height)
                cropped_clip.write_videofile(f"{output_path}/{clip_name}_{str(i)}_{str(j)}.mp4")
                
    def frame_select(self, interval_no, input_path, output_path):
        images = os.listdir(input_path)
        for image_name in images:
            if int(image_name[-8:-4]) % interval_no == 0:
                shutil.copy(input_path + "/" + image_name, output_path)
