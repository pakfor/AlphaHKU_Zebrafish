# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 05:51:03 2024

@author: ALPHAHKU_HTC1
"""

import moviepy.editor as mpy
from moviepy.video.fx.all import crop, resize
import moviepy.video.fx.all as vfx

clip = mpy.VideoFileClip("G:/Zebrafish Blood Analysis/YOLOV8/Raw Data/Video/20240830_004.mp4")
(w, h) = clip.size
cropped_clip = crop(clip, width=320, height=320, x_center=w/4, y_center=h/2)
#cropped_clip = cropped_clip.fx(vfx.lum_contrast, lum=1, contrast=2, contrast_thr=126)
#cropped_clip = cropped_clip.resize((720, 720))
cropped_clip.write_videofile("G:/Zebrafish Blood Analysis/YOLOV8/Raw Data/Video/20240830_004_cropped2.mp4")
