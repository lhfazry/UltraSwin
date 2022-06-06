from vidaug import augmentors as va
import random
import os
import pathlib
import collections

import numpy as np
import torch
import cv2
from datasets.EchoSet import loadvideo


def save_video(name, video, fps):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    data = cv2.VideoWriter(name, fourcc, float(fps), (video.shape[1], video.shape[2]))

    for v in video:
        data.write(v)

    data.release()

dataset_dir = '/Users/fazry/Downloads/EchoNet'
video_dir = os.path.join(dataset_dir, 'Videos')
files = os.listdir(video_dir)

vid_augs1 = va.Sequential([va.RandomRotate(degrees=10)])
vid_augs2 = va.Sequential([va.HorizontalFlip()]) 
vid_augs3 = va.Sequential([va.VerticalFlip()])  
vid_augs4 = va.Sequential([va.GaussianBlur(0.75)])

video1 = loadvideo(os.path.join(video_dir, files[10])).astype(np.float32)
#print(video1.shape)
video1 = np.asarray(vid_augs1(video1)).astype(np.uint8)
#print(video1.shape)

video2 = loadvideo(os.path.join(video_dir, files[11])).astype(np.float32)
video2 = np.asarray(vid_augs2(video2)).astype(np.uint8)

video3 = loadvideo(os.path.join(video_dir, files[12])).astype(np.float32)
video3 = np.asarray(vid_augs3(video3)).astype(np.uint8)

video4 = loadvideo(os.path.join(video_dir, files[13])).astype(np.float32) #(C, F, H, W)   
video4 = video4.transpose((1, 2, 3, 0)) #(F, H, W, C) 
video4 = np.asarray(vid_augs4(video4)).astype(np.uint8)


#save_video(files[10], video1, 50)
#save_video(files[11], video2, 50)
#save_video(files[12], video3, 50)
save_video(files[13], video4, 50)

# Load video into np.array
#video = loadvideo(path).astype(np.float32)

