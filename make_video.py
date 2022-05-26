import cv2
import numpy as np
from glob import glob
import sys
import os
import skvideo.io
from tqdm import tqdm

def save_files(x):
    f = cv2.imread(x[0])
    h, w = f.shape[0], f.shape[1]
    out_video =  np.empty([len(x), h, w, 3], dtype = np.uint8)
    out_video =  out_video.astype(np.uint8)
    for i, s in enumerate(tqdm(x)):
        img = cv2.imread(s)
        out_video[i] = img
    # Writes the the output image sequences in a video file
    skvideo.io.vwrite("video/video.mp4", out_video)


if __name__ == '__main__':
    if len(sys.argv)==1:
        raise Exception('No folder')
    folder_name = str(sys.argv[1])
    x = glob(folder_name+'*.png')
    x = sorted(x, key=lambda i: int(i.split('/')[-1].split('.')[0].split('_')[-1]))
    save_files(x)