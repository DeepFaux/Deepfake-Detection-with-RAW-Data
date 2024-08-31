import os
import numpy as np
import cv2
from natsort import natsorted
from glob import glob


def load_img(filename, norm=True):
    img = cv2.imread(filename).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if norm:
        img = img / 255.
    return img


def concat_rgb(input_path):
    row, col = [], []
    data_lis = {}
    for img_path in natsorted(glob(os.path.join(input_path, "*.jpg"))):
        img = load_img(img_path)

        row.append(img)
        if len(row) == 8:
            col.append(np.concatenate(row, axis=1))
            row = []
        if len(col) == 6:
            out_img = np.concatenate(col, axis=0)
            col = []
            data_lis[img_path.split("/")[-1].split("_")[0]] = out_img
    return data_lis


def demosaic_resize (raw):
    """Simple demosaicing to visualize RAW images
    Inputs:
     - raw: (h,w,4) RAW RGGB image normalized [0..1] as float32
    Returns: 
     - Simple Avg. Green Demosaiced RAW image with shape (h*2, w*2, 3)
    """
    assert raw.shape[-1] == 4
    shape = raw.shape
    
    red        = raw[:,:,0]
    green_red  = raw[:,:,1]
    green_blue = raw[:,:,2]
    blue       = raw[:,:,3]
    avg_green  = (green_red + green_blue) / 2
    image      = np.stack((red, avg_green, blue), axis=-1)
    image      = cv2.resize(image, (shape[1]*2, shape[0]*2))
    image = image.astype('float32') / 255.0
    return image
