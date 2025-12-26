import cv2
import numpy as np
from skimage import exposure
from config import IMAGE_SIZE

def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def resize(img, size=IMAGE_SIZE):
    return cv2.resize(img, size)

def normalize(img):
    return img.astype('float32') / 255.0

def equalize_hist(img):
    if len(img.shape) == 3:
        img = to_gray(img)
    return exposure.equalize_adapthist(img)