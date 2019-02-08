import os
import re
import cv2
import numpy as np
from skimage.io import imsave, imread
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from utils.params import IMG_TARGET_ROWS, IMG_TARGET_COLS


def norm(img, mean, std):
    img -= mean
    img /= std
    return img


def preprocess(img):
    img = cv2.resize(img, (IMG_TARGET_COLS, IMG_TARGET_ROWS))
    return img


def calc_mean_std(imgs_train):
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    return mean, std


def load_img(path, grayscale=False, target_size=None):
    if grayscale:
        img = imread(path)
    else:
        img = cv2.imread(path)
    if target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    return img


def list_images(directory, ext='jpg|jpeg|bmp|png|tif'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]


def gray_to_RGB(img):
    return np.dstack((img, img, img))
