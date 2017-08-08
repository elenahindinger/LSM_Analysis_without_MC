__author__ = 'helmbrecht'

import os
from filepicker import *
import skimage as sk
import skimage.io
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
import pickle
import pandas as pd
import time
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

import caiman as cm
import cv2
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
from caiman.motion_correction import MotionCorrect


def ds_image(images, resize_factor):
    frames_ds = sk.transform.downscale_local_mean(images, (1, resize_factor, resize_factor))
    return frames_ds


def uniform_filter3d(images, frames=(3,2,2)):
    # print("unfiform filtering")
    # return ndimage.filters.uniform_filter(images, size=(frames, 2, 2), output=None, mode='reflect', cval=0.0, origin=0)
    return ndimage.filters.uniform_filter(images, size=frames, output=None, mode='reflect', cval=0.0, origin=0)


def median_filter(images, frames=10):
    new_image = ndimage.filters.median_filter(images, size=(frames,1,1), footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
    return new_image


def set_baseline_to_zero(images):
    base = np.amin(images, axis=0)
    return images-base


def correct_first_frames(images, number_of_frames=99, percentile=40):
    percentile_image = np.percentile(images[number_of_frames:, :, :], percentile, 0)
    images[0:number_of_frames, :,:] = percentile_image
    return images


def column(matrix, i):
    return [row[i] for row in matrix]
