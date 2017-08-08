__author__ = 'helmbrecht'

import os
from filepicker import *
import skimage as sk
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
from motion_correction_functions import *

big_folder = r'J:\Jessica Burgstaller\LSM Analysis\LSM all raw data gridstacks no zoom\motion test'
basepath, folder = os.path.split(big_folder)
folder_list = os.listdir(big_folder)
for sub_folder in folder_list:
    sub_folder_path = os.path.join(big_folder, sub_folder)
    sub_folder_list = os.listdir(sub_folder_path)
    out_dir = os.path.join(basepath, 'CORRECTED', sub_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for stack in sub_folder_list:
        stack_path = os.path.join(sub_folder_path, stack)
        print 'Currently Processing:', sub_folder, stack

        now = time.time()
        ''' ONLY RIGID TRANSFORM SEEMS TO WORK REALLY NICE '''

        filex = cm.load(stack_path)
        filex = filex-np.min(filex)
        filex[np.where(filex>1000)]=1000

        mc = filex.motion_correct(3,3)[0]

        dsfactor = 1
        m = mc.resize(1,1,dsfactor)
        num_frames = m.shape[0]

        inputImage = m[10]
        mapX = np.zeros((num_frames,inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
        mapY = np.zeros((num_frames,inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
        templ = np.median(m,0)
        map_orig_x = np.zeros((inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)
        map_orig_y = np.zeros((inputImage.shape[0], inputImage.shape[1]), dtype=np.float32)

        for j in range(inputImage.shape[0]):
            for i in range(inputImage.shape[1]):
                map_orig_x[j,i] = i
                map_orig_y[j,i] = j

        for k in range(num_frames):
            pyr_scale = .5; levels = 3; winsize = 20;  iterations = 15; poly_n = 7; poly_sigma = 1.2/5; flags = 0;
            flow = cv2.calcOpticalFlowFarneback(templ,m[k],None,pyr_scale, levels, winsize, iterations, poly_n,
                                                poly_sigma, flags)
            mapX[k, :] = map_orig_x + flow[:, :, 0]
            mapY[k, :] = map_orig_y+ flow[:, :, 1]

        num_frames = mc.shape[0]
        mapX_res = cm.movie(mapX).resize(1,1, 1/dsfactor)
        mapY_res = cm.movie(mapY).resize(1,1, 1/dsfactor)

        fact = np.max(m)
        bl = np.min(m)
        times = []
        new_ms = np.zeros(mc[:num_frames].shape)
        for counter, mm in enumerate(mc[:num_frames]):
            new_img = cv2.remap(mm, mapX_res[counter], mapY_res[counter], cv2.INTER_CUBIC, None, cv2.BORDER_CONSTANT)
            new_ms[counter] = new_img

        image_to_save = np.copy(new_ms)
        image_to_save[np.where(np.isnan(image_to_save) == True)] = 0
        savename = os.path.join(out_dir, (sub_folder + '_' + stack.replace('.tiff', '_corrected.tif')))
        sk.io.imsave(savename, image_to_save.astype('int16'), plugin='tifffile')
        print 'Done Saving'

print 'DONE'
