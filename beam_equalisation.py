from __future__ import division
import os
import numpy as np
from skimage import io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import skimage
from skimage import transform as tf

big_folder = r'K:\lsm analysis\big stack\CORRECTED'
basepath, folder = os.path.split(big_folder)
folder_list = os.listdir(big_folder)
for sub_folder in folder_list:
    sub_folder_path = os.path.join(big_folder, sub_folder)
    sub_folder_list = os.listdir(sub_folder_path)
    out_dir = os.path.join(basepath, 'EQUALISED', sub_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for stackfile in sub_folder_list:
        stack_path = os.path.join(sub_folder_path, stackfile)
        print 'Currently Processing:', sub_folder, stackfile
        stack = io.imread(stack_path, plugin='tifffile')
        mean_subtraction = stack[:-10, ...].mean(axis=0)
        new = (stack - mean_subtraction).astype('float32')
        savename = os.path.join(out_dir, (stackfile.replace('_corrected.tiff', '_equalised.tiff')))
        io.imsave(savename, new)


