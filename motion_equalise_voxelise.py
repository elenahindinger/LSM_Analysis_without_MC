from __future__ import division
import os
import numpy as np
from skimage import io
from natsort import natsorted
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
from natsort import natsorted
from motion_correction_functions import *
from functions import *
# from pca_trial import *

__author__ = 'ehindinger'

''' ALL THE FUNCTIONS WE NEED LATER ON IN THIS SCRIPT '''

def motion_preprocessing(folder, new_path):
    print 'PRE-PROCESSING FOR MOTION CORRECTION'
    filelist = os.listdir(folder)
    for file in filelist:
        print 'Processing ' + file
        filepath = os.path.join(folder, file)
        stack = io.imread(filepath, plugin="tifffile")
        outdir = os.path.join(new_path, 'split by plane', file)  # file[:18] for just date and genotype
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # use this in future
        for j in np.arange(30):
            print 'Processing plane %s' % str(j)
            list_of_plane = []
            for i in np.arange(j, 36000, 30):
                list_of_plane.append(stack[i, ...])
            plane1 = np.stack(list_of_plane, axis=0).astype('float32')
            io.imsave(os.path.join(outdir, 'plane_%s.tiff' % str(j+1)), plane1)
    print 'FINISHED: PRE-PROCESSING FOR MOTION CORRECTION'


def motion_correction(input_folder):
    big_folder = os.path.join(input_folder, 'split by plane')
    print 'CORRECTING FOR MOTION ARTEFACTS'
    basepath, folder = os.path.split(big_folder)
    folder_list = os.listdir(big_folder)
    for sub_folder in folder_list:
        sub_folder_path = os.path.join(big_folder, sub_folder)
        sub_folder_list = natsorted(os.listdir(sub_folder_path))
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
            savename = os.path.join(out_dir, (sub_folder + '_' + stack.replace('.tiff', '_corrected.tiff')))
            sk.io.imsave(savename, image_to_save.astype('float32'), plugin='tifffile')

    print 'FINISHED: MOTION CORRECTION.'


# def equalise(big_folder):
#     print 'EQUALISING BEAM DIFFERENCES'
#     basepath, folder = os.path.split(big_folder)
#     folder_list = os.listdir(big_folder)
#     for sub_folder in folder_list:
#         sub_folder_path = os.path.join(big_folder, sub_folder)
#         sub_folder_list = os.listdir(sub_folder_path)
#         out_dir = os.path.join(basepath, 'EQUALISED', sub_folder)
#         if not os.path.exists(out_dir):
#             os.makedirs(out_dir)
#
#         for stackfile in sub_folder_list:
#             stack_path = os.path.join(sub_folder_path, stackfile)
#             print 'Currently Processing:', sub_folder, stackfile
#             stack = io.imread(stack_path, plugin='tifffile')
#             mean_subtraction = stack[:-10, ...].mean(axis=0)
#             new = (stack - mean_subtraction).astype('float32')
#             savename = os.path.join(out_dir, (stackfile.replace('_corrected.tiff', '_equalised.tiff')))
#             io.imsave(savename, new)
#     print 'FINISHED: EQUALISING BEAM DIFFERENCES'


def recombination(big_folder, input_file):
    print 'RECOMBINING STACKS, OUTPUT FROM ', input_file
    right_folder = os.path.join(big_folder, input_file)
    filelist = natsorted(os.listdir(right_folder))
    outdir = os.path.join(big_folder, 'RECOMBINED')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for sub_folder in filelist:
        folder_containing_single_planes = os.path.join(right_folder, sub_folder)
        plane_list = natsorted(os.listdir(folder_containing_single_planes))
        list_of_planes = []
        for i in plane_list:
            ipath = os.path.join(folder_containing_single_planes, i)
            plane = io.imread(ipath, plugin='tifffile')
            list_of_planes.append(plane)
        # zipped_list = zip(list_of_planes)  # zips so that first planes are all together, then second planes, then third ...
        # combined = [np.concatenate(arrays) for arrays in zipped_list]  # concatenates them into little arrays
        # recombined = np.concatenate(combined)  # makes one big array
        new_list = []
        for plane in np.arange(list_of_planes[0].shape[0]):
            for array in list_of_planes:
                new_list.append(array[plane])
        recombined = np.stack(new_list, axis=0)
        basepath, final_folder = os.path.split(folder_containing_single_planes)
        io.imsave(os.path.join(outdir, (final_folder + '_' + str.lower(input_file) + '_recombined.tiff')), recombined.astype('float32'))
    print 'FINISHED: RECOMBINING PLANES OF INPUT ', input_file


# def do_pca(new_path):
#     print 'TRYING PCA'
#     correct_folder = os.path.join(new_path, 'RECOMBINED')
#     folderlist = os.listdir(correct_folder)
#     gr_original = os.path.join(correct_folder, folderlist[0])
#     het_original = os.path.join(correct_folder, folderlist[3])
#
#     # whole stacks
#     gr_stack = io.imread(gr_original, plugin="tifffile")  # reads as a numpy array, time, y, x
#     het_stack = io.imread(het_original, plugin="tifffile")  # reads as a numpy array, time, y, x
#
#     # # downscale to 5x5x5 voxels
#     # gr_voxel = tf.downscale_local_mean(gr_stack, (1, 5, 5))
#     # het_voxel = tf.downscale_local_mean(het_stack, (1, 5, 5))
#
#     # # calculate df/f
#     # gr_dff = dffb(gr_stack)
#     # het_dff = dffb(het_stack)
#
#     # # calculate df/f of voxelised stacks
#     # gr_dff = dffb(gr_voxel)
#     # het_dff = dffb(het_voxel)
#
#     transformed_stack = pca_prep_part2(gr_stack, het_stack)
#     new_pca_plot(transformed_stack, new_path)

    print 'FINISHED PCA- IT WORKED!'

''' Calling functions for serial processing '''

# ''' Trying small input folder '''
# trial_input_folder = r'K:\lsm analysis\trial input\today'
# trial_new_path = r'K:\lsm analysis\small trial stacks new pipeline'
#
# # motion_preprocessing(trial_input_folder, trial_new_path)
#
# motion_correction(trial_new_path)
#
# equalise(os.path.join(trial_new_path, 'CORRECTED'))
#
# recombination(trial_new_path, 'EQUALISED')
# recombination(trial_new_path, 'CORRECTED')

''' Trying big stack input folder '''
input_folder = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis\10min input stacks\part2'
new_path = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis'

motion_preprocessing(input_folder, new_path)

motion_correction(new_path)

# equalise(os.path.join(new_path, 'CORRECTED'))

recombination(new_path, 'CORRECTED')
# recombination(new_path, 'EQUALISED')


print 'IT WORKED, ALL DONE !!!'

# filepaths
