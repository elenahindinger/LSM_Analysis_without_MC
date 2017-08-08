from __future__ import division
import os
import numpy as np
import pandas as pd
from skimage import io
from natsort import natsorted
from filepicker import *
import skimage as sk
from PIL import Image
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import caiman as cm
import cv2
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
from caiman.motion_correction import MotionCorrect
from sklearn.decomposition import PCA, FastICA
from skimage import io
from skimage import transform as tf
import scipy.ndimage as nd
from motion_correction_functions import *
from functions import *
from DRT_functions import *


__author__ = 'ehindinger'

length = 20
frames = 2400
rolling_duration = 15
voxel_size = (1, 4, 4)

''' ALL THE FUNCTIONS WE NEED LATER ON IN THIS SCRIPT '''


def motion_preprocessing(input_folder, new_path, length=length):
    print 'PRE-PROCESSING FOR MOTION CORRECTION'
    filelist = natsorted(os.listdir(input_folder))
    raw_frames = length * 60 * 30 * 2
    for file in filelist:
        if file.endswith('BTF'):
            print 'Processing ' + file
            filepath = os.path.join(input_folder, file)
            stack = io.imread(filepath, plugin="tifffile")
            outdir = os.path.join(new_path, 'split by plane', file[:-4])  # file[:18] for just date and genotype
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            # use this in future
            for j in np.arange(30):
                print 'Processing plane %s' % str(j)
                list_of_plane = []
                for i in np.arange(j, raw_frames, 30):
                    list_of_plane.append(stack[i, ...])
                plane1 = np.stack(list_of_plane, axis=0).astype('float32')
                io.imsave(os.path.join(outdir, file.replace('.BTF', ('_plane_%s.tiff' % str(j+1)))), plane1.astype('float32'))
        else:
            print 'Skipping thumbs.'
            pass
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
            if stack.endswith('tiff'):
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
                savename = os.path.join(out_dir, stack.replace('.tiff', '_corrected.tiff'))
                sk.io.imsave(savename, image_to_save.astype('float32'), plugin='tifffile')
            else:
                pass
    print 'FINISHED: MOTION CORRECTION.'


def recombination_mc_dff_max(big_folder):
    print 'RECOMBINING STACKS, OUTPUT FROM CORRECTED'
    right_folder = os.path.join(big_folder, 'CORRECTED')
    filelist = natsorted(os.listdir(right_folder))
    outdir = os.path.join(big_folder, 'RECOMBINED')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for sub_folder in filelist:
        print 'Processing ', sub_folder
        folder_containing_single_planes = os.path.join(right_folder, sub_folder)
        plane_list = natsorted(os.listdir(folder_containing_single_planes))
        print 'Recombining basic stack ...'
        list_of_planes = []
        for i in plane_list:
            if i.endswith('tiff'):
                ipath = os.path.join(folder_containing_single_planes, i)
                plane = io.imread(ipath, plugin='tifffile')
                list_of_planes.append(plane)
            else:
                pass
        new_list = []
        for plane in np.arange(list_of_planes[0].shape[0]):
            for array in list_of_planes:
                new_list.append(array[plane])
        recombined = np.stack(new_list, axis=0)
        io.imsave(os.path.join(outdir, sub_folder + '_mc_recombined.tiff'), recombined.astype('float32'))
        print 'Calculating dff ...'
        dff_stack = dffb(recombined)
        dff_out = os.path.join(big_folder, 'MOTION CORRECTED DFF STACKS')
        if not os.path.exists(dff_out):
            os.makedirs(dff_out)
        io.imsave(os.path.join(dff_out, sub_folder + '_mc_dff_recombined.tiff'), dff_stack.astype('float32'))
        max_out = os.path.join(big_folder, 'MC DFF MAX STACKS FOR VIDEOS')
        if not os.path.exists(max_out):
            os.makedirs(max_out)
        print 'Calculating max proj of dff over time ...'
        io.imsave(path.join(max_out, sub_folder + '_mc_dff_max_over_time.tiff'),
                  split_stack_merge(dff_stack).astype('float32'))

    print 'FINISHED: RECOMBINING PLANES OF CORRECTED INPUT'


def run_all_DRTS(genotype, corrected_path, big_out, frames=frames,
                 voxel_size=voxel_size, rolling_duration=rolling_duration):
    # USE THIS FOR A BIG ANALYSIS
    print 'Working on ', genotype
    outdir = os.path.join(big_out, genotype)
    voxelised_stack = filter_and_voxelise(corrected_path, rolling_duration)  # read, filter and voxelise ONE corrected stack
    # voxelised_stack = merge_filter_and_voxelise(part1_path, part2_path, rolling_duration)  # read, filter and voxelise TWO PARTS, COMBINE BEFORE FILTERING !

    pre_processed = pca_prep(voxelised_stack)  # prepare voxelised stack for dimensionality reduction by reshaping matrix

    pca_plots(pre_processed, outdir, voxelised_stack, genotype=genotype, number_of_components=6, frames=frames)  # run pca, n_components stands for number saved in plots, not for input to pca function

    total_ICA(pre_processed, voxelised_stack=voxelised_stack, outdir=outdir, genotype=genotype, frames=frames)  # run ica for components 3, 4, 5, 6, 7

    run_NMF(pre_processed=pre_processed, voxelised_stack=voxelised_stack, genotype=genotype, outdir=outdir, numco=20, frames=frames)  # run NMF on 20 components

    run_isomap(pre_processed=pre_processed, genotype=genotype, outdir=outdir, frames=frames)  # run isomap on 5 components

    # run_tsne(pre_processed=pre_processed, genotype=genotype, outdir=outdir, numco=2, frames=frames)  # run t-SNE on 2 components

    print 'ALL DONE WITH DIMENSIONALITY REDUCTION FOR %s!!!' % (genotype)
    print 'and it probably looks shit, no offense...'

''' Trying big stack input folder '''
input_folder = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\LAST DAY ANALYSIS\raw input data'
new_path = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\LAST DAY ANALYSIS'

motion_preprocessing(input_folder, new_path)

motion_correction(new_path)

recombination_mc_dff_max(new_path)

make_montage(new_path)

print 'Running DRTs'
geno_list = ['gr', 'het']
counter = 0
drt_out = os.path.join(new_path, 'DRT OUTPUT')
if not os.path.exists(drt_out):
    os.makedirs(drt_out)
for fish in natsorted(os.listdir(os.path.join(new_path, 'CORRECTED'))):
    print 'Running DRT on fish ', fish
    print 'Assigned genotype: ', geno_list[counter]
    fish_path = os.path.join(new_path, 'CORRECTED', fish)
    run_all_DRTS(genotype=geno_list[counter], corrected_path=fish_path, big_out=drt_out)
    counter += 1
print 'IT WORKED, ALL DONE !!!'
