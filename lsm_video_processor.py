from __future__ import division
from os import path, listdir
import numpy as np
from skimage import io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import skimage
from functions import dffb, split_stack_merge

filepath = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\motion corrected dff stacks'
basepath, filename = path.split(filepath)
new_path = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\videos'

for filename in listdir(filepath):
    current_file = path.join(filepath, filename)
    print('Processing ' + current_file)

    stack = io.imread(current_file, plugin="tifffile") # reads as a numpy array, time, y, x
    # dffedbase_temp = dffb(stack)
    max_projections_of_dffb_over_time = split_stack_merge(stack).astype('float32')
    # max_projections_over_time = split_stack_merge(stack).astype('float32')
    # dffedbase = dffedbase_temp.astype('float32')

    # max_projection_total = np.amax(dffedbase_temp, axis=0).astype('float32')
    # io.imsave(path.join(new_path, '{0}_dffedbase.tif'.format(path.splitext(filename)[0])), dffedbase)
    # io.imsave(path.join(new_path, '{0}_dffedbase_mean.tif'.format(path.splitext(filename)[0])), dffedbase.mean(0))
    # # io.imsave(path.join(new_path, '{0}_dffedbase_max_projection.tif'.format(path.splitext(filename)[0])),
    # #           max_projection_total)
    io.imsave(path.join(new_path, '{0}_max_dff_over_time.tif'.format(path.splitext(filename)[0])),
              max_projections_of_dffb_over_time)
    # io.imsave(path.join(new_path, '{0}_max_over_time.tif'.format(path.splitext(filename)[0])),
    #           max_projections_over_time)
print("Finished")