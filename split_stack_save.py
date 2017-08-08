from __future__ import division
from os import path, listdir
import numpy as np
from skimage import io

filepath = r'J:\Jessica Burgstaller\LSM Analysis\LSM all raw data\to split by elena'
basepath, filename = path.split(filepath)
new_path = r'J:\Jessica Burgstaller\LSM Analysis\LSM all raw data\finished splits'

for filename in listdir(filepath):
    current_file = path.join(filepath, filename)
    print('Processing ' + current_file)
    stack = io.imread(current_file, plugin="tifffile") # reads as a numpy array, time, y, x
    length = int(stack.shape[0] / 2)
    # max_projection_total = np.amax(dffedbase_temp, axis=0).astype('float32')
    io.imsave(path.join(new_path, '{0}_part1.tif'.format(path.splitext(filename)[0])), stack[:length, ...].astype('float32'))
    io.imsave(path.join(new_path, '{0}_part2.tif'.format(path.splitext(filename)[0])), stack[length:, ...].astype('float32'))
    print('Saved.')

print("Finished")