import os
import numpy as np
from skimage import io
from os import path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it
from natsort import natsorted
from skimage import transform as tf


def read_and_voxelise(stack_path, x=5, y=5):
    print 'Reading ', stack_path
    fish = io.imread(stack_path, plugin='tifffile')
    print 'Voxelising file.'
    voxelised_stack = tf.downscale_local_mean(fish, (1, x, y))
    return voxelised_stack


def dffb(stack, percentile=10, offset=0.1):
    """takes in a t (x y) stack"""

    baseline = np.percentile(stack, percentile, axis=0)
    return (stack - baseline) / (baseline + offset)
    # return (baseline)


def split_stack_merge(stack, planes=30):
    x = (stack.shape[0]) / 30
    list_of_stacks = np.array_split(stack, x, axis=0)
    index = 0
    list_of_projection_stacks = []
    for i in list_of_stacks:
        list_of_projection_stacks.append(np.amax(i, axis=0))
        index += 1
    max_stack = np.stack(list_of_projection_stacks, axis=0)
    return max_stack


def stackplanes(planes):
    ds = [plane.shape[0] for plane in planes]
    # should clip to smallest plane?
    stack = np.concatenate([plane[:min(ds), ..., None] for plane in planes], axis=3)
    return stack


def make_montage(input_dir):
    filepath = os.path.join(input_dir, 'RECOMBINED')
    filelist = natsorted(os.listdir(filepath))
    outdir = os.path.join(input_dir, 'MONTAGES')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print 'Making Montage'
    for f in filelist:
        current_file = os.path.join(filepath, f)
        print 'Processing ', current_file
        print 'Reading ...'
        stack = io.imread(current_file, plugin="tifffile")
        num_planes = 30
        # reshape the stack
        planes = [stack[i::num_planes, ...] for i in range(num_planes)]

        stack = stackplanes(planes)
        del planes

        ### Grid planes together
        nrows = 5
        ncols = 6

        rowindicies = np.array_split(np.arange(num_planes), nrows)

        gridstack = np.hstack([np.dstack([stack[..., r] for r in ri]) for ri in rowindicies])
        print 'Saving ...'
        io.imsave(os.path.join(outdir, '{0}_gridstack.tiff'.format(os.path.splitext(f)[0])),
                  gridstack.astype('float32'))
        print 'Done with this file.'
    print 'Montage finished!'
