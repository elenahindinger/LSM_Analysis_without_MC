from __future__ import division
import os
import numpy as np
from skimage import io

folder = r'K:\lsm analysis\trial input\big stack\input'
filelist = os.listdir(folder)
new_path = r'K:\lsm analysis\big stack\output'


for file in filelist:
    print 'Processing ' + file
    filepath = os.path.join(folder, file)
    stack = io.imread(filepath, plugin="tifffile")
    outdir = os.path.join(new_path, file[:18])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # use this in future
    for j in np.arange(30):
        print 'Processing plane %s' % str(j)
        list_of_plane = []
        for i in np.arange(j, 3600, 30):
            list_of_plane.append(stack[i, ...])
        plane1 = np.stack(list_of_plane, axis=0).astype('float32')
        io.imsave(os.path.join(outdir, 'plane_%s.tiff' % str(j+1)), plane1)