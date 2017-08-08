import os
import numpy as np
import skimage as sk
from skimage import io
from natsort import natsorted


big_folder = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis\recombine part 2'
new_path = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis'

def recombination2(big_folder, input_file):
    print 'RECOMBINING STACKS, OUTPUT FROM ', input_file
    right_folder = os.path.join(big_folder, input_file)
    filelist = natsorted(os.listdir(right_folder))
    outdir = os.path.join(big_folder, 'RECOMBINED')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for sub_folder in filelist:
        print 'Working on ', sub_folder
        folder_containing_single_planes = os.path.join(right_folder, sub_folder)
        plane_list = natsorted(os.listdir(folder_containing_single_planes))
        list_of_planes = []
        for i in plane_list:
            ipath = os.path.join(folder_containing_single_planes, i)
            plane = io.imread(ipath, plugin='tifffile')
            list_of_planes.append(plane)
        new_list = []
        for plane in np.arange(list_of_planes[0].shape[0]):
            for array in list_of_planes:
                new_list.append(array[plane])
        recombined = np.stack(new_list, axis=0)
        basepath, final_folder = os.path.split(folder_containing_single_planes)
        io.imsave(os.path.join(outdir, (final_folder + '_' + str.lower(input_file) + '_recombined.tiff')), recombined.astype('float32'))
    print 'FINISHED: RECOMBINING PLANES OF INPUT ', input_file


# recombination(new_path, 'EQUALISED')
recombination2(big_folder, 'CORRECTED')

corrected_recombined_path = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis\RECOMBINED'
filelist = natsorted(os.listdir(corrected_recombined_path))
output = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis\big corrected stacks'

filelist = natsorted(os.listdir(corrected_recombined_path))
for subfolder in filelist[1:]:
    print 'Working on', subfolder
    partlist = natsorted(os.listdir(os.path.join(corrected_recombined_path, subfolder)))
    stack1 = io.imread(os.path.join(corrected_recombined_path, subfolder, partlist[0]), plugin='tifffile')
    stack2 = io.imread(os.path.join(corrected_recombined_path, subfolder, partlist[1]), plugin='tifffile')
    combo = np.vstack((stack1, stack2))
    saveloc = os.path.join(output, subfolder + '.tiff')
    print 'Saving', subfolder
    io.imsave(saveloc, combo.astype('float32'))
    print 'saved: ', subfolder
