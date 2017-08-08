import os
import numpy as np
from sklearn.decomposition import PCA, FastICA
from skimage import io
from skimage import transform as tf
from natsort import natsorted
import scipy.ndimage as nd
from functions import *
from DRT_functions import *

frames = 2400
rolling_duration = 15
voxel_size = (1, 4, 4)
big_out = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis\20min DRT output'

het_path1 = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis\CORRECTED\21-06-17_fish2_Het_H2B_6dpf_2bps_60fps_6ms_4x4_5,6V_20min_trial1_00878_part1'
het_path2 = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis\CORRECTED\21-06-17_fish2_Het_H2B_6dpf_2bps_60fps_6ms_4x4_5,6V_20min_trial1_00878_part2'
gr_path1 = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis\CORRECTED\20-06-17_fish3_GR_H2B_6dpf_2bps_60fps_6ms_4x4_5,6V_20min_trial1_00864_part1'
gr_path2 = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis\CORRECTED\20-06-17_fish3_GR_H2B_6dpf_2bps_60fps_6ms_4x4_5,6V_20min_trial1_00864_part2'


def run_all_DRTS(genotype, corrected_path=None, part1_path=None, part2_path=None, big_out=big_out, frames=2400,
                 voxel_size=(1, 4, 4), rolling_duration=15):
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


run_all_DRTS(genotype='het', part1_path=het_path1, part2_path=het_path2)
run_all_DRTS(genotype='gr', part1_path=gr_path1, part2_path=gr_path2)

# ''' This section is for old stacks with fluoxetine '''
# path_folder = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\CORRECTED'
# folder_list = natsorted(os.listdir(path_folder))
# gr_dmso1 = os.path.join(path_folder, folder_list[0])
# gr_fluox1 = os.path.join(path_folder, folder_list[2])
# het_dmso1 = os.path.join(path_folder, folder_list[4])
# het_fluox1 = os.path.join(path_folder, folder_list[6])
# gr_dmso2 = os.path.join(path_folder, folder_list[1])
# gr_fluox2 = os.path.join(path_folder, folder_list[3])
# het_dmso2 = os.path.join(path_folder, folder_list[5])
# het_fluox2 = os.path.join(path_folder, folder_list[7])
# run_all_DRTS(genotype='gr_dmso', part1_path=gr_dmso1, part2_path=gr_dmso2)
# run_all_DRTS(genotype='gr_fluox', part1_path=gr_fluox1, part2_path=gr_fluox2)
# run_all_DRTS(genotype='het_dmso', part1_path=het_dmso1, part2_path=het_dmso2)
# run_all_DRTS(genotype='het_fluox', part1_path=het_fluox1, part2_path=het_fluox2)
