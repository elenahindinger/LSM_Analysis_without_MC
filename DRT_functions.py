import os
import numpy as np
from sklearn.decomposition import PCA, FastICA, NMF
from skimage import io
from skimage import transform as tf
import matplotlib.ticker as ticker
import seaborn as sns
import scipy.ndimage as nd
from sklearn.manifold import TSNE, Isomap
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from natsort import natsorted
from functions import *

def saver(fig, savename):
    formats = ['tiff', 'svg', 'pdf']
    for fileformat in formats:
        temp_savename = os.path.join(savename + '.%s' % fileformat)
        fig.savefig(temp_savename, format=fileformat, bbox_inches='tight', dpi=300)


def merge_filter_and_voxelise(part1_path, part2_path, rolling_duration=15, voxel_size=(1, 4, 4)):
    part1_filelist = natsorted(os.listdir(part1_path))
    part2_filelist = natsorted(os.listdir(part2_path))
    filtered_plane_list = []
    print 'Reading planes.'
    counter = 0
    for plane in part1_filelist:
        if plane.endswith('tiff'):
            plane_path = os.path.join(part1_path, plane)
            print 'part1: ', plane_path
            part2_plane_path = os.path.join(part2_path, part2_filelist[counter])
            print 'part2: ', part2_plane_path
            part1 = io.imread(plane_path, plugin='tifffile')
            part2 = io.imread(part2_plane_path, plugin='tifffile')
            combo = np.vstack((part1, part2))
            filtered_plane_list.append(nd.uniform_filter(combo, (1, 1, rolling_duration)))
            counter += 1
        else:
            pass
    new_list = []
    print 'Filtering planes.'
    for plane in np.arange(filtered_plane_list[0].shape[0]):
        for array in filtered_plane_list:
            new_list.append(array[plane])
    mean_filtered_stack = np.stack(new_list, axis=0)
    print 'Voxelising stack'
    voxelised_stack = tf.downscale_local_mean(mean_filtered_stack, voxel_size)
    return voxelised_stack


def filter_and_voxelise(corrected_path, rolling_duration=15, voxel_size=(1, 4, 4)):
    corrected_files = natsorted(os.listdir(corrected_path))
    filtered_plane_list = []
    print 'Reading planes.'
    for plane in corrected_files:
        if plane.endswith('tiff'):
            plane_path = os.path.join(corrected_path, plane)
            stack = io.imread(plane_path, plugin='tifffile')
            filtered_plane_list.append(nd.uniform_filter(stack, (1, 1, rolling_duration)))
        else:
            pass
    new_list = []
    print 'Filtering planes.'
    for plane in np.arange(filtered_plane_list[0].shape[0]):
        for array in filtered_plane_list:
            new_list.append(array[plane])
    mean_filtered_stack = np.stack(new_list, axis=0)
    print 'Voxelising stack'
    voxelised_stack = tf.downscale_local_mean(mean_filtered_stack, voxel_size)
    return voxelised_stack


def pca_prep(voxelised_input):
    print 'Preprocessing for time linearised series.'
    length = voxelised_input.shape[0]
    time_linearised_list = []
    for start in np.arange(0, length, 30):
        time_linearised_list.append(voxelised_input[start:start+30, ...].flatten())
    time_linearised = np.stack(time_linearised_list, axis=0)
    mean, std = time_linearised.mean(axis=0), time_linearised.std(axis=0)
    clean_data = (time_linearised - mean) / std
    return clean_data


def pca_explained_variance_plot(pca, outdir, genotype, frames):
    print 'Calculating explained variance'
    sns.set_style('white')
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    plt.margins(0.05)
    axes.plot(np.cumsum(pca.explained_variance_ratio_[:20]), linewidth=2, color='b', marker='s')
    axes.set_ylabel('Cumulative Sum of Explained Variance', fontsize=16)
    plt.suptitle('Cumulative Sum of Explained Variance per Principal Component', fontsize=18)
    plt.axis('tight')
    axes.set_xlabel('Number of Components', fontsize=16)
    plt.xticks(np.arange(0, 20, 2), np.arange(1, 21, 2))
    axes.tick_params(labelsize=12)
    savename = os.path.join(outdir, genotype + '_PCA_explained_variance')
    saver(fig, savename)
    plt.close('all')

    sns.set_style('white')
    fig1, axes = plt.subplots(1, 1, figsize=(10, 5))
    plt.margins(0.05)
    axes.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=2, color='b', marker='s')
    axes.set_ylabel('Cumulative Sum of Explained Variance', fontsize=16)
    plt.suptitle('Cumulative Sum of Explained Variance per Principal Component', fontsize=18)
    plt.axis('tight')
    axes.set_xlabel('Number of Components', fontsize=16)
    plt.xticks(np.arange(0, frames, 200), np.arange(1, (frames+1), 200))
    axes.tick_params(labelsize=12)
    savename = os.path.join(outdir, genotype + '_PCA_explained_variance_ALL')
    saver(fig1, savename)
    plt.close('all')


def pca_brain_location_plot(pca, outdir, voxel_dimensions, genotype, number_of_components):
    print 'Mapping back to brain.'
    pc_list = []
    for i in np.arange(number_of_components):
        pc_temp = pca.components_[i].reshape((30, voxel_dimensions[1], voxel_dimensions[2]))
        pc_list.append(pc_temp)
    fig, axes = plt.subplots(number_of_components, 1, figsize=(20, 35))
    # plt.suptitle('Anatomical Location of Principal Components', fontsize=36)
    for ax in np.arange(number_of_components):
        axes[ax].imshow(pc_list[ax][15], cmap='RdBu_r')
        axes[ax].set_ylabel('PC%s' % str(ax + 1), fontsize=18)
        axes[ax].tick_params(axis='both', which='both', bottom='off', top='off', right='off', left='off',
                             labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    fig.tight_layout()
    savename = os.path.join(outdir, genotype + '_PCA_anatomical_location_%sPCs' % number_of_components)
    saver(fig, savename)
    plt.close('all')


def brain_location_3d(pca, outdir, voxelised_stack, genotype, number_of_components, method):
    voxel_dimensions = voxelised_stack.shape
    pc_list = []
    component_out = os.path.join(outdir, 'single plane mapped components')
    if not os.path.exists(component_out):
        os.makedirs(component_out)
    for i in np.arange(number_of_components):
        print i
        pc_temp = pca.components_[i].reshape((30, voxel_dimensions[1], voxel_dimensions[2]))
        pc_list.append(pc_temp)
    for ax in np.arange(number_of_components):
        for plane in np.arange(30):
            plt.imsave(os.path.join(component_out, '%s_%s_anatomical_location_volume_PC%s_plane%s.tiff' % (genotype, method, str(ax+1), str(plane+1))),
                       pc_list[ax][plane], cmap='RdBu_r', format='tiff')


def activity_profile(pca, X, number_of_components, outdir, genotype, frames):
    print 'Plotting activity profile.'
    minutes = int(frames / 60 / 2)
    sns.set_style('white')
    transformed = pca.transform(X)
    name = 'PCA'
    fig, axes = plt.subplots(number_of_components, 1, figsize=(30, 20), sharey=True)
    plt.suptitle('Coefficients over Time', fontsize=52)
    # plt.suptitle('Coefficients of %s components over time' % method, fontsize=36)
    colors = ['orange', 'r', 'pink', 'magenta', 'purple', 'navy', 'blue', 'lightblue', 'cyan', 'g', 'gold', 'brown']
    nco = np.arange(number_of_components)
    for pc in nco:
        pc_temp = transformed[:, pc]
        sns.despine(ax=axes[pc], left=True, bottom=True)
        axes[pc].plot(pc_temp, linewidth=3, color=colors[pc])
        axes[pc].set_ylabel('PC %s' % (str(pc + 1)), fontsize=32, rotation='horizontal', labelpad=50)
        axes[pc].set_xticks(np.arange(0))
        axes[pc].set_yticks(np.arange(0))
    plt.xticks(np.arange(0, (frames+1), 120), np.arange(minutes+1))
    axes[nco[-1]].tick_params(labelsize=32)
    axes[nco[-1]].set_xlabel('Time (min)', fontsize=42, labelpad=30)
    savename = os.path.join(outdir, '%s_%s_%s_components_coefficients_over_time' % (genotype, name, str(number_of_components)))
    saver(fig, savename)
    plt.close('all')


def pca_plots(pre_processed, outdir, voxelised_stack, genotype, number_of_components, frames):
    ''' Take output from pca_prep as input here. '''
    print 'Running PCA.'
    new_path = os.path.join(outdir, 'PCA')
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    X = pre_processed.copy()
    pca = PCA()
    pca.fit(X)

    # pca_explained_variance_plot(pca=pca, outdir=new_path, type='ratio', genotype=genotype)
    pca_explained_variance_plot(pca=pca, outdir=new_path, genotype=genotype, frames=frames)

    pca_brain_location_plot(pca=pca, outdir=new_path, voxel_dimensions=voxelised_stack.shape, genotype=genotype, number_of_components=number_of_components)

    brain_location_3d(pca=pca, outdir=new_path, voxelised_stack=voxelised_stack, genotype=genotype, number_of_components=number_of_components, method='PCA')

    activity_profile(pca=pca, X=X, number_of_components=number_of_components, outdir=new_path, genotype=genotype, frames=frames)

    print 'PCA completed.'


def activity_profile_ica(transformed, num, outdir, genotype, frames):
    sns.set_style('white')
    minutes = int(frames / 60 / 2)
    fig1, axes = plt.subplots(num, 1, figsize=(20, 20), sharey=True)
    plt.suptitle('Coefficients over Time', fontsize=52)
    # plt.suptitle('Coefficients of %s components over time' % method, fontsize=36)
    colors = ['orange', 'r', 'pink', 'magenta', 'purple', 'navy', 'blue', 'lightblue', 'cyan', 'g', 'gold', 'brown']
    nco = np.arange(num)
    for pc in nco:
        sns.despine(ax=axes[pc], left=True, bottom=True)
        pc_temp = transformed[:, pc]
        axes[pc].plot(pc_temp, linewidth=3, color=colors[pc])
        axes[pc].set_ylabel('PC %s' % (str(pc + 1)), fontsize=32, rotation='horizontal', labelpad=50)
        axes[pc].set_xticks(np.arange(0))
        axes[pc].set_yticks(np.arange(0))
    plt.xticks(np.arange(0, (frames+1), 120), np.arange(minutes + 1))
    axes[nco[-1]].tick_params(labelsize=32)
    axes[nco[-1]].set_xlabel('Time (min)', fontsize=42, labelpad=30)
    savename = os.path.join(outdir, '%s_ICA_%s_components_coefficients_over_time' % (genotype, str(num)))
    saver(fig1, savename)
    plt.close('all')


def anatomical_location_ica(outdir, ica, voxel_dimensions, genotype, i):
    pc_list = []
    for j in np.arange(i):
        pc_temp = ica.components_[j].reshape((30, voxel_dimensions[1], voxel_dimensions[2]))
        pc_list.append(pc_temp)
    fig, axes = plt.subplots(i, 1, figsize=(20, 35))
    # plt.suptitle('Anatomical Location of Principal Components', fontsize=36)
    for ax in np.arange(i):
        axes[ax].imshow(pc_list[ax][15], cmap='RdBu_r')
        axes[ax].set_ylabel('PC%s' % str(ax + 1), fontsize=18)
        axes[ax].tick_params(axis='both', which='both', bottom='off', top='off', right='off', left='off',
                             labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    fig.tight_layout()
    savename = os.path.join(outdir, '%s_ICA_anatomical_location_%sPCs' % (genotype, str(i)))
    saver(fig, savename)
    plt.close('all')


def total_ICA(pre_processed, voxelised_stack, outdir, genotype, frames):
    print 'Running ICA.'
    voxel_dimensions = voxelised_stack.shape
    ica_output = os.path.join(outdir, 'ICA')
    if not os.path.exists(ica_output):
        os.makedirs(ica_output)
    for i in np.arange(3, 8):
        print 'Number of components: ', i
        numco_out = os.path.join(ica_output, 'ICA_on_%s_components' % i)
        if not os.path.exists(numco_out):
            os.makedirs(numco_out)
        ica = FastICA(n_components=i)
        transformed = ica.fit_transform(pre_processed)
        activity_profile_ica(transformed=transformed, num=i, outdir=numco_out, genotype=genotype, frames=frames)
        anatomical_location_ica(outdir=numco_out, ica=ica, voxel_dimensions=voxel_dimensions, genotype=genotype, i=i)
        brain_location_3d(pca=ica, outdir=numco_out, voxelised_stack=voxelised_stack, genotype=genotype,
                          number_of_components=i, method='ICA')

    print 'ICA done.'


def run_tsne(pre_processed, genotype, frames, outdir, numco=2):
    minutes = frames / 60 / 2
    print 'Running t-SNE.'
    tsne_out = os.path.join(outdir, 't-SNE')
    if not os.path.exists(tsne_out):
        os.makedirs(tsne_out)
    tsne = TSNE(n_components=numco, init='pca', random_state=0)
    transformed = tsne.fit_transform(pre_processed)

    fig, axes = plt.subplots(numco, 1, figsize=(20, 20), sharey=True)
    plt.suptitle('t-SNE Coefficients over Time', fontsize=36)
    # plt.suptitle('Coefficients of %s components over time' % method, fontsize=36)
    colors = ['orange', 'r', 'pink', 'magenta', 'purple', 'navy', 'blue', 'lightblue', 'cyan', 'g', 'gold', 'brown']
    nco = np.arange(numco)
    for pc in nco:
        pc_temp = transformed[:, pc]
        sns.despine(ax=axes[pc], left=True, bottom=True)
        axes[pc].plot(pc_temp, linewidth=3, color=colors[pc])
        axes[pc].set_ylabel('Component %s' % (str(pc + 1)), fontsize=24)
        axes[pc].set_xticks(np.arange(0))
        axes[pc].set_yticks(np.arange(0))
        axes[pc].xaxis.set_major_formatter(NullFormatter())
        axes[pc].yaxis.set_major_formatter(NullFormatter())
    plt.xticks(np.arange(0, (frames+1), 120), np.arange(minutes+1))
    axes[nco[-1]].tick_params(labelsize=18)
    axes[nco[-1]].set_xlabel('Time (min)', fontsize=24)
    fig.tight_layout()
    savename = os.path.join(tsne_out, genotype + '_t-SNE_anatomical_location_%sPCs' % numco)
    saver(fig, savename)
    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.set_style('white')
    plt.suptitle('2D projection by t-SNE', fontsize=36)
    ax.scatter(transformed[:, 0], transformed[:, 1])
    sns.despine(ax=ax, right=True, top=True)
    sns.despine(trim=True, offset=25)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.set_xlabel('Component 1', fontsize=24)
    ax.set_ylabel('Component 2', fontsize=24)
    savename = os.path.join(tsne_out, genotype + '_t-SNE_scatter_plot')
    saver(fig, savename)
    plt.close('all')


def run_isomap(pre_processed, genotype, outdir, frames, numco=5, numneb=30):
    minutes = frames / 60 / 2
    print 'Running IsoMap'
    iso = Isomap(n_neighbors=numneb, n_components=numco)
    transformed = iso.fit_transform(pre_processed)

    iso_out = os.path.join(outdir, 'IsoMap')
    if not os.path.exists(iso_out):
        os.makedirs(iso_out)

    fig, axes = plt.subplots(numco, 1, figsize=(20, 20), sharey=True)
    plt.suptitle('IsoMap Coefficients over Time', fontsize=36)
    # plt.suptitle('Coefficients of %s components over time' % method, fontsize=36)
    colors = ['orange', 'r', 'pink', 'magenta', 'purple', 'navy', 'blue', 'lightblue', 'cyan', 'g', 'gold', 'brown']
    nco = np.arange(numco)
    for pc in nco:
        pc_temp = transformed[:, pc]
        sns.despine(ax=axes[pc], left=True, bottom=True)
        axes[pc].plot(pc_temp, linewidth=3, color=colors[pc])
        axes[pc].set_ylabel('Component %s' % (str(pc + 1)), fontsize=24)
        axes[pc].set_xticks(np.arange(0))
        axes[pc].set_yticks(np.arange(0))
        axes[pc].xaxis.set_major_formatter(NullFormatter())
        axes[pc].yaxis.set_major_formatter(NullFormatter())
    plt.xticks(np.arange(0, (frames+1), 120), np.arange(minutes+1))
    axes[nco[-1]].tick_params(labelsize=18)
    axes[nco[-1]].set_xlabel('Time (min)', fontsize=24)
    savename = os.path.join(iso_out, genotype + '_IsoMap_anatomical_location_%sPCs' % numco)
    saver(fig, savename)
    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.set_style('white')
    plt.suptitle('2D projection by IsoMap', fontsize=36)
    ax.scatter(transformed[:, 0], transformed[:, 1])
    sns.despine(ax=ax, right=True, top=True)
    sns.despine(trim=True, offset=25)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.set_xlabel('Component 1', fontsize=24)
    ax.set_ylabel('Component 2', fontsize=24)
    savename = os.path.join(iso_out, genotype + '_IsoMap_scatter_plot')
    saver(fig, savename)
    plt.close('all')

    # # 3D projection of IsoMap
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.suptitle('3D projection by IsoMap', fontsize=36)
    #
    # ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2])
    # ax.set_xlabel('Component 1', fontsize=24)
    # ax.set_ylabel('Component 2', fontsize=24)
    # ax.set_zlabel('Component 3', fontsize=24)
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    # ax.zaxis.set_major_formatter(NullFormatter())
    # print 'Please take a screenshot and continue.'
    # plt.show()
    print 'Finished IsoMap'


def brain_location_3d_nmf(new_comp, outdir, genotype, number_of_components):
    component_out = os.path.join(outdir, 'single plane mapped components')
    if not os.path.exists(component_out):
        os.makedirs(component_out)
    for ax in np.arange(number_of_components):
        for plane in np.arange(30):
            plt.imsave(os.path.join(component_out, '%s_NMF_anatomical_location_volume_comp%s_plane%s.tiff' % (genotype, str(ax+1), str(plane+1))),
                       new_comp[ax][plane], cmap='RdBu_r', format='tiff')


def run_NMF(pre_processed, voxelised_stack, genotype, outdir, frames, numco=20):
    print 'Running NMF.'
    minutes = frames / 60 / 2
    nmf_out = os.path.join(outdir, 'NMF')
    if not os.path.exists(nmf_out):
        os.makedirs(nmf_out)
    voxel_dimensions = voxelised_stack.shape
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(pre_processed)
    nnmf = NMF(n_components=numco, init=None, solver='cd', tol=0.0001, max_iter=200, random_state=None, alpha=0.0,
               l1_ratio=0.0, verbose=0, shuffle=False, nls_max_iter=2000, sparseness=None, beta=1, eta=0.1)
    W = nnmf.fit_transform(scaled)
    comp = nnmf.components_
    new_comp = np.reshape(comp, (numco, 30, voxel_dimensions[1], voxel_dimensions[2]))
    for i in np.arange(numco):
        fig = plt.figure()
        sns.set_style('white')
        plt.subplot(2, 1, 1), plt.imshow(np.amax(new_comp[i], axis=0), cmap='Greens'), plt.colorbar(pad=0.1)
        plt.xticks(np.arange(0))
        plt.yticks(np.arange(0))
        sns.despine(left=True, bottom=True)
        plt.subplot(2, 1, 2), plt.margins(0.05), plt.plot(W[:, i], color='g')
        plt.xlabel('Time (min)', fontsize=18)
        plt.ylabel('Component %s' % (str(i + 1)), fontsize=18)
        plt.yticks(np.arange(0))
        plt.xticks(np.arange(0, (frames+1), 120), np.arange(minutes+1))
        plt.tick_params(labelsize=14)
        savename = os.path.join(nmf_out, '%s_NMF_component_%s' % (genotype, str(i+1)))
        saver(fig, savename)
        plt.close('all')

    sns.set_style('white')
    fig1, axes = plt.subplots(numco, 1, figsize=(20, 20), sharey=True)
    plt.suptitle('Coefficients over Time', fontsize=36)
    # plt.suptitle('Coefficients of %s components over time' % method, fontsize=36)
    colors = ['orange', 'r', 'pink', 'magenta', 'purple', 'navy', 'blue', 'lightblue', 'cyan', 'g', 'gold', 'brown',
              'orange', 'r', 'pink', 'magenta', 'purple', 'navy', 'blue', 'lightblue', 'cyan', 'g', 'gold', 'brown']
    nco = np.arange(numco)
    for i in np.arange(numco):
        sns.despine(ax=axes[i], left=True, bottom=True)
        axes[i].plot(W[:, i], linewidth=2, color=colors[i])
        axes[i].set_ylabel(str(i + 1), fontsize=24)
        axes[i].set_xticks(np.arange(0))
        axes[i].set_yticks(np.arange(0))
    plt.xticks(np.arange(0, (frames+1), 120), np.arange(minutes+1))
    axes[nco[-1]].tick_params(labelsize=18)
    axes[nco[-1]].set_xlabel('Time (min)', fontsize=24)
    savename = os.path.join(nmf_out, '%s_NMF_%s_components_coefficients_over_time' % (genotype, str(numco)))
    saver(fig1, savename)
    plt.close('all')
    brain_location_3d_nmf(new_comp=new_comp, outdir=nmf_out, genotype=genotype, number_of_components=numco)

    print 'Finished NMF.'


# test = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis\CORRECTED\21-06-17_fish2_Het_H2B_6dpf_2bps_60fps_6ms_4x4_5,6V_20min_trial1_00878_part1\21-06-17_fish2_Het_H2B_6dpf_2bps_60fps_6ms_4x4_5,6V_20min_trial1_00878_part1_plane_15_corrected.tiff'
# plane = io.imread(test, plugin='tifffile')
# datacro = nd.uniform_filter(plane, (1, 1, 15))
# # datacro = tf.downscale_local_mean(filtered, (1, 4, 4))
# frames= len(datacro[:,0,0])
# rows= len(datacro[0,:,0])
# columns= len(datacro[0,0,:])
# components = 20
# pre_processed=np.reshape(datacro,(frames,rows*columns))
# scaler = MinMaxScaler()
# scaled = scaler.fit_transform(pre_processed)
# #rT=datacro
# model = NMF(n_components=components, init=None, solver='cd', tol=0.0001, max_iter=200, random_state=None, alpha=0.0,
#             l1_ratio=0.0, verbose=0, shuffle=False, nls_max_iter=2000, sparseness=None, beta=1, eta=0.1)
# W=model.fit_transform(scaled)
# comp=model.components_
# comp=np.reshape(comp,(components, rows,columns))
# for i in range(components):
#     fig3=plt.figure()
#     plt.subplot(2,1,1),plt.imshow(comp[i,:,:],cmap='Greens'),plt.colorbar()
#     plt.subplot(2,1,2),plt.plot(W[:,i])
# error=model.reconstruction_err_
# print error
