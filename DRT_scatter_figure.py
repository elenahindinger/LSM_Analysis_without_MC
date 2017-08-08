import numpy as np
from skimage import io
import os
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn import random_projection
from skimage import transform as tf
from natsort import natsorted
import scipy.ndimage as nd
from functions import *
from DRT_functions import *

corrected_path = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis\CORRECTED\21-06-17_fish2_Het_H2B_6dpf_2bps_60fps_6ms_4x4_5,6V_20min_trial1_00878_part1.tif'
new_path = r'I:\Elena H\BIG PC LIGHT SHEET PRE-DR\final analysis'

def mean_filter(corrected_path, rolling_duration=10):
    corrected_files = natsorted(os.listdir(corrected_path))
    filtered_plane_list = []
    print 'Reading planes.'
    for plane in corrected_files:
        plane_path = os.path.join(corrected_path, plane)
        stack = io.imread(plane_path, plugin='tifffile')
        filtered_plane_list.append(nd.uniform_filter(stack, (1, 1, rolling_duration)))
    new_list = []
    print 'Filtering planes.'
    for plane in np.arange(filtered_plane_list[0].shape[0]):
        for array in filtered_plane_list:
            new_list.append(array[plane])
    recombined = np.stack(new_list, axis=0)
    return recombined

def filter_and_voxelise(corrected_path):
    mean_filtered_stack = mean_filter(corrected_path)
    print 'Voxelising stack'
    voxelised_stack = tf.downscale_local_mean(mean_filtered_stack, (1, 4, 4))
    return voxelised_stack


voxelised_stack = filter_and_voxelise(corrected_path)
pre_processed = pca_prep(voxelised_stack)

# random section
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
random_projection = rp.fit_transform(pre_processed)
plt.scatter(random_projection[:,0], random_projection[:,1])

# pca
pca = PCA(n_components=2)
pca_projection = pca.fit_transform(pre_processed)
plt.scatter(pca_projection[:,0], pca_projection[:,1])

# ica
ica = FastICA(n_components=2)
ica_projection = ica.fit_transform(pre_processed)
plt.scatter(ica_projection[:,0], ica_projection[:,1])

# tsne
tsne = TSNE(n_components=2)
tsne_projection = tsne.fit_transform(pre_processed)
plt.scatter(tsne_projection[:,0], tsne_projection[:,1])

# isomap
iso = Isomap(n_components=2)
iso_projection = iso.fit_transform(pre_processed)
plt.scatter(iso_projection[:,0],iso_projection[:,1])

# NMF
iso = NMF(n_components=2)
iso_projection = iso.fit_transform(pre_processed)
plt.scatter(iso_projection[:,0],iso_projection[:,1])
