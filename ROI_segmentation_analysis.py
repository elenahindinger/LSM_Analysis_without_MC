import os
import pandas as pd
import numpy as np
from natsort import natsorted
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import sparse

__author__ = 'Elena Maria Daniela Hindinger'


def saver(fig, savename):
    formats = ['tiff', 'svg', 'pdf']
    for fileformat in formats:
        temp_savename = os.path.join(savename + '.%s' % fileformat)
        fig.savefig(temp_savename, format=fileformat, bbox_inches='tight', dpi=300)


def fixed_baseline(y, lam=100000000, p=0.005, niter=10):
    """Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm
    Y is one trace as a np array
    (P. Eilers, H. Boelens 2005)
    CODE FROM http://stackoverflow.com/questions/29156532/python-baseline-correction-library - Paper:ALS_Linda
    """
    if np.shape(y)[0] > np.shape(y)[1]:
        y = y.T
    elif len(np.shape(y)) > 1:
        y = y[0]
    else:
        pass
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return y-z, z


data_sheet = r'I:\Elena H\LIGHT SHEET MICROSCOPE\Elena Light Sheet Analysis\LAST DAY ANALYSIS\ROI Segmentation\het whole\Results_het_whole_new.csv'
outdir = r'I:\Elena H\LIGHT SHEET MICROSCOPE\Elena Light Sheet Analysis\LAST DAY ANALYSIS\ROI Segmentation\flattened baseline'
genotype = 'het'
frames = 2400

print 'Reading input data for ', data_sheet
print 'Genotype: ', genotype
df = pd.read_csv(data_sheet)
# df = df.drop(df.columns[[0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28]], axis=1)
df = df.drop(df.columns[[0]], axis=1)
dff_df = pd.DataFrame()
for name, temp in df.iteritems():
    column = fixed_baseline(np.array([temp.values]))[0]
    baseline = np.percentile(column, 10, axis=0)
    dff = (column - baseline) / (baseline+0.1)
    as_df = pd.DataFrame(dff)
    as_df.columns = [name]
    dff_df = pd.concat([dff_df, as_df], axis=1)

mean_df = pd.concat([dff_df[dff_df.columns[i]] for i in np.arange(0, 21, 2)], axis=1)
median_df = pd.concat([dff_df[dff_df.columns[i]] for i in np.arange(1, 22, 2)], axis=1)


def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)


def dff10_traces(df, mode, genotype=genotype, outdir=outdir, frames=frames):
    print 'Drawing traces for ', mode
    minutes = frames / 60 / 2
    sns.set_style('white')
    fig, axes = plt.subplots(11, 1, sharey=True, figsize=(20, 20))
    adjustFigAspect(fig, aspect=0.8)
    plt.suptitle('Average fluorescence traces over time', fontsize=52)
    sns.despine(bottom=True, right=True, top=True)
    colors = ['b', 'g', 'gold', 'orange', 'r', 'darkred', 'magenta', 'blueviolet', 'navy', 'teal', 'cyan']
    ylabel = ['L pallium', 'R pallium', 'L habenula', 'R habenula', 'L midbrain', 'R midbrain', 'V midbrain',
              'L cerebellum', 'R cerebellum', 'L hindbrain', 'R hindbrain']
    # ylabel = np.arange(1, 12)
    plt.subplots_adjust(hspace=-0.02)
    for ax in np.arange(11):
        axes[ax].plot(np.arange(frames), df[df.columns[ax]], color=colors[ax])
        axes[ax].set_ylabel(ylabel[ax], fontsize=32, rotation='horizontal', labelpad=110)
        axes[ax].set_xticks(np.arange(0))
        axes[ax].set_yticks(np.arange(0))
    plt.xticks(np.arange(0, (frames + 1), 120), np.arange(minutes + 1))
    plt.tick_params(labelsize=32)
    axes[10].set_xlabel('Time (min)', fontsize=42, labelpad=50)
    savename = os.path.join(outdir, '%s_%s_dff_fluorescence_traces' % (genotype, mode))
    saver(fig, savename)
    plt.close('all')

dff10_traces(mean_df, mode='mean')
dff10_traces(median_df, mode='median')

def correlation_matrix(df, outdir, genotype, mode):
    print 'Calculating correlation matrix for ', mode
    array = df.values.T
    cm = np.corrcoef(array)
    ylabel = ['L pallium', 'R pallium', 'L habenula', 'R habenula', 'L midbrain', 'R midbrain', 'V midbrain',
              'L cerebellum', 'R cerebellum', 'L hindbrain', 'R hindbrain']
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(11), ylabel, fontsize=18, rotation='vertical')
    plt.yticks(np.arange(11), ylabel, fontsize=18)
    plt.suptitle('ROI Correlation Heat Map', fontsize=24)
    savename = os.path.join(outdir, '%s_%s_correlation_matrix' % (genotype, mode))
    saver(fig, savename)
    plt.close('all')

def cluster_map(df, outdir, genotype, mode):
    print 'Calculating cluster map for ', mode
    # df.columns = ['L pallium', 'R pallium', 'L habenula', 'R habenula', 'L midbrain', 'R midbrain', 'V midbrain',
    #               'L cerebellum', 'R cerebellum', 'L hindbrain', 'R hindbrain']
    df.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    colors = ['b', 'g', 'gold', 'orange', 'r', 'darkred', 'magenta', 'blueviolet', 'navy', 'teal', 'cyan']
    network_lut = dict(zip(map(str, df.columns), colors))
    networks = df.columns
    network_colors = pd.Series(networks, index=df.columns).map(network_lut)
    sns.set()
    sns.set_style('white')

    cg = sns.clustermap(df.corr(), cmap="BrBG",
                        row_colors=network_colors, col_colors=network_colors, figsize=(20, 20), vmin=-1, vmax=1)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    # plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(cg.ax_heatmap.tick_params(labelsize=48))
    savename = os.path.join(outdir, '%s_%s_cluster_map' % (genotype, mode))
    formats = ['tiff', 'svg', 'pdf']
    for fileformat in formats:
        temp_savename = os.path.join(savename + '.%s' % fileformat)
        plt.savefig(temp_savename, format=fileformat, bbox_inches='tight', dpi=300)
    plt.close('all')


cluster_map(df=mean_df, outdir=outdir, genotype=genotype, mode='mean')
cluster_map(df=median_df, outdir=outdir, genotype=genotype, mode='median')

print 'Finished with all!'