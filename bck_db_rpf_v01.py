import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from  skimage.io import imread , imsave

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from RobustGaussianFittingLibrary import MSSE, fitValue
from RobustGaussianFittingLibrary.misc import scatter3
from RobustGaussianFittingLibrary.useMultiproc import fitBackgroundTensor_multiproc as fitback
from scipy.spatial import distance

import RobustPeakFinder as RPF
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import hdbscan

from sklearn.cluster import KMeans
from tifffile import imwrite


img = imread('/home/ali/Projects/NEW_RPF/21.08.18 Mouse 1 G-CSFR Imiq Ear.tif')
Z_DIST_um = 4
R_DIST_um = 0.85
C_DIST_um = 0.85
PEAK_MAX_PIX = 5000
PEAK_MIN_PIX = 50
PATCH_SIZE = 40
min_snr = 3.5
max_allowed_dist = 8

output_dir = 'output/'
if(True):
    winx_cent = 673;
    winy_cent = 1666;
    win_side = 400;
    IMG = img[10:38, 
              winx_cent - win_side:winx_cent + win_side,
              winy_cent - win_side:winy_cent + win_side]

vmin = IMG.min()
vmax = IMG.max()
if(0):
    for idx, an_image in enumerate(IMG):
        plt.imshow(an_image[70-30:70+30, 
                            300-30:300+30],cmap='gray', vmin = 0, vmax = 255)
        plt.title(idx)
        plt.draw()
        plt.pause(2)
        plt.clf()

if(True):
    inMask = np.ones(IMG.shape, dtype = 'uint8')
    inMask[IMG<=0] = 0
    background_model = fitback(
        inDataSet = IMG, 
        inMask = inMask, 
        winX = PATCH_SIZE, winY = PATCH_SIZE,
        topKthPerc = 0.5,
        bottomKthPerc = 0.3,
        MSSE_LAMBDA = 3.0,
        stretch2CornersOpt = 0,
        numModelParams = 4,
        optIters = 12,
        showProgress = True,
        numStrides = 4,
        minimumResidual = 0.01,
        numProcesses = None)
    np.savez('output/bck.npz',background_model = background_model)
else:
    background_model = np.load('output/bck.npz')['background_model']

peakMap = (IMG - background_model[0])/background_model[1]
peakMap[peakMap<=min_snr] = 0
peakMap[peakMap>min_snr] = 1
peakMap[background_model[1]==0] = 0

if(0):
    plt.imshow(peakMap.sum(0), cmap='hsv')
    plt.show()

peakMap = peakMap*IMG
if(0):
    for idx, an_image in enumerate(peakMap):
        plt.imshow(an_image,cmap='gray', vmin = 0, vmax = 1)
        plt.title(idx)
        plt.draw()
        plt.pause(2)
        plt.clf()
        
if(True):
    print('getting data')
    inds_i, inds_j, inds_k = np.where(peakMap>0)
    data = np.array([inds_i, inds_j, inds_k, inds_k])
    data[0, :] = data[0, :]*Z_DIST_um;
    data[1, :] = data[1, :]*R_DIST_um;
    data[2, :] = data[2, :]*C_DIST_um;
    data[3, :] = peakMap[inds_i, inds_j, inds_k]/4;
    
    del inds_i
    del inds_j
    del inds_k
    print('goign to DBSCAN')
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=PEAK_MIN_PIX)
    cluster_labels = clusterer.fit_predict(data.T)
    
    np.save('output/cluster_labels.npy', cluster_labels)
else:
    cluster_labels = np.load('output/cluster_labels.npy')
    
print('finished DBSCAN')

peakMap[peakMap>0] = cluster_labels
peakMap = peakMap.astype('int')

if(0):
    plt.imshow(peakMap[15], cmap='hsv')
    plt.show()

if(0):
    plt.imshow(peakMap.sum(0), cmap='hsv')
    plt.show()

if(0):
    #peakMap = peakMap*IMG
    for idx, an_image in enumerate(peakMap):
        plt.imshow(an_image)
        plt.title(idx)
        plt.draw()
        plt.pause(1)
        plt.clf()
'''
if(False):
    peakMap[peakMap>0] = 1
    if(True):
        #inMask = np.ones(IMG.shape, dtype = 'uint8')
        #inMask[IMG<=0] = 0
        _, _, peakMap = RPF.robustPeakFinderPyFunc_multiproc(
            inData = np.array([IMG]), 
            inMask = None,
            peakMask = peakMap,
            minBackMeanMap = 0, 
            singlePhotonADU = 1,
            maxBackMeanMap = None, 
            bckSNR = 0,
            pixPAPR = 0,
            PTCHSZ = PATCH_SIZE,
            PTCHSZz = 12,
            PEAK_MAX_PIX = 2*PEAK_MAX_PIX,
            PEAK_MIN_PIX = PEAK_MIN_PIX,
            MAXIMUM_NUMBER_OF_PEAKS = 1024,
            optIters = 1,
            finiteSampleBias = 100,
            downSampledSize = 100,
            MSSE_LAMBDA = 2.0,
            searchSNR = None,
            highPoissonTh = 0,
            lowPoissonTh = 0,
            multiproc_stride = None,
            returnPeakMap = True)
        peakMap = peakMap[0]
        print('n_peaks: ', peakMap.max())
        peakMap_file_name = output_dir + 'RPF_peakMap' + '.npz'
        print(peakMap_file_name)
        np.savez(peakMap_file_name, peakMap = peakMap)
        print('peakMap is calculated!')
    else:
        peakMap_file_name = output_dir + 'RPF_peakMap' + '.npz'
        peakMap = np.load(peakMap_file_name)['peakMap']
    peakMap = peakMap.astype('int')
'''
        
if(True):
    for label_cnt in range(1, peakMap.max() + 1):
        print(label_cnt, end = '')
        inds_i, inds_j, inds_k = np.where(peakMap == label_cnt)
        n_pts = inds_i.shape[0]
        data = np.array([inds_i, inds_j, inds_k]).astype('float64')
        data[0, :] = data[0, :]*Z_DIST_um;
        data[1, :] = data[1, :]*R_DIST_um;
        data[2, :] = data[2, :]*C_DIST_um;
        
        dist2 = distance.cdist(data.T, data.T, 'euclidean')
        dist2_stat = np.median(dist2, axis=1)
        cent_ind = np.argmin(dist2_stat)
        blob_cent = data[:, cent_ind]
        dist2_stat = dist2[cent_ind, :]
        labels = np.ones(n_pts)
        labels[dist2_stat>max_allowed_dist] = 0
        if((labels==1).sum() > PEAK_MIN_PIX):
            tmp_data = data[:, labels == 1]
            for dim_cnt in range(3):
                tmp_data[dim_cnt, :] = tmp_data[dim_cnt, :] - blob_cent[dim_cnt]
            u, s, _ = np.linalg.svd(tmp_data, full_matrices=False)
            tmp_data2 = data.copy()
            for dim_cnt in range(3):
                tmp_data2[dim_cnt, :] = tmp_data2[dim_cnt, :] - blob_cent[dim_cnt]
            v = u @ tmp_data2
            dist2_stat = (v**2).sum(0)**0.5
            mP_std = MSSE(dist2_stat, 
                          k = int(PEAK_MIN_PIX/2), 
                          minimumResidual = 1)
            #fig = scatter3(data[:, labels==0], returnFigure = True, label = 'zero')
            #scatter3(data[:, labels==1], inFigure = fig , label = 'one')
            #plt.show()
            labels = np.ones(n_pts)
            labels[dist2_stat > 1.5 * mP_std] = 0
            #fig = scatter3(data[:, labels==0], returnFigure = True, label = 'zero')
            #scatter3(data[:, labels==1], inFigure = fig , label = 'one')
            #plt.show()
            print(' done.')
        else:
            print(' removed with ', (labels==1).sum(), 'points')
            #fig = scatter3(data[:, labels==0], returnFigure = True, label = 'zero')
            #scatter3(data[:, labels==1], inFigure = fig , label = 'one')
            #plt.show()
            labels = np.zeros(n_pts)
        loser = 0
        
        peakMap[inds_i[labels == loser], 
                inds_j[labels == loser], 
                inds_k[labels == loser]] = 0
    np.save('output/peakMap_fixed_bad.npy', peakMap)
else:
    peakMap = np.load('output/peakMap_fixed_bad.npy')
        
if(True):
    for label_cnt in range(peakMap.max()):
        n_peaks = (peakMap == label_cnt).sum()
        if(n_peaks > PEAK_MAX_PIX):
            peakMap[peakMap == label_cnt] = 0

if(0):
    plt.imshow(peakMap.sum(0), cmap='hsv')
    plt.show()
    
if(0):
    for idx, an_image in enumerate(peakMap):
        plt.imshow(an_image, cmap='hsv', 
                   vmin = 0, vmax = peakMap.max())
        plt.draw()
        plt.pause(5)
        plt.clf()

if(0):
    inds_i, inds_j, inds_k = np.where(peakMap>0)
    data = np.array([inds_i, inds_j, inds_k])
    scatter3(data)

if(0):
    print(peakMap.max())
    fig = plt.figure()
    for idx, an_image in enumerate(IMG):
        print(idx)
        ax = fig.gca()
        ax.imshow(an_image, cmap='gray', vmin=vmin, vmax=vmax)
        for peakCnt in range(1, peakMap.max() + 1):
            inds_i, inds_j = np.where(peakMap[idx]==peakCnt)
            print(peakCnt)
            if(inds_i.shape[0]>0):
                if(inds_i.shape[0]>4):
                    data = np.array([inds_i, inds_j])   
                    if(True):
                        hull = ConvexHull(data.T)
                        for simplex in hull.simplices:
                            plt.plot(data[1, simplex], data[0, simplex], 'r-')
                else:
                    inds_img, inds_row, inds_clm = np.where(peakMap==peakCnt)
                    sorted_inds_img = np.sort(inds_img)
                    first_imag = sorted_inds_img[0]
                    last_imag = sorted_inds_img[-1]
                    
                    if( (first_imag <= idx) & (idx <= last_imag) ):
                    
                        first_row = inds_row.min()
                        last_row = inds_row.max()
                        first_clm = inds_clm.min()
                        last_clm = inds_clm.max()
                        rect = Rectangle((first_clm, first_row),
                                         last_row-first_row,
                                         last_clm-first_clm,
                                         linewidth=1, edgecolor='red' ,facecolor='none')
                        ax.add_patch(rect)
                        ax.set_title(idx)
        fig.savefig('tmp.jpg')
        plt.draw()
        plt.pause(1)
        plt.clf()
        
        
    
    
IMG_out = np.array([IMG, 0*IMG, 0*IMG])
print(IMG_out.shape)
IMG_out = np.swapaxes(IMG_out, 0, 1)
IMG_out = np.swapaxes(IMG_out, 1, 2)
IMG_out = np.swapaxes(IMG_out, 2, 3)

    
print(peakMap.shape)
print(IMG.shape[0])     
       
for i in range(IMG.shape[0]):
    

    AMAP=peakMap[i,:,:]

 
    inds_i, inds_j = np.where(AMAP>0)
    IMG_out[ i, inds_i, inds_j , 1] = 255        
        

imwrite('/home/ali/Projects/NEW_RPF/3D/TestAfterRefine.tif', IMG_out , imagej=True)
