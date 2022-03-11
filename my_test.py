import numpy as np
#from skimage.data import cells3d
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from  skimage.io import imread , imsave

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


import RobustPeakFinder as RPF
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
import hdbscan

from RobustGaussianFittingLibrary.misc import scatter3

import imageio



class IndexTracker:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind], cmap="gray")
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

########################
            
img = imread('/home/ali/Projects/NEW_RPF/21.08.18 Mouse 1 G-CSFR Imiq Ear.tif')

if(False):
    winx_cent = 673;
    winy_cent = 1666;
    win_side = 400;
    IMG = img[20:32, 
              winx_cent - win_side:winx_cent + win_side,
              winy_cent - win_side:winy_cent + win_side]
else:
    
    IMG = img


vmin = IMG.min()
vmax = IMG.max()

if(False):
    fig , axs = plt.subplots(nrows=3, ncols=4, figsize=(16, 14))
    fig.subplots_adjust(hspace = .5, wspace=.001)
    
    axs = axs.ravel()
    
    for idx, an_image in enumerate(IMG):
        axs[idx].imshow(an_image, cmap='gray', vmin=vmin, vmax=vmax)
    
    plt.show()

if(False):
    for idx, an_image in enumerate(IMG):
        plt.imshow(an_image,cmap='gray')
        plt.title(idx)
        plt.draw()
        plt.pause(0.5)
        plt.clf()
if(False):
    IMG = np.random.randn(30, 50, 50);
    IMG[3:7, 8:12, 13:17] = 10; # z y x , image, mat_row, mat_col
    IMG[14:18, 19:23, 24:28] = 4; # z y x , image, mat_row, mat_col

    plt.imshow(IMG[5]), plt.show()

    print(IMG)

print('IMG.shape: ', IMG.shape)
PEAK_MAX_PIX = 5000
if(False):
    rgrid = np.linspace(0, IMG.shape[1], 20, dtype = 'int')
    cgrid = np.linspace(0, IMG.shape[2], 10, dtype = 'int')
    for rcnt in np.arange(15, rgrid.shape[0]-1, 1):
        for ccnt in np.arange(cgrid.shape[0]-1):
            peakList , peakMap = RPF.robustPeakFinderPyFunc(
                inData = IMG[:, rgrid[rcnt]:rgrid[rcnt+1], cgrid[ccnt]:cgrid[ccnt+1]], 
                                    inMask = None,
                                    peakMask = None,
                                    minBackMeanMap = 30, #30 or 0 or 100
                                    singlePhotonADU = 1.0,
                                    maxBackMeanMap = None, 
                                    bckSNR = 3.0,
                                    pixPAPR = 1.0,
                                    PTCHSZ = 50, #50
                                    PTCHSZz = 5,
                                    PEAK_MAX_PIX = PEAK_MAX_PIX, #500
                                    PEAK_MIN_PIX = 50, #60
                                    MAXIMUM_NUMBER_OF_PEAKS = 1024,
                                    optIters = 6,
                                    finiteSampleBias = 500,#1000
                                    downSampledSize = 500, #1000
                                    MSSE_LAMBDA = 3.5,
                                    searchSNR = None,
                                    highPoissonTh = 0,
                                    lowPoissonTh = 0,
                                    returnPeakMap = True)
            fname = 'output/peakMap' + '%2d'%rcnt +'_' + '%2d'%ccnt + '.npz'
            print(fname)
            np.savez(fname, 
                 peakList = peakList,
                 peakMap = peakMap)
else:
    rgrid = np.linspace(0, IMG.shape[1], 20, dtype = 'int')
    cgrid = np.linspace(0, IMG.shape[2], 10, dtype = 'int')
    peakMap = np.zeros(shape = IMG.shape, dtype='int')
    win_cnt = 0;
    for rcnt in np.arange(rgrid.shape[0]-1):
        for ccnt in np.arange(cgrid.shape[0]-1):
            fname = 'output/peakMap' + '%2d'%rcnt +'_' + '%2d'%ccnt + '.npz'
            print(fname)
            dictz = np.load(fname)
            peakList_tmp = dictz['peakList']
            peakMap_tmp = dictz['peakMap']
            peakMap_tmp[peakMap_tmp>0] = peakMap_tmp[peakMap_tmp>0] + peakMap.max() 
            peakMap[:, rgrid[rcnt]:rgrid[rcnt+1], cgrid[ccnt]:cgrid[ccnt+1]] = peakMap_tmp;
    np.savez('output/peakMap.npz', peakMap = peakMap)

print('peakMap is made')

'''
if(False):
    peakList , peakMap = RPF.robustPeakFinderPyFunc(inData = IMG, 
                                inMask = None,
                                peakMask = None,
                                minBackMeanMap = 30, #30 or 0 or 100
                                singlePhotonADU = 1.0,
                                maxBackMeanMap = None, 
                                bckSNR = 3.0,
                                pixPAPR = 1.0,
                                PTCHSZ = 50, #50
                                PTCHSZz = 5,
                                PEAK_MAX_PIX = PEAK_MAX_PIX, #500
                                PEAK_MIN_PIX = 50, #60
                                MAXIMUM_NUMBER_OF_PEAKS = 1024,
                                optIters = 6,
                                finiteSampleBias = 500,#1000
                                downSampledSize = 500, #1000
                                MSSE_LAMBDA = 3.5,
                                searchSNR = None,
                                highPoissonTh = 0,
                                lowPoissonTh = 0,
                                returnPeakMap = True)
    
    np.savez('peakMap6.npz', 
             peakList = peakList,
             peakMap = peakMap)
else:
    dictz = np.load('peakMapS.npz')
    peakList = dictz['peakList']
    peakMap = dictz['peakMap']
'''
n_peaks = peakMap.max()
print('n_peaks: ', n_peaks)
print(peakMap.shape)
#print(peakList)
print(IMG.shape)

#fig, ax = plt.subplots()

#fig.subplots_adjust(hspace = .5, wspace=.001)

#axs = axs.ravel()


########################################PEAK SHOW
if(False):
    fig = plt.figure()
    for idx, an_image in enumerate(IMG):
        
        ax = fig.gca()
        
        #c = next(color)
         
        ax.imshow(an_image, cmap='gray', vmin=vmin, vmax=vmax)
        for peakCnt in range(1, n_peaks):
            inds_img, inds_row, inds_clm = np.where(peakMap==peakCnt)
    
            if(inds_img.shape[0]>0):
        
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
        plt.pause(2)
        plt.clf()

##########################################


#n = peakList.shape[0]

    
#pass
 #   plt.pause(2)
  #  plt.close()
    #ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    #ax.set_xticks([])
    #ax.set_yticks([])
#plt.show()


###################JUST FOR NOW    
#pass
##### Calculating the distances between the peaks
#peakList1 = peakList[0:np.int(peakList.shape[0]/2),:]
#peakList2 = peakList[np.int(peakList.shape[0]/2):,:]
#p1 = np.array([peakList1[:,0], peakList1[:,1], peakList1[:,2]])
#p2 = np.array([peakList2[:,0], peakList2[:,1], peakList2[:,2]])

#squared_dist = np.sum((p1-p2)**2, axis=0)
#dist = np.sqrt(squared_dist)
#########################HERE
#data, _ = make_blobs(1000)

inds_i, inds_j, inds_k = np.where(peakMap>0)
data = np.array([inds_i, inds_j, inds_k])
data[0, :] = data[0, :]*4;
data[1, :] = data[1, :]*0.83;
data[2, :] = data[2, :]*0.83;

clusterer = hdbscan.HDBSCAN(min_cluster_size=30, alpha = 4.0)
cluster_labels = clusterer.fit_predict(data.T)

if(False):
    for idx, an_image in enumerate(peakMap):
        plt.imshow(an_image, cmap='hsv', vmin = 0, vmax = peakMap.max())
        plt.draw()
        plt.pause(3)
        plt.clf()

peakMap[peakMap>0] = cluster_labels
peakMap = peakMap.astype('int')
if(False):
    (fig, ax, figCnt) = scatter3(data, returnFigure = True)
    for idx, an_image in enumerate(peakMap):
        plt.imshow(an_image, cmap='hsv', vmin = 0, vmax = peakMap.max())
        plt.draw()
        plt.pause(3)
        plt.clf()

peaks_stat = np.zeros((np.unique(peakMap).shape[0], 3))
for label_cnt in np.unique(peakMap[peakMap>0]):
    inds_i, inds_j, inds_k = np.where(peakMap == label_cnt)
    if(inds_i.shape[0]<PEAK_MAX_PIX):
        
        data = np.array([inds_i, inds_j, inds_k])   
        
        peaks_stat[label_cnt, 0] = data.size
        data[0, :] = data[0, :] - data[0, :].mean()
        data[1, :] = data[1, :] - data[1, :].mean()
        data[2, :] = data[2, :] - data[2, :].mean()
        
        _, s, _ = np.linalg.svd(data, full_matrices=False)
        vol = np.prod(s)
        peaks_stat[label_cnt, 1] = vol
        peaks_stat[label_cnt, 2] = s[0]/s[1]
    else:
        peaks_stat[label_cnt, 0] = inds_i.shape[0]
        peaks_stat[label_cnt, 1] = 1e10

        
crit_vol_to_size = peaks_stat[1:, 1]/peaks_stat[1:, 0];
print(peaks_stat)
print(crit_vol_to_size)
if(False):
    worst_peak_ind = 1 + np.argmax(crit_vol_to_size)
    peakMap_tmp = peakMap.copy()
    peakMap_tmp[peakMap != worst_peak_ind] = 0
    for idx, an_image in enumerate(peakMap_tmp):
        plt.imshow(an_image, cmap='hsv', vmin = 0, vmax = peakMap_tmp.max())
        plt.title(idx)
        plt.draw()
        plt.pause(1)
        plt.clf()

if(False):
    fig = plt.figure()
    for idx, an_image in enumerate(IMG):
        
        ax = fig.gca()
        
        #c = next(color)
         
        ax.imshow(an_image, cmap='gray', vmin=vmin, vmax=vmax)
        for peakCnt in range(1, peakList.shape[0]):
            inds_img, inds_row, inds_clm = np.where(peakMap==peakCnt)
    
            if(inds_img.shape[0]>0):
        
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
        plt.pause(0.2)
        plt.clf()

if(True):
    print(crit_vol_to_size)
    for peak_cnt in np.unique(peakMap[peakMap>0]):
        if(crit_vol_to_size[peak_cnt-1]>2000):
            peakMap[peakMap == peak_cnt] = 0
        #if(peaks_stat[peak_cnt-1, 1]>1e+8):
        #    peakMap[peakMap == peak_cnt] = 0
        if(peaks_stat[peak_cnt-1, 2]>5):
            peakMap[peakMap == peak_cnt] = 0

if(False):
    for idx, an_image in enumerate(peakMap):
        plt.imshow(an_image, cmap='hsv', vmin = 0, vmax = peakMap.max())
        plt.draw()
        plt.pause(3)
        plt.clf()
## for plotting subplot
##fig , axs = plt.subplots(nrows=3, ncols=4)
##fig.subplots_adjust(hspace = .5, wspace=.001)
##axs = axs.ravel()
#fig, ax = plt.subplots(1, 1)

#tracker = IndexTracker(ax, IMG.T)

n = np.unique(peakMap).shape[0]
color=iter(cm.rainbow(np.linspace(0,1,n)))
#GIF WRITER
with imageio.get_writer(uri='output.gif', mode='i',fps=1) as writer:
    
    for idx, an_image in enumerate(IMG):
        
        fig = plt.figure()
        ax = fig.gca()
        
        #c = next(color)
         
        ax.imshow(an_image, cmap='gray', vmin=vmin, vmax=vmax)
        for peakCnt in range(1, peakList.shape[0]):
            inds_img, inds_row, inds_clm = np.where(peakMap==peakCnt)
    
            if(inds_img.shape[0]>0):
        
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
        plt.pause(2)
        plt.close()

        buf = imageio.imread('tmp.jpg')
    
        writer.append_data(buf)
        print('idx: ', idx)

#plt.show()


#############3D  viwer



