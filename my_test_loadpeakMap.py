import numpy as np
import os
import RobustPeakFinder as RPF
import skimage.io

########################
PEAK_MAX_PIX = 5000
output_dir = 'output/'
in_file_path = '/home/ali/Projects/NEW_RPF/21.08.18 Mouse 1 G-CSFR Imiq Ear.tif'
n_seg_R = 29
n_seg_C = 14

if(not os.path.isdir(output_dir)):
    os.system('mkdir -p ' + output_dir)

IMG = skimage.io.imread(in_file_path)
IMG = IMG[:, :, 78:-78]

print('IMG loaded with shape ', IMG.shape)

n_D, n_R, n_C = IMG.shape
n_Rs = int(n_R/n_seg_R)
n_Cs = int(n_C/n_seg_C)

IMG = IMG.reshape(n_D, n_seg_R, n_Rs, n_C)
IMG = np.swapaxes(IMG, 0, 1)
IMG = IMG.reshape(n_seg_R, n_D, n_Rs, n_seg_C, n_Cs)
IMG = np.swapaxes(IMG, 2, 3)
IMG = np.swapaxes(IMG, 1, 2)
IMG = IMG.reshape(n_seg_R * n_seg_C, n_D, n_Rs, n_Cs)

inMask = np.ones(IMG.shape, dtype = 'uint8')
inMask[IMG<=0] = 0
_, _, peakMap = RPF.robustPeakFinderPyFunc_multiproc(
    inData = IMG, 
    inMask = inMask,
    peakMask = None,
    minBackMeanMap = 0, 
    singlePhotonADU = 1,
    maxBackMeanMap = None, 
    bckSNR = 3.0,
    pixPAPR = 2.0,
    PTCHSZ = 190,
    PTCHSZz = 10,
    PEAK_MAX_PIX = PEAK_MAX_PIX,
    PEAK_MIN_PIX = 50,
    MAXIMUM_NUMBER_OF_PEAKS = 1024,
    optIters = 8,
    finiteSampleBias = 250,
    downSampledSize = 250,
    MSSE_LAMBDA = 3.5,
    searchSNR = None,
    highPoissonTh = 0,
    lowPoissonTh = 0,
    multiproc_stride = None,
    returnPeakMap = True)
print(len(peakMap))
fname = output_dir + 'peakMap' + '.npz'
print(fname)
np.savez(fname, peakMap = peakMap)
print('peakMap is calculated!')
exit()