'''
This is a simple example to show how to use the Robust Peak Finder. 
The program generates a syntethic pattern with a noisy background and a few peaks where some of them are masked out
The program, then, calls the wrapper and generates a 2D array output including the peak locations as described in the wrapper.
Run the code many times!
'''

import RobustPeakFinder_Python_Wrapper
import matplotlib.pyplot as plt
from os import getpid
import numpy
import scipy.stats
import time

def gkern(kernlen):
    lim = kernlen//2 + (kernlen % 2)/2
    x = numpy.linspace(-lim, lim, kernlen+1)
    kern1d = numpy.diff(scipy.stats.norm.cdf(x))
    kern2d = numpy.outer(kern1d, kern1d)
    return kern2d/(kern2d.flatten().max())

def diffractionPatternMaker(XSZ, YSZ, WINSIZE, inputPeaksNumber, numOutliers):    
    inData = numpy.zeros((XSZ, YSZ), dtype='float32')
    
    n_R = inData.shape[0]
    n_C = inData.shape[1]
    
    
    inMask = numpy.ones(inData.shape, dtype = 'uint8')
    inMask[-1, :] = 0
    inMask[ 0, :] = 0
    inMask[ :, 0] = 0
    inMask[:, -1] = 0
    
    for ccnt in range(inData.shape[1]):
        for rcnt in range(inData.shape[0]):
            inData[rcnt, ccnt] += 100 + 400*numpy.exp(-(((rcnt-n_R/2)**2+(ccnt-n_C/2)**2)**0.5 - n_R/4)**2/(2*(n_R/15)**2))
            inData[rcnt, ccnt] += 3*numpy.sqrt(inData[rcnt, ccnt])*numpy.random.randn(1)    
    
    randomLocations = numpy.random.rand(2,inputPeaksNumber)
    randomLocations[0,:] = XSZ/2 + numpy.floor(XSZ*0.8*(randomLocations[0,:] - 0.5))
    randomLocations[1,:] = YSZ/2 + numpy.floor(YSZ*0.8*(randomLocations[1,:] - 0.5))
    
    for cnt in numpy.arange(inputPeaksNumber):    
        bellShapedCurve = 1000*gkern(WINSIZE)
        winXStart = (randomLocations[0, cnt] - (WINSIZE-1)/2).astype(numpy.int)
        winXEnd = (randomLocations[0, cnt] + (WINSIZE+1)/2).astype(numpy.int)
        winYStart = (randomLocations[1, cnt] - (WINSIZE-1)/2).astype(numpy.int)
        winYEnd = (randomLocations[1, cnt] + (WINSIZE+1)/2).astype(numpy.int)
        inData[ winXStart : winXEnd, winYStart : winYEnd ] += bellShapedCurve;
        if (cnt >= inputPeaksNumber - numOutliers):
            inMask[ winXStart : winXEnd, winYStart : winYEnd ] = 0;    
    
    return(inData, inMask, randomLocations)
    
numpy.set_printoptions(precision=2, suppress=True)
   
if __name__ == '__main__':    
    print('PID ->' + str(getpid()))
    XSZ = 300
    YSZ = 350
    WINSIZE = 7
    inputPeaksNumber = 25
    numOutliers = 5
    print("Generating a pattern with " + str(inputPeaksNumber) + " peaks...")
    
    inData, inMask, randomLocations = diffractionPatternMaker(XSZ, YSZ, WINSIZE, inputPeaksNumber, numOutliers)

    #peakMask = 1 - inMask.copy()

    print("Pattern Ready! Calling the Robust Peak Finder...")
    time_time = time.time()
    peakList, peakMap = RobustPeakFinder_Python_Wrapper.robustPeakFinderPyFunc(inData = inData, 
                                                                               inMask = inMask, 
                                                                               bckSNR=6.0,
                                                                               returnPeakMap = True)
    print('RPF finished in ' + '%4f'%(time.time() - time_time) +' seconds')
    print("RPF: There are " + str(peakList.shape[0]) + " peaks in this image!")
    print(peakList[:, :2].T)
    print(randomLocations)
    peakMap[peakMap>0] = 1
    plt.imshow((inData*inMask*peakMap).T)
    plt.plot(peakList[:, 0], peakList[:, 1],'o')
    plt.plot(randomLocations[0,:], randomLocations[1,:],'x')
    plt.show()
    inTensorData = numpy.zeros([2, inData.shape[0], inData.shape[1]])
    inTensorMask = numpy.zeros([2, inData.shape[0], inData.shape[1]])
    for cnt in range(inTensorData.shape[0]):
        _tmpData, _tmpMask, randomLocations = diffractionPatternMaker(XSZ, YSZ, WINSIZE, inputPeaksNumber, numOutliers)
        inTensorData[cnt] = _tmpData
        inTensorMask[cnt] = _tmpMask
        print('pattern ' +str(cnt) + ' is generated.', flush = True)
    
    nPeaks, peakListTensor, peakMapTensor = \
        RobustPeakFinder_Python_Wrapper.robustPeakFinderPyFunc_multiproc(inData = inTensorData, 
                                                                         inMask = inTensorMask, 
                                                                         bckSNR = 6.0,
                                                                         returnPeakMap = True)
    peakMapTensor[peakMapTensor>0] = 1
    proc = (_tmpData*_tmpMask*peakMapTensor[1]).copy()
    plt.imshow(proc.T)
    plt.plot(peakListTensor[1, :, 0], peakListTensor[1, :, 1],'o')
    plt.plot(randomLocations[0,:], randomLocations[1,:],'x')
    plt.show()

    proc = (_tmpData*_tmpMask).copy()
    plt.imshow(proc.T)
    plt.plot(peakListTensor[1, :, 0], peakListTensor[1, :, 1],'o')
    plt.plot(randomLocations[0,:], randomLocations[1,:],'x')
    plt.show()
    exit(0)
