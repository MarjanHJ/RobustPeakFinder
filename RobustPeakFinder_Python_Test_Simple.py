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

numpy.set_printoptions(precision=2, suppress=True)

def gkern(kernlen=21, nsig=3):
    lim = kernlen//2 + (kernlen % 2)/2
    x = numpy.linspace(-lim, lim, kernlen+1)
    kern1d = numpy.diff(scipy.stats.norm.cdf(x))
    kern2d = numpy.outer(kern1d, kern1d)
    return kern2d/(kern2d.flatten().max())

if __name__ == '__main__':    
    print('PID ->' + str(getpid()))
    XSZ = 1020
    YSZ = 980
    WINSIZE = 21
    inputPeaksNumber = 30
    print("Generating a pattern with " + str(inputPeaksNumber) + " peaks...")

    inData = numpy.zeros((XSZ, YSZ), dtype='float32')
    
    inMask = numpy.ones(inData.shape, dtype = 'uint8')
    inMask[-1, :] = 0
    inMask[ 0, :] = 0
    inMask[ :, 0] = 0
    inMask[:, -1] = 0
    
    for ccnt in range(inData.shape[1]):
        for rcnt in range(inData.shape[0]):
            inData[rcnt, ccnt] += 400*numpy.exp(-(((rcnt-512)**2+(ccnt-512)**2)**0.5 - 250)**2/(2*75**2))
            inData[rcnt, ccnt] += 3*numpy.sqrt(inData[rcnt, ccnt])*numpy.random.randn(1)    
    

    randomLocations = XSZ/2 + numpy.floor(XSZ*0.8*(numpy.random.rand(2,inputPeaksNumber) - 0.5))
    for cnt in numpy.arange(inputPeaksNumber):    
        bellShapedCurve = 400*gkern(WINSIZE, nsig = 1)
        winXStart = (randomLocations[0, cnt] - (WINSIZE-1)/2).astype(numpy.int)
        winXEnd = (randomLocations[0, cnt] + (WINSIZE+1)/2).astype(numpy.int)
        winYStart = (randomLocations[1, cnt] - (WINSIZE-1)/2).astype(numpy.int)
        winYEnd = (randomLocations[1, cnt] + (WINSIZE+1)/2).astype(numpy.int)
        inData[ winXStart : winXEnd, winYStart : winYEnd ] += bellShapedCurve;
        if (cnt >= 25):
            inMask[ winXStart : winXEnd, winYStart : winYEnd ] = 0;

    print("Pattern Ready! Calling the Robust Peak Finder...")
    outdata = RobustPeakFinder_Python_Wrapper.robustPeakFinderPyFunc(inData = inData, inMask = inMask)
    print("RPF: There is " + str(outdata.shape[0]) + " peaks in this image!")
    print(outdata[:, :2].T)
    print(randomLocations)

    plt.imshow((inData*inMask).T)
    plt.scatter(randomLocations[0], randomLocations[1], s=40, facecolors='none', edgecolors='c')
    plt.scatter(outdata[:, 0], outdata[:, 1], s=40, facecolors='none', edgecolors='r')
    plt.show()
    exit()
