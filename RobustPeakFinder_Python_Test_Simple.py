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
    XSZ = 300
    YSZ = 400
    WINSIZE = 21
    inputPeaksNumber = 30
    print("Generating a pattern with " + str(inputPeaksNumber) + " peaks...")

    inData = 300 + numpy.floor(50 * numpy.random.randn(XSZ, YSZ))
    inMask = 1 + 0*inData
    randomLocations = XSZ/2 + numpy.floor(XSZ*0.8*(numpy.random.rand(2,inputPeaksNumber) - 0.5))
    for cnt in numpy.arange(inputPeaksNumber):    
        bellShapedCurve = 500*gkern(WINSIZE, nsig = 1)
        winXStart = (randomLocations[0, cnt] - (WINSIZE-1)/2).astype(numpy.int)
        winXEnd = (randomLocations[0, cnt] + (WINSIZE+1)/2).astype(numpy.int)
        winYStart = (randomLocations[1, cnt] - (WINSIZE-1)/2).astype(numpy.int)
        winYEnd = (randomLocations[1, cnt] + (WINSIZE+1)/2).astype(numpy.int)
        inData[ winXStart : winXEnd, winYStart : winYEnd ] += bellShapedCurve;
        if (cnt == 25):
            print("Masking 5 peaks...")
        if (cnt > 25):
            inMask[ winXStart : winXEnd, winYStart : winYEnd ] = 0;
        
    print("Pattern Ready! Calling the Robust Peak Finder...")
    outdata = RobustPeakFinder_Python_Wrapper.robustPeakFinderPyFunc(inData = inData, inMask = inMask, bckSNR = 5.0, MAXIMUM_NUMBER_OF_PEAKS = 1024)
    print("RPF: There is " + str(outdata.shape[0]) + " peaks in this image!")
    print(outdata)

    plt.imshow(inData*inMask)
    plt.scatter(outdata[:, 0], outdata[:, 1], s=80, facecolors='none', edgecolors='r')
    plt.show()
