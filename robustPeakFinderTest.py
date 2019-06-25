import robustPeakFinderWrapper

import numpy
import scipy.stats
import imageio
import matplotlib.pyplot as plt
import matplotlib

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def gkern(kernlen=21, nsig=3):
    lim = kernlen//2 + (kernlen % 2)/2
    x = numpy.linspace(-lim, lim, kernlen+1)
    kern1d = numpy.diff(scipy.stats.norm.cdf(x))
    kern2d = numpy.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

XSZ = 300
YSZ = XSZ
WINSIZE = 21
inputPeaksNumber = 25
indata = numpy.floor(30 * numpy.random.rand(XSZ, YSZ))
randomLocations = XSZ/2 + numpy.floor(XSZ*0.8*(numpy.random.rand(2,inputPeaksNumber) - 0.5))
for cnt in numpy.arange(inputPeaksNumber):	
    bellShapedCurve = 1000*gkern(WINSIZE, nsig = 1)
    winXStart = (randomLocations[0, cnt] - (WINSIZE-1)/2).astype(numpy.int)
    winXEnd = (randomLocations[0, cnt] + (WINSIZE+1)/2).astype(numpy.int)
    winYStart = (randomLocations[1, cnt] - (WINSIZE-1)/2).astype(numpy.int)
    winYEnd = (randomLocations[1, cnt] + (WINSIZE+1)/2).astype(numpy.int)
    indata[ winXStart : winXEnd, winYStart : winYEnd ] += bellShapedCurve;

outdata = robustPeakFinderWrapper.robustPeakFinderPyFunc(indata)
print("I found " + str(outdata.shape[0]) + " peaks in your diffraction pattern!")


showdata = outdata[:,:2]
fig = plt.figure()
plt.pcolor(indata)
plt.scatter(showdata[:,0],showdata[:,1])
plt.show()

indata = rgb2gray(imageio.imread('peaks.jpg'))
# indata = indata[:400,:400]
indata = indata.astype(numpy.double)
outdata = robustPeakFinderWrapper.robustPeakFinderPyFunc(indata, PEAK_MAX_PIX = 150, SNR_ACCEPT = 5.0)
print("I found " + str(outdata.shape[0]) + " stars in your sky!")
showdata = outdata[:,:2]

fig = plt.figure()
plt.pcolor(indata)
plt.scatter(showdata[:,0],showdata[:,1])
plt.show()