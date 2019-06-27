'''
This is an example of how to use the wrapper for the Australian Synchotron HDF5 files

The example requires two files, YOUR_H5_MASTER_FILE_nAME and YOUR_H5_DATA_FILE_NAME

after that the example reads the first image in the data file and performs the
Robust Peak Finder and generates the output outdata.

The wrapper is called for default settings.

the output numpy 2d-array outdata has a number of rows equal to number of peaks
and four coloumns in the style of Cheetah's output.
Rows are peaks and coloums are:
-------------------------------------------------------------------------
Mass_Center_X, Mass_Center_Y, Mass_Total, Number of pixels in a peak
-------------------------------------------------------------------------

In this example, the input image is then plotted and peaks are plotted on top of that

'''

import RobustPeakFinder_Python_Wrapper

import hdf5plugin
import h5py
import numpy as np
import matplotlib.pyplot as plt

MasterFileName = YOUR_H5_MASTER_FILE_nAME
ReadFileName = YOUR_H5_DATA_FILE_NAME

file = h5py.File(MasterFileName,'r')
mpbdata = file['/entry/instrument/detector/detectorSpecific/pixel_mask']
mpbdata = mpbdata[()]
mpbdata[mpbdata<1]=0;
mpbdata[mpbdata>0]=-1;
inMask = 1+mpbdata;
file.close()

file = h5py.File(ReadFileName,'r')
datatmp = file['/entry/data/data']
file.close()

Frame_number = 1
inData = datatmp[Frame_number,:,:]
inData = inData.astype(np.double)
outdata = RobustPeakFinder_Python_Wrapper.robustPeakFinderPyFunc(inData, inMask)
print("There is " + str(outdata.shape[0]) + " peaks in this diffraction pattern!")

fig = plt.figure()
plt.imshow(inData, vmin=0, vmax= 100)
showdata = outdata[:,:2]
plt.scatter(showdata[:,0],showdata[:,1])
