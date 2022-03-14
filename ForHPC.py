import numpy as np
#import cv2
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

#from RobustGaussianFittingLibrary.misc import scatter3

import imageio






########################
img = imread('/home/ali/Projects/NEW_RPF/21.08.18 Mouse 1 G-CSFR Imiq Ear.tif')
print(img.shape[0])

#for i in range(img.shape[0]):
IMG=img[10:32,500:700,1000:1200]
IMG_out = np.array([IMG, 0*IMG, 0*IMG])
print(IMG_out.shape)
IMG_out = np.swapaxes(IMG_out, 0, 1)
IMG_out = np.swapaxes(IMG_out, 1, 2)
IMG_out = np.swapaxes(IMG_out, 2, 3)

print(IMG_out.shape)
print(IMG.shape)

vmin = IMG.min()
vmax = IMG.max()

fig = plt.figure()
for i in range(IMG.shape[0]):
    

#fig = plt.figure()
#plt.imshow(IMG, cmap='gray', vmin=vmin, vmax=vmax)
#plt.show()

    AMAP=np.zeros(IMG[i].shape)
    AMAP[50:70,100:120] =1
    
    #a1 = IMG[i,:,:]
    #a2 = np.zeros(a1.shape)
    
    #a =np.stack((a1 , a2 , a2)) 
    
    #IMG_out = a.T
    #IMG_out = IMG_out[..., np.newaxis]
    #print(IMG_out.shape)
 
    inds_i, inds_j = np.where(AMAP>0)
    IMG_out[ i, inds_i, inds_j , 1] = 255
    
    
    #plt.imshow(IMG_out)
    plt.imshow(IMG_out[i,:,:, :])
    #plt.imsave('TEST.png', IMG_out.astype('uint8'))
    plt.draw()
    plt.pause(1)
    plt.clf()
    
    
imsave('test2.tif',  IMG_out,photometric='minisblack')    
#imageio.imwrite("image.tif", IMG_out)
np.save('IMGtotif.npy',IMG_out)



    

#backtorgb = cv2.cvtColor(IMG_out,cv2.COLOR_GRAY2RGB)

#cv2.imshow('RGB image', backtorgb)



'''
#img = cv2.imread(file_path, 1) 
#print(img.shape) # this should give you (img_h, img_w, 3)
img2 = cv2.cvtColor(IMG_out, cv2.COLOR_BGR2GRAY)

#blur = cv2.GaussianBlur(IMG_out,(25,25),0) # apply blur for contour
#ret, binary = cv2.threshold(blur,25,255,cv2.THRESH_BINARY) # apply threshold to blur image

#contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find countour
#obj_index = contours.index(max(contours, key=len)) # find index of largest object
contour_img = cv2.drawContours(IMG_out, [inds_i, inds_j],[inds_i, inds_j], (0,255,0), 3) # draw coutour on original image

plt.imshow(contour_img, cmap='gray')
plt.show()



'''
