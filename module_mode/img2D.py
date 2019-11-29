import numpy as np
import skimage
import scipy.io
import pydicom
import os
import sys
import datetime
from pylab import *
import ants
from time import time

#image display and store them in visuals folder
def imgplot(volume1, volume2 = None,  size = 'big',scaling_factor = 1, cur_path = (os.getcwd()),
            slice_no = [None], titles = [None]):
    import matplotlib.pyplot as plt
    import datetime
    if type(slice_no[0]).__name__ == 'NoneType':
        print("Slice no for image 1 not provided. Dont worry, I am taking the default value to plot.")
        slice_no1 = int(volume1.shape[2]/2)
    else:
        slice_no1 = slice_no[0]
    name = str((datetime.datetime.now()).strftime("%d%m%Y_%H%M%S")) + '.png'
    if type(volume2).__name__ == 'NoneType':
        if size == 'big':
            plt.figure(figsize = (8,6))
            plt.imshow(volume1[:,:,slice_no1], cmap = plt.get_cmap('gray'))
            plt.savefig(cur_path + '/visuals/stills_2d/' + name)
            plt.show()
        elif size == 'small':
            plt.imshow(volume1[:,:,slice_no1], cmap = plt.get_cmap('gray'))
            plt.savefig(cur_path + '/visuals/stills_2d/' + name)
            plt.show()
    elif type(volume2).__name__ != 'NoneType':    
        if type(slice_no[1]).__name__ == 'NoneType':
            print("Slice no for image 2 not provided. Dont worry, I am taking the default value to plot.")
            slice_no2 = int(volume2.shape[2]/2)
        else:
            slice_no2 = slice_no[1]
        plt.figure(figsize = (14,16))
        plt.subplot(2,2,1)
        ax = plt.gca()
        im = ax.imshow(volume1[:,:,slice_no1], cmap = plt.get_cmap('gray'))
        plt.colorbar(im, fraction = 0.046, pad = 0.04)
        plt.subplot(2,2,2)
        ax = plt.gca()
        im = ax.imshow(volume2[:,:,slice_no2], cmap = plt.get_cmap('gray'))
        plt.colorbar(im, fraction = 0.046, pad = 0.04)
        plt.savefig(cur_path + '/visuals/stills_2d/' + name)
        plt.show()
        
        '''
        ax = plt.gca()
        im = ax.imshow(p2_arr[:,:,j], cmap = plt.get_cmap('gray'))
        plt.title("SWI MRI slice of the data")
        plt.colorbar(im, fraction = 0.046, pad = 0.04)
        '''

def grad_image(image):
    from scipy.ndimage import sobel, generic_gradient_magnitude, gaussian_filter
    grad = generic_gradient_magnitude(image, sobel)
    return grad

    
