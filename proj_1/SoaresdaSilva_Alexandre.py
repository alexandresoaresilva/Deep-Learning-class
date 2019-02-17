# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 20:28:08 2019

@author: alexa
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt


I = cv.imread('Matlab_Data\cameraman.tif', cv.IMREAD_GRAYSCALE)

#plt.imshow(I)


#If = np.fft.fftshift(np.fft.fft2(I))
If = np.fft.fftshift(np.fft.fft2(I))
If_log = np.log(1 + np.abs(If))
If_disp = If_log/np.max(If_log)
plt.imshow(If_disp,cmap=plt.cm.gray)
plt.show()

#import matplotlib.pyplot as plt
#
#data = [[0, 0.25], [0.5, 0.75]]
#
#fig, ax = plt.subplots()
#im = ax.imshow(data, cmap=plt.get_cmap('hot'), interpolation='nearest',
#               vmin=0, vmax=1)
#fig.colorbar(im)
#plt.show()
