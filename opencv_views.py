# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 20:06:36 2016

@author: channerduan
"""

import cv2
import numpy as np

def nothing(x):
    pass

# Create a black image, a window
image = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)
while(1):
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        image[:] = 0
    else:
        image[:] = [b,g,r]
    cv2.imshow('image',image)
cv2.destroyAllWindows()