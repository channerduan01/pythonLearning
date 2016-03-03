# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:07:52 2016

@author: yx
"""

#Demo: extract from the background

import numpy as np
import cv2
from numba.decorators import autojit


sigma = 2
size = 11

def video_flow_FB():

    cam = cv2.VideoCapture(0)
    ret, frame1 = cam.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)


    while(1):
        ret, frame2 = cam.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)

        imFilter = cv2.GaussianBlur(next,(5,5),1.5)
        bg_numba = autojit(get_background)
        bg = bg_numba(imFilter,flow)
        edge_numba = autojit(get_edge)
        lap = edge_numba(bg)
        lap = get_edge_sobel(bg)
        cv2.imshow('flow', bg)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cam.release()
            cv2.destroyAllWindows()
            break

        prvs = next
    
def get_edge(im):
    gray_lap = cv2.Laplacian(im,cv2.CV_16S,ksize = 3)  
    dst = cv2.convertScaleAbs(gray_lap) 
    return dst

def get_edge_sobel(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)  
    y = cv2.Sobel(img,cv2.CV_16S,0,1)  
   
    absX = cv2.convertScaleAbs(x)   
    absY = cv2.convertScaleAbs(y)  
  
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)  
    return dst


def get_background(im,flow):
    #tem = np.zeros(im.shape)
    tem = im
    N,M = tem.shape
    for (i,j),n in np.ndenumerate(tem):
        if j >= 2 and j < (M-1) and i >= 2 and i < (N-1): 
            if np.abs(flow[i][j][0]-flow[i][j][1])<3:
                tem[i][j] = 0
            else :
                tem[i][j] = 255
    return tem

video_flow_FB()