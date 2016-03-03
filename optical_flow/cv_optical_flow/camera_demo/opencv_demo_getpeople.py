# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:07:52 2016

@author: yx
"""

#Demo: extract from the background

import numpy as np
import cv2
from numpy import *
from numba.decorators import autojit


sigma = 2
size = 11

def video_flow_FB_getpeople():

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
        
        (cnts, _) = cv2.findContours(bg.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)        
        for c in cnts:
            if cv2.contourArea(c) < 8000:
                continue 
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('flow', frame2)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cam.release()
            cv2.destroyAllWindows()
            break
        
        prvs = next


def get_background(im,flow):
    tem = im
    N,M = tem.shape
    for (i,j),n in np.ndenumerate(tem):
        if j >= 2 and j < (M-1) and i >= 2 and i < (N-1): 
            if np.abs(flow[i][j][0]-flow[i][j][1])<3:
                tem[i][j] = 0
            else :
                tem[i][j] = 255
    return tem

video_flow_FB_getpeople()