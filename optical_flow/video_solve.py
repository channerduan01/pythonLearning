# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 09:29:16 2016

@author: channerduan
"""

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def drawFigures(params):
    length = len(params)
    if (length < 2 or length%2 == 1):
        raise Exception("illegal input")
    for i in range(0,length,4):
        plt.figure()
        plt.subplot(121)
        plt.title(params[i])
        plt.axis('off')
        plt.imshow(params[i+1])
        if (i+2 < length):
            plt.subplot(122)
            plt.title(params[i+2])
            plt.axis('off')
            plt.imshow(params[i+3])
    return

videoCapture = cv2.VideoCapture('punch.mov')
fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), 
        int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        
print fps
print size

list_ = []

for i in range(100):
    success, frame = videoCapture.read()
    if success == False:
        break
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if i%10 == 0:
        list_.append(np.str(i/10*2+1))
        r = 200.0 / frame.shape[1]
        dim = (200, int(frame.shape[0] * r))
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        list_.append(frame)
        
drawFigures(list_)

videoCapture.release()


cv2.imwrite("test.png", list_[1])

#cap = cv2.VideoCapture("punch.mov")
#ret, frame1 = cap.read()
#prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#hsv = np.zeros_like(frame1)
#hsv[...,1] = 255
#
#while(1):
#    ret, frame2 = cap.read()
#    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
#
#    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#
#    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#    hsv[...,0] = ang*180/np.pi/2
#    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#
#    cv2.imshow('frame2',rgb)
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        break
#    elif k == ord('s'):
#        cv2.imwrite('opticalfb.png',frame2)
#        cv2.imwrite('opticalhsv.png',rgb)
#    prvs = next
#
#cap.release()
#cv2.destroyAllWindows()