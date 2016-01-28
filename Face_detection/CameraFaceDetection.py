# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 22:31:42 2016

@author: channerduan
"""
import cv2
import numpy as np
cv2.namedWindow("test")
cap=cv2.VideoCapture(0)
suc,frame=cap.read()
classifier=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

print frame.shape

while suc:
    suc,frame=cap.read()
    size=frame.shape[:2]
    image=np.zeros(size,dtype=np.float16)
    image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(image,image)
    divisor=8
    h,w=size
    minSize=(w/divisor,h/divisor)
    faceRects=classifier.detectMultiScale(image,1.2,2,cv2.CASCADE_SCALE_IMAGE,minSize)
    if len(faceRects)>0:
        for faceRect in faceRects:
            x,y,w,h=faceRect
            cv2.circle(frame,(x+w/2,y+h/2),min(w/2,h/2),(255,0,0))
            cv2.circle(frame,(x+w/4,y+h/4),min(w/8,h/8),(255,0,0))
            cv2.circle(frame,(x+3*w/4,y+h/4),min(w/8,h/8),(255,0,0))
            cv2.rectangle(frame,(x+3*w/8,y+3*h/4),(x+5*w/8,y+7*h/8),(255,0,0))    
    cv2.imshow("test",frame) 
    key=cv2.waitKey(10)
    c=chr(key&255)
    if c in ['q']:
        break

cv2.destroyWindow("test")