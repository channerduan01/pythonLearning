# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 09:29:16 2016

@author: channerduan
"""

import cv2
cv2.namedWindow("test")
cap = cv2.VideoCapture(0)
while True:
	suc,frame = cap.read()
	cv2.imshow("test",frame)
	key = chr(cv2.waitKey(100)&255)
	if key in ['q']:
		break
cap = None
cv2.destroyWindow("test")