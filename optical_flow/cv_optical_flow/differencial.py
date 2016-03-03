# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:00:16 2016

@author: yx
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from numpy import *
from scipy import signal



L1  = np.array(Image.open('1.png').convert('L'))
L2  = np.array(Image.open('2.png').convert('L'))

def draw_flow(img, u, v, step=6):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)

    fx = np.zeros(u[y,x].shape)
    fy = np.zeros(v[y,x].shape)
    fx = u[y,x]
    fy = v[y,x]

    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis
    
def optical_flow_HS(im1, im2, s, n):
    N,M = im1.shape
    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)    
    tu = np.zeros(im1.shape)
    tv = np.zeros(im1.shape)
    fx,fy,ft = conv(im1,im2)
   
    for i in range(n):
        for (y, x), n in np.ndenumerate(im1):
            if x >= 2 and x < (M-1) and y >= 2 and y < (N-1): 
                Ex = fx[y][x]
                Ey = fy[y][x]
                Et = ft[y][x]
                AU = (u[y][x-1] + u[y][x+1] + u[y-1][x] + u[y+1][x])/4
                AV = (v[y][x-1] + v[y][x+1] + v[y-1][x] + v[y+1][x])/4
                
                A = (Ex*AU + Ey*AV +Et)
                B = (1 + s*(Ex*Ex + Ey*Ey))
                tu[y][x] = AU - (Ex*s*A/B)
                tv[y][x] = AV - (Ey*s*A/B)
        
        for (y, x), n in np.ndenumerate(im1):
            if x >= 2 and x < (M-1) and y >= 2 and y < (N-1):     
                u[y][x] = tu[y][x] 
                v[y][x] = tv[y][x]
                
    return  u,v
    

def LK_flow(im1,im2):
    N,M = im1.shape
    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)

    fx = np.zeros(im1.shape)
    fy = np.zeros(im1.shape)
    ft = np.zeros(im1.shape)
   
    fx,fy,ft = conv(im1,im2)
   
    for (i,j), n in np.ndenumerate(im1):
        if j >= 2 and j < (M-1) and i >= 2 and i < (N-1): 
			
			FX = ([fx[i-1,j-1],fx[i,j-1],fx[i-1,j-1],fx[i-1,j],fx[i,j],fx[i+1,j],fx[i-1,j+1],fx[i,j+1],fx[i+1,j-1]]) #The x-component of the gradient vector
			FY = ([fy[i-1,j-1],fy[i,j-1],fy[i-1,j-1],fy[i-1,j],fy[i,j],fy[i+1,j],fy[i-1,j+1],fy[i,j+1],fy[i+1,j-1]]) #The Y-component of the gradient vector
			FT = ([ft[i-1,j-1],ft[i,j-1],ft[i-1,j-1],ft[i-1,j],ft[i,j],ft[i+1,j],ft[i-1,j+1],ft[i,j+1],ft[i+1,j-1]]) #The XY-component of the gradient vector						
			A = (FX,FY)
			A = matrix(A)
			At = array(matrix(A))
			A = array(np.matrix.transpose(A)) 			
			U1 = np.dot(At,A) 
			U2 = np.linalg.pinv(U1)
			U3 = np.dot(U2,At)
			(u[i,j],v[i,j]) = np.dot(U3,FT) 
            
    return u,v

def conv(img1,img2):
    fx = signal.convolve2d(img1,[[-0.25,0.25],[-0.25,0.25]],'same') + signal.convolve2d(img2,[[-0.25,0.25],[-0.25,0.25]],'same')

    fy = signal.convolve2d(img1,[[-0.25,-0.25],[0.25,0.25]],'same') + signal.convolve2d(img2,[[-0.25,-0.25],[0.25,0.25]],'same')

    ft = signal.convolve2d(img1,[[0.25,0.25],[0.25,0.25]],'same') + signal.convolve2d(img2,[[-0.25,-0.25],[-0.25,-0.25]],'same')
    
    return fx,fy,ft
    
u,v = optical_flow_HS(L1,L2,1,2)

u_l,v_l = LK_flow(L1,L2)
plt.figure(1)
plt.imshow( draw_flow(L1,u,v))
plt.figure(2)
plt.imshow( draw_flow(L1,u_l,v_l))