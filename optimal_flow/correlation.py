# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 09:29:16 2016

@author: channerduan
"""

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from numba import double
from numba.decorators import jit, autojit

def readImage(path):
    im = np.array(Image.open(path).convert('L'))
    return im.astype(np.uint8),np.array(Image.open(path))

def createGaussianTemplate(size,sigma):
    template=np.zeros((size,size),dtype=np.float)
    c = np.floor(size/2)
    amount = 0
    for i in range(size):
        for j in range(size):
            template[j,i] = np.exp( -( (np.power(j-c,2)+np.power(i-c,2))/(2*np.power(sigma,2)) ) )
            amount += template[j,i]
    template = template/amount
    return template
    
def normalize(matrix):
    mean_ = np.mean(matrix)
    max_ = np.max(matrix)
    min_ = np.min(matrix)
    return (matrix-mean_)/(max_-min_)   
    
def expandMatrix(matrix,size):
    result = np.zeros((size[0],size[1]),dtype=matrix.dtype)
    cx = np.float(matrix.shape[0]/2)
    cy = np.float(matrix.shape[1]/2)
    ccx = np.float(result.shape[0]/2)
    ccy = np.float(result.shape[1]/2)    
    result[ccx-cx:ccx+cx+1,ccy-cy:ccy+cy+1] = matrix
    return result
    
def imageFourierFilter(src,template):
    res = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(src)*np.fft.fft2(expandMatrix(template,src.shape))))
    res = np.real(res)
#    res = normalize(res)
#    res = res*255
#    res = np.round(res)
    return res.astype(np.int)

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

def draw_flow(img,N,V,step=6):
    de_color = [0, 255, 125, 255]
    h, w = N.shape
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx = N[y,x]
    fy = V[y,x]
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
#    cv2.polylines(img, lines, 0, de_color,thickness=1)
    for (x1, y1), (x2, y2) in lines:
        if x1 != x2 or y1 != y2:
            cv2.line(img, (x1, y1), (x2, y2),de_color, 1)
#            cv2.circle(img, (x1, y1), 1, de_color, -1)
    return img
    
def createSearchIndex(distance):
    list_ = [0]
    for i in range(1,distance):
        list_.append(i)
        list_.append(-i)
    return list_

def flowCorrSAD(image1,image2,d,w):
    '''correlation optical flow, max displacement = d, window = 2*w+1, '''
    if image1.shape != image2.shape:
        raise Exception('illegal input') 
    s = image1.shape
    N = np.zeros(s,dtype=np.double)
    V = np.zeros(s,dtype=np.double)  
    margin = d+w
    # search start at center
    displacement = createSearchIndex(d)
    for x in range(margin,s[0]-margin):
            for y in range(margin,s[1]-margin):
                minIntensities = 999999
                n = v = 0
                for dx in displacement:
                    for dy in displacement:
                        intensities = np.sum( np.abs( \
                            image2[x+dx-w:x+dx+w+1,y+dy-w:y+dy+w+1]-image1[x-w:x+w+1,y-w:y+w+1] \
                            ) )
#                        if x == 25 and y == 40:
#                            print "%d,%d,%d,%d,%f,%f" %(x,y,x+dx,y+dy,intensities,minIntensities)                      
                        if (intensities < minIntensities):
                            minIntensities = intensities
                            n = dy
                            v = dx
                N[x,y] = n
                V[x,y] = v
    return N,V
    
def flowCorrSSD(image1,image2,d,w):
    '''correlation optical flow, max displacement = d, window = 2*w+1, '''
    if image1.shape != image2.shape:
        raise Exception('illegal input') 
    s = image1.shape
    N = np.zeros(s,dtype=np.double)
    V = np.zeros(s,dtype=np.double)  
    margin = d+w
    # search start at center
    displacement = createSearchIndex(d)
    for x in range(margin,s[0]-margin):
            for y in range(margin,s[1]-margin):
                minIntensities = 999999
                n = v = 0
                for dx in displacement:
                    for dy in displacement:
                        intensities = np.sum( ( \
                            image2[x+dx-w:x+dx+w+1,y+dy-w:y+dy+w+1]-image1[x-w:x+w+1,y-w:y+w+1] \
                            )**2 )
                        if (intensities < minIntensities):
                            minIntensities = intensities
                            n = dy
                            v = dx
                N[x,y] = n
                V[x,y] = v
    return N,V    
    
def flowCorrCC(image1,image2,d,w):
    '''correlation optical flow, max displacement = d, window = 2*w+1, '''
    if image1.shape != image2.shape:
        raise Exception('illegal input') 
    s = image1.shape
    N = np.zeros(s,dtype=np.double)
    V = np.zeros(s,dtype=np.double)  
    margin = d+w
    # search start at center
    displacement = createSearchIndex(d)
    for x in range(margin,s[0]-margin):
            for y in range(margin,s[1]-margin):
                maxIntensities = -999999
                n = v = 0
                for dx in displacement:
                    for dy in displacement:
                        intensities = np.sum(
                            image2[x+dx-w:x+dx+w+1,y+dy-w:y+dy+w+1]*image1[x-w:x+w+1,y-w:y+w+1] \
                            )
                        if (intensities > maxIntensities):
                            maxIntensities = intensities
                            n = dy
                            v = dx
                N[x,y] = n
                V[x,y] = v
    return N,V      
    
    


def drawQuiver(N,V,gap=1,figsize=5,scale_=1,title=''):
    plt.figure(figsize=(figsize, figsize))
    X, Y = np.meshgrid(range(N.shape[0]), np.arange(N.shape[1]-1,-1,-1))
    
    Q = plt.quiver(X[::gap, ::gap], Y[::gap, ::gap], N[::gap, ::gap], V[::gap, ::gap],
               color='b', units='x',scale=scale_,
               linewidths=(0,), edgecolors=('k'), headaxislength=3)    
    plt.axis('off')
#    plt.axis('equal')
#    plt.axis([-10,200,-10,300])
    plt.title(title)
    return Q

def simpleDemo():
    im1,image1 = readImage('ta.png')
    im2,image2 = readImage('tb.png')
    N,V = flowCorrSAD_numba(255-im1,255-im2,4,0)
    drawQuiver(N,-V)
    return


def differentMethodCompare(im1,im2,d,w):
    N1,V1 = flowCorrSAD_numba(im1,im2,d,w)
    N2,V2 = flowCorrSAD_numba(im1-np.mean(im1),im2-np.mean(im2),d,w)  
    N3,V3 = flowCorrSSD_numba(im1,im2,d,w)
    N4,V4 = flowCorrSSD_numba(im1-np.mean(im1),im2-np.mean(im2),d,w)   
    
    drawFigures(['SAD',np.sqrt(N1**2+V1**2),'ZSAD',np.sqrt(N1**2+V1**2),\
        'SSD',np.sqrt(N1**2+V1**2),'ZSSD',np.sqrt(N1**2+V1**2)])    
    return


plt.gray()
flowCorrSAD_numba = autojit(flowCorrSAD)
flowCorrSSD_numba = autojit(flowCorrSSD)

flowCorrCC_numba = autojit(flowCorrCC)

#simpleDemo()

im1,image1 = readImage('ffa.png')
im2,image2 = readImage('ffb.png')

#im1,image1 = readImage('s1.png')
#im2,image2 = readImage('s2.png')

sigma = 1.5
im1_ = imageFourierFilter(im1,createGaussianTemplate(11,sigma))
im2_ = imageFourierFilter(im2,createGaussianTemplate(11,sigma))

#differentMethodCompare(im1_,im2_,5,2)
#N,V = flowCorrCC_numba(im1_,im2_,5,2)
#N,V = flowCorrCC_numba(im1_-np.mean(im1_),im2_-np.mean(im2_),5,2)


# dense flow
#flow = cv2.calcOpticalFlowFarneback(im1, im2, 0.5, 3, 15, 3, 5, 1.2, 0)
#N = flow[:,:,0]
#V = flow[:,:,1]

#drawFigures(['image1',image1,'image2',image2,'image1-2',im1-im2,'image2-1',im2-im1,'N',np.abs(N),'V',np.abs(V),'Magnitude',np.sqrt(N**2+V**2)])
##
#plt.figure(figsize=(7, 7))
#plt.axis('off')
#plt.imshow(draw_flow(image1.copy(),N,V))
#drawQuiver(N,-V,5,10,0.7)
    
