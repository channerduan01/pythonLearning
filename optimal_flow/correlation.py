# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 09:29:16 2016

@author: channerduan
"""

from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from numba import double
from numba.decorators import jit, autojit

def readImage(path):
    im = np.array(Image.open(path).convert('L'))
    return im.astype(np.int)
    
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
    
def createSearchIndex(distance):
    list_ = [0]
    for i in range(1,distance):
        list_.append(i)
        list_.append(-i)
    return list_

def flowCorr(image1,image2,d,w):
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
                        intensities = np.sqrt( np.sum( np.power( \
                            image2[x+dx-w:x+dx+w+1,y+dy-w:y+dy+w+1]-image1[x-w:x+w+1,y-w:y+w+1], \
                            2)))
#                        if image1[x,y] == 255:
#                            print "%d,%d,%d,%d,%f,%f" %(dx,dy,x+dx,y+dy,intensities,minIntensities)
                        if (intensities < minIntensities):
                            minIntensities = intensities
                            n = dy
                            v = -dx
                N[x,y] = n
                V[x,y] = v
    return N,V


def drawQuiver(N,V,gap=1,figsize=5,scale_=1,title=''):
    plt.figure(figsize=(figsize, figsize))
    X, Y = np.meshgrid(range(N.shape[0]), np.arange(N.shape[1]-1,-1,-1))
    
    Q = plt.quiver(X[::gap, ::gap], Y[::gap, ::gap], N[::gap, ::gap], V[::gap, ::gap],
               color='b', units='x',scale=scale_,
               linewidths=(0,), edgecolors=('k'), headaxislength=5)    
    plt.axis('off')
#    plt.axis('equal')
#    plt.axis([-10,200,-10,300])
    plt.title(title)
    return Q


def simpleDemo():
    image1 = 255-readImage('ta.png')
    image2 = 255-readImage('tb.png')
    N,V = flowCorr_numba(image1,image2,4,0)
    drawQuiver(N,V,)
    return
    
plt.gray()
flowCorr_numba = autojit(flowCorr)

#simpleDemo()

image1 = 255-readImage('pa.png')
image2 = 255-readImage('pc.png')


#
#
N,V = flowCorr_numba(image1,image2,5,0)
drawFigures(['image1',image1,'image2',image2,'image1-2',image1-image2,'image2-1',image2-image1,'N',np.abs(N),'V',np.abs(V),'Magnitude',np.sqrt(N**2+V**2)])

#N = np.zeros(V.shape,dtype=float)
drawQuiver(N,V,4,7,1.2)