# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 22:31:42 2016

@author: channerduan
"""

from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from timer import Timer


def readImage(path):
    '''Read image as np array, from path'''
    im = np.array(Image.open(path).convert('L'))
    return im

def marrByCv(im,size,sigma):
    im = cv.GaussianBlur(im,ksize=(size,size),sigmaX=sigma,sigmaY=sigma)
    im = cv.convertScaleAbs(cv.Laplacian(im,cv.CV_16S,ksize = size))
    return im;
    
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
    
def createLaplacianTemplate(size):
    template=np.zeros((size,size),dtype=np.float)
    c = np.floor(size/2)
    template[:] = -1;
    template[c,c] = np.int(np.power(size,2)-1);
    return template
    
def createLoGTemplate(size,sigma):
    template=np.zeros((size,size),dtype=np.float)
    c = np.floor(size/2)
    for i in range(size):
        for j in range(size):
            template[i,j] = (np.power(j-c,2)+np.power(i-c,2)-2*np.power(sigma,2))/np.power(sigma,4) * \
            np.exp( -( (np.power(j-c,2)+np.power(i-c,2))/(2*np.power(sigma,2)) ) )
    template /= np.sum(template)
    return template
    
def imageFilter(src,template):
    s = src.shape
    s1 = template.shape
    c = np.int(np.floor(s1[0]/2))
    image = np.zeros((s[0],s[1]),dtype=np.int)
    for i in range(c,s[0]-c):
        for j in range(c,s[1]-c):
            image[i][j] = np.sum(src[i-c:i+c+1,j-c:j+c+1] * template)
    return image
    
def zeroCross(src):
    s = src.shape
    image = np.zeros((s[0],s[1]),dtype=np.uint8)
    index = [(-1,0,-1,0),(0,1,-1,0),(-1,0,0,1),(0,1,0,1)]
    for i in range(1,s[0]-1):
        for j in range(1,s[1]-1):
            amounts = []
            for p in index:
                amounts.append(np.sum(src[i+p[0]:i+p[1]+1,j+p[2]:j+p[3]+1]))
            if (np.max(amounts) > 0 and np.min(amounts) < 0):
                image[i][j] = 255
    return image


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
    
def compareDifferentTypes(image):
    with Timer() as t:
        test1 = imageFilter(image,createGaussianTemplate(11,2))
        test1 = imageFilter(test1,createLaplacianTemplate(11))
        test1 = zeroCross(test1)
    print "=> Gaussian+Laplacian spent: %s s" % t.secs
    with Timer() as t:
        test2 = imageFilter(image,createLoGTemplate(11,2))
        test2 = zeroCross(test2)
    print "=> loG spent: %s s" % t.secs
    with Timer() as t:
        test3 = marrByCv(image,3,1.0)
    print "=> opencv marr spent: %s s" % t.secs
    drawList = ['original',image,'opencv marr-hildreth',test3,'gaussian+laplacian',test1,'loG',test2]
    drawFigures(drawList)     
    return
    
def compareParametersOfloG(image):
    imgList = []
    paramList = []
    for i in np.arange(1,3.0,0.2):
        for j in range(3,31,2):
            paramList.append([j,i])
    length = len(paramList)
    for i in range(length):
        imgList.append(zeroCross(imageFilter(image,createLoGTemplate(paramList[i][0],paramList[i][1]))))
        print("%.d%%\r"%(np.round(np.float(i+1)/length*100)))
    drawList = []
    for i in range(length):
        str_ = 'size=%d, sigma=%f' % (paramList[i][0],paramList[i][1])
        drawList.append(str_)
        drawList.append(imgList[i])
    drawFigures(drawList)
    return

def main(path):
    plt.gray()
    image = readImage(path)
    
#    compareDifferentTypes(image)
    compareParametersOfloG(image)
    return


inputfile='nevermore.png'
#inputfile='nature.png'
#inputfile='lenna.png'
main(inputfile)

