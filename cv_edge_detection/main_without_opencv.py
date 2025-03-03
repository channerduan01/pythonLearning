# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 22:31:42 2016

@author: channerduan
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from timer import Timer
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter  


def readImage(path):
    '''Read image as np array, from path'''
    im = np.array(Image.open(path).convert('L'))
    return im

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
    
def createLaplacianTemplate(size,complex_=True):
    template=np.zeros((size,size),dtype=np.float)
    c = np.floor(size/2)
    if complex_:
        template[:] = -1
        template[c,c] = np.int(np.power(size,2)-1)
    else:
        template[:,c] = -1
        template[c,:] = -1
        template[c,c] = 2*c-1
    return template
    
def createLoGTemplate(size,sigma):
    template=np.zeros((size,size),dtype=np.float)
    c = np.floor(size/2)
    for i in range(size):
        for j in range(size):
            template[i,j] = (np.power(j-c,2)+np.power(i-c,2)-2*np.power(sigma,2))/np.power(sigma,4) * \
            np.exp( -( (np.power(j-c,2)+np.power(i-c,2))/(2*np.power(sigma,2)) ) )
    template /= np.sum(template)
    return normalize(template)
    
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
    sigma = 2
    size = 11
    with Timer() as t:
        test1 = imageFilter(image,createGaussianTemplate(size,sigma))
        test1 = imageFilter(test1,createLaplacianTemplate(size,True))
    print "=> Gaussian+Laplacian spent: %s s" % t.secs
    test1 = zeroCross(test1)
    with Timer() as t:
        test2 = imageFilter(image,createLoGTemplate(size,sigma))
    print "=> LoG spent: %s s" % t.secs
    test2 = zeroCross(test2)
    with Timer() as t:
        test4 = imageFourierFilter(image,createLoGTemplate(size,sigma))
    print "=> Fourier LoG spent: %s s" % t.secs
    test4 = zeroCross(test4)
    drawList = ['original',image,'gaussian+laplacian',test1,'LoG',test2,'Fourier LoG',test4]
    drawFigures(drawList)
    return
    
def compareParametersOfloG(image):
    imgList = []
    ringList = []
    paramList = []
    for i in range(3,22,2):
        for j in np.arange(0.2,3.0,0.2):
            paramList.append([i,j])
#    paramList = [(101,2.8)]
    length = len(paramList)
    for i in range(length):
        template = createLoGTemplate(paramList[i][0],paramList[i][1])
        ringList.append(np.abs(np.fft.fftshift(np.fft.fft2(expandMatrix(template,image.shape)))))
        imgList.append(zeroCross(imageFourierFilter(image,template)))
        print("%.d%%\r"%(np.round(np.float(i+1)/length*100)))
    drawList = []
    for i in range(length):
        str_ = 'size=%d, sigma=%f' % (paramList[i][0],paramList[i][1])
        drawList.append(str_)
        drawList.append(ringList[i])
        drawList.append(str_)
        drawList.append(imgList[i])
    drawFigures(drawList)
    return
    
def extractTheCentreOfMatrix(matrix,size=1):
    size = np.floor(size/2);
    s = matrix.shape;
    cx = np.floor(s[0]/2)
    cy = np.floor(s[1]/2)
    return matrix[cx-size:cx+size+1,cy-size:cy+size+1]
    
def deleteTheCentreOfMatrix(matrix,size=1):
    size = np.floor(size/2);
    s = matrix.shape;
    cx = np.floor(s[0]/2)
    cy = np.floor(s[1]/2)
    matrix[cx-size:cx+size+1,cy-size:cy+size+1] = 0
    return matrix

# tricky function... should add more condition check in future...
def expandMatrix(matrix,size):
    result = np.zeros((size[0],size[1]),dtype=matrix.dtype)
    cx = np.float(matrix.shape[0]/2)
    cy = np.float(matrix.shape[1]/2)
    ccx = np.float(result.shape[0]/2)
    ccy = np.float(result.shape[1]/2)    
    result[ccx-cx:ccx+cx+1,ccy-cy:ccy+cy+1] = matrix
    return result

def createFourierAnalysisForImage(image,size,avoidCentre=True):
    fourier = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    # the centre is often too large, so remove it for better visualization
    if avoidCentre:
        fourier[np.floor(fourier.shape[0]/2), np.floor(fourier.shape[1]/2)] = 0
    min_ = np.min(fourier)
    max_ = np.max(fourier)
    # map to [0,1]
    fourier = (fourier-min_)/(max_-min_)
    # extract center pixels for better visualization
    fourier = extractTheCentreOfMatrix(fourier,size)
    fourier = np.round(fourier * 255)
    # replenish the centre
    if avoidCentre:
        fourier[np.floor(fourier.shape[0]/2), np.floor(fourier.shape[1]/2)] = 255
    return fourier.astype(np.uint8)

def lowpassFourierFilter(image,width):
    res = np.fft.fftshift(np.fft.fft2(image))
    res = extractTheCentreOfMatrix(res,width)
    res = expandMatrix(res,image.shape)
    res = np.fft.ifft2(np.fft.ifftshift(res))
    return np.abs(res)

def highpassFourierFilter(image,width):
    res = np.fft.fftshift(np.fft.fft2(image))
    res = deleteTheCentreOfMatrix(res,width)
    res = np.fft.ifft2(np.fft.ifftshift(res))
    return np.abs(res)   

def draw3D(Z,title='',vx=0,vy=0):
    X = range(Z.shape[0])    
    Y = range(Z.shape[1])   
    X, Y = np.meshgrid(X,Y)  
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1,projection='3d')
    ax.axis('off')
    ax.set_title(title)
    if (vx != 0 and vy != 0):
        ax.view_init(vx, vy)
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0,antialiased=False)     
    return

def show3DforloG():
    Z = createLoGTemplate(51,5)
    draw3D(Z,'LoG in time domain')
    Z = np.abs(np.fft.fftshift(np.fft.fft2(expandMatrix(createLoGTemplate(11,2),(101,101)))))
    draw3D(Z,'LoG in frequency domain',85,30)
    drawFigures(['LoG in frequency domain',Z])
    return
    
def mapTo255(matrix):
    max_ = np.max(matrix)
    min_ = np.min(matrix)
    matrix = (matrix-min_)/(max_-min_)
    matrix *= 255
    matrix = np.round(matrix)
    return matrix.astype(np.uint8)

def normalize(matrix):
    mean_ = np.mean(matrix)
#    std_ = np.std(matrix)
    max_ = np.max(matrix)
    min_ = np.min(matrix)
#    return (matrix-mean_)/std_
    return (matrix-mean_)/(max_-min_)

def demoLowHighpassFilter(image):
    plt.figure()
    plt.axis('off')
    plt.imshow(lowpassFourierFilter(image,21))
    plt.figure()
    plt.axis('off')
    plt.imshow(highpassFourierFilter(image,21))
    return
    
def imageFourierFilter(src,template):
    res = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(src)*np.fft.fft2(expandMatrix(template,src.shape))))
    res = np.real(res)
    res = normalize(res)
    res = res*255
    res = np.round(res)
    return res.astype(np.int)

def demoProcessOfFourier(src,template):
    list_ = []
    list_.append('original')
    list_.append(src)
    p1 = np.fft.fftshift(np.fft.fft2(src))
    list_.append('original fourior')
    list_.append(np.abs(p1))
    list_.append('template')
    list_.append(template)
    p2 = np.fft.fftshift(np.fft.fft2(expandMatrix(template,src.shape)))
    list_.append('template fourior')
    list_.append(np.abs(p2))
    p3 = p1*p2
    list_.append('multiply fourior')
    list_.append(np.abs(p3))
    p4 = np.fft.fftshift(np.fft.ifft2(p3))
    list_.append('inverse fourior')
    list_.append(np.abs(p4))
    
    p5 = imageFourierFilter(src,template)
    list_.append('normalize')
    list_.append(p5)
    p6 = zeroCross(p5)
    list_.append('zero cross')
    list_.append(p6)
    drawFigures(list_)
    return

def demoImageFourierForm(image,zoomSize,avoidCentre=True):
    plt.figure()
    plt.axis('off')
    plt.imshow(createFourierAnalysisForImage(image,zoomSize,avoidCentre))
    return
    
def demoSimpleButtonProblem():
    sigma = 1.5;
    size = 17;
    test = expandMatrix(np.ones((15,15),dtype=float),(71,71))*255
#    a1 = zeroCross(imageFilter(imageFilter(test,createGaussianTemplate(size,sigma)),createLaplacianTemplate(size)))
#    a2 = zeroCross(imageFilter(test,createLoGTemplate(size,sigma)))
#    a3 = zeroCross(imageFourierFilter(test,createLoGTemplate(size,sigma)))
#    drawFigures(['Gaussian+Laplacian',a1,'LoG',a2,'Fourier LoG',a3])
    demoProcessOfFourier(test,createLoGTemplate(size,sigma))
    return

inputfile='nevermore.png'

plt.gray()
image = readImage(inputfile)

print 'comparison'
compareDifferentTypes(image)
#compareParametersOfloG(image)
print 'shape of template of LoG in frequency domain'
show3DforloG()
#demoImageFourierForm(image,71,True)
#demoLowHighpassFilter(image)
print 'process of fourier for LoG'
demoProcessOfFourier(image,createLoGTemplate(11,2))
demoSimpleButtonProblem()

