# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 09:29:16 2016

@author: channerduan
"""
import cv2
import numpy as np


def flowCorrZNCC_sparse(image1,image2,d,w,p0):
    '''correlation optical flow, max displacement = d, window = 2*w+1, interest points = p'''
    if image1.shape != image2.shape:
        raise Exception('illegal input') 
    s = image1.shape
    N = np.zeros(s,dtype=np.double)
    V = np.zeros(s,dtype=np.double)
    
    
    for row in p0:
        tu = tuple(row[0])
        print tu
        maxIntensities = -9999999
        
    
    
    margin = d+w
    # search start at center
    displacement = createSearchIndex(d)
    for x in range(margin,s[0]-margin):
            for y in range(margin,s[1]-margin):
                maxIntensities = -9999999
                n = v = 0
                tmpMean1 = np.mean(image1[x-w:x+w+1,y-w:y+w+1])
                deno1 = np.sum((image1[x-w:x+w+1,y-w:y+w+1]-tmpMean1)**2)
                for dx in displacement:
                    for dy in displacement:
                        tmpMean2 = np.mean(image2[x+dx-w:x+dx+w+1,y+dy-w:y+dy+w+1])
                        deno = deno1 * np.sum((image2[x+dx-w:x+dx+w+1,y+dy-w:y+dy+w+1]-tmpMean2)**2)
                        if deno != 0:
                            intensities = np.sum(
                                (image2[x+dx-w:x+dx+w+1,y+dy-w:y+dy+w+1]-tmpMean2) \
                                * (image1[x-w:x+w+1,y-w:y+w+1]-tmpMean1) \
                                /(np.sqrt(deno))
                                )
                        else:
                            intensities = 0
                        if (intensities > maxIntensities):
                            maxIntensities = intensities
                            n = dy
                            v = dx
                N[x,y] = n
                V[x,y] = v
    return N,V

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
de_color = [255, 255, 255]

p_list = []
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global p_list
    if event==cv2.EVENT_LBUTTONUP:
        print '%f,%f' %(x,y)
        p_list.append((x,y))

cv2.namedWindow("test")
cv2.setMouseCallback('test',draw_circle)
cap = cv2.VideoCapture(0)

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_gray)
p0 = None
#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

while True:
    if len(p_list) != 0:
        x = p_list[0][0]
        y = p_list[0][1]
        if p0 == None:
            p0 = np.array([[[x,y]]], dtype=np.float32)
        else:
            p0 = np.concatenate((p0,np.array([[[x,y]]], dtype=np.float32)),0)
        p_list = []
    suc,frame=cap.read()
    size=frame.shape[:2]
    if suc != True:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if p0 != None:
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # draw the tracks
        for i,(new,old) in enumerate(zip(p1,p0)):
            a,b = new.ravel()
            c,d = old.ravel()
            cv2.line(mask, (a,b),(c,d),de_color, 2)
            cv2.circle(mask,(a,b),4,de_color,-1)
    img = cv2.add(frame_gray,mask)
    cv2.imshow("test",img)
    key=cv2.waitKey(100)
    c=chr(key&255)
    if c in ['q']:
        break
    if c in ['c']:
        mask = np.zeros_like(frame_gray)
    if c in ['a']:
        p0 = p1 = None
    
    old_gray = frame_gray.copy()
    if p0 != None:
        p0 = p1.reshape(-1,1,2)

cap = None
cv2.destroyWindow("test")
