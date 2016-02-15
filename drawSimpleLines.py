# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 11:33:20 2016

@author: channerduan
"""
import numpy as np
import matplotlib.pyplot as plt
import math

x = np.array([ 2, 4, 8, 10, 12, 14, 16])
y = np.array([ 5, 10, 15, 14, 17, 20, 20])
angles = np.array([45,275,190,100,280,18,45]) 
length = np.array([ 3, 4, 8, 5, 2, 7, 1])

def draw_line(x,y,angle,length):
  cartesianAngleRadians = (450-angle)*math.pi/180.0
  terminus_x = x + length * math.cos(cartesianAngleRadians)
  terminus_y = y + length * math.sin(cartesianAngleRadians)
  plt.plot([x, terminus_x],[y,terminus_y])
  print [x, terminus_x],[y,terminus_y]

plt.axis('equal')
plt.axis([0,20,0,30])
for i in range(0,len(x)):
    print x[i],y[i],angles[i]
    draw_line(x[i],y[i],angles[i],length[i])
plt.show()