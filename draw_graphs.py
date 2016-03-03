# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:50:12 2016

@author: channerduan
"""


from PIL import Image
import numpy as np
import scipy as sc
import math
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}


plt.figure()
step = 0.05
x = np.arange(-0.25,1.25+step,step)
x = x*math.pi
y = -np.sin(x)
#plt.title('sin')
#plt.axis('off')
axis_font = {'fontname':'Arial', 'size':'20'}
plt.plot(x,y,lw=2)
plt.xlabel('space x',axis_font)
#plt.ylabel('Temperature',axis_font)
plt.ylabel('Curvature',axis_font)

plt.plot(x,np.zeros(x.shape),'.r',lw=3)