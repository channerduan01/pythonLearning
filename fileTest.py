# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 15:58:26 2016

@author: channerduan
"""

str="a string to print to file"
f = open('out.txt','w')
print >>f,str
f.close()
