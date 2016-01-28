# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 22:31:42 2016

@author: channerduan
"""
from timer import Timer

with Timer() as t:
    for i in range(1,10000000):
        a = 1
print "=> time: %s s" % t.secs
