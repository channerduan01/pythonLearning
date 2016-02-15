# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:15:24 2016

@author: channerduan
"""

from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext


if __name__ == "__main__":
#    if len(sys.argv) != 2:
#        print("Usage: wordcount <file>", file=sys.stderr)
#        exit(-1)
    a = 'main.py'
    sc = SparkContext(appName="PythonWordCount")
#    lines = sc.textFile(sys.argv[1], 1)
    lines = sc.textFile(a)
    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add)
    output = counts.collect()
    for (word, count) in output:
        print("%s: %i" % (word, count))

    sc.stop()