# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:40:01 2016

@author: channerduan
"""

import os
from PIL import Image, ImageDraw
import cv2 as cv
import numpy as np


def detect_object(image):
    '''检测图片，获取人脸在图片中的坐标'''
    gray = np.zeros(image.shape[:2], dtype=np.float16)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    classifier=cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
    rect=classifier.detectMultiScale(gray,1.2,2,cv.CASCADE_SCALE_IMAGE,(20,20))
    result = []
    for r in rect:
        result.append((r[0], r[1], r[0]+r[2], r[1]+r[3]))
    return result

def process(infile):
    '''在原图上框出头像并且截取每个头像到单独文件夹'''
    image = cv.imread(infile);
    print image.shape
#    if image:
    faces = detect_object(image)

    im = Image.open(infile)
    path = os.path.abspath(infile)
    save_path = os.path.splitext(path)[0]+"_face"
    try:
        os.mkdir(save_path)
    except:
        pass
    if faces:
        draw = ImageDraw.Draw(im)
        count = 0
        for f in faces:
            count += 1
            draw.rectangle(f, outline=(255, 0, 0))
            a = im.crop(f)
            file_name = os.path.join(save_path,str(count)+".jpg")
     #       print file_name
            a.save(file_name)

        drow_save_path = os.path.join(save_path,"out.jpg")
        im.save(drow_save_path, "JPEG", quality=80)
    else:
        print "Error: cannot detect faces on %s" % infile

if __name__ == "__main__":
    process("./opencv_in.jpg")