# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:07:32 2016

@author: channerduan
"""

import sys
import numpy as np
from PyQt4 import QtGui, QtCore

class Object(object):
    latestObjectIndex = 0

    def __init__(self):
        Object.latestObjectIndex += 1

        self.objectIndex = Object.latestObjectIndex


class CanvasWidget(QtGui.QWidget):
    def __init__(self):
        super(CanvasWidget, self).__init__()

        self.setMouseTracking(True)

    def paintEvent(self, event):
        stage._onShow()
        
        
class DisplayObject(object):
    def __init__(self):
        super(DisplayObject, self).__init__()

        self.parent = None
        self.x = 0
        self.y = 0
        self.alpha = 1
        self.rotation = 0
        self.scaleX = 1
        self.scaleY = 1
        self.visible = True

    @property
    def width(self):
        return self._getOriginalWidth() * abs(self.scaleX)

    @property
    def height(self):
        return self._getOriginalHeight() * abs(self.scaleY)

    def _show(self, c):
        if not self.visible:
            return

        c.save()

        c.translate(self.x, self.y)
        c.setOpacity(self.alpha * c.opacity())
        c.rotate(self.rotation)
        c.scale(self.scaleX, self.scaleY)

        self._loopDraw(c)

        c.restore()

    def _loopDraw(self, c):
        pass

    def _getOriginalWidth(self):
        return 0

    def _getOriginalHeight(self):
        return 0

    def remove(self):
        self.parent.removeChild(self)
        
class Bitmap(DisplayObject):
    def __init__(self,image,x = 0,y = 0):
        super(Bitmap, self).__init__()
        self.image = image
        self.x = x
        self.y = y
        
    def _getOriginalWidth(self):
        return self.image.shape[1]

    def _getOriginalHeight(self):
        return self.image.shape[0]

    def _loopDraw(self, c):
        bmpd = self.bitmapData

        c.drawImage(0, 0, bmpd.image, bmpd.x, bmpd.y, bmpd.width, bmpd.height)

class Stage(Object):
    def __init__(self):
        super(Stage, self).__init__()

        self.parent = "root"
        self.width = 0
        self.height = 0
        self.speed = 0
        self.app = None
        self.canvasWidget = None
        self.canvas = None
        self.timer = None
        self.childList = []
        self.backgroundColor = None

    def _setCanvas(self, speed, title, width, height):
        self.speed = speed
        self.width = width
        self.height = height

        self.canvas = QtGui.QPainter()

        self.canvasWidget = CanvasWidget()
        self.canvasWidget.setWindowTitle(title)
        self.canvasWidget.setFixedSize(width, height)
        self.canvasWidget.show()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(speed)
        self.timer.start();

        QtCore.QObject.connect(self.timer, QtCore.SIGNAL("timeout()"), self.canvasWidget, QtCore.SLOT("update()"))

    def _onShow(self):
        self.canvas.begin(self.canvasWidget)

        if self.backgroundColor is not None:
            self.canvas.fillRect(0, 0, self.width, self.height, getColor(self.backgroundColor))
        else:
            self.canvas.eraseRect(0, 0, self.width, self.height)

        self._showDisplayList(self.childList)

        self.canvas.end()

    def _showDisplayList(self, childList):
        for o in childList:
            if hasattr(o, "_show") and hasattr(o._show, "__call__"):
                o._show(self.canvas)

    def addChild(self, child):
        if child is not None:
            child.parent = self

            self.childList.append(child)
        else:
            raise ValueError("parameter 'child' must be a display object.")

    def removeChild(self, child):
        if child is not None:
            self.childList.remove(child)

            child.parent = None
        else:
            raise ValueError("parameter 'child' must be a display object.")

def init(speed, title, width, height, callback):
    stage.app = QtGui.QApplication(sys.argv)

    stage._setCanvas(speed, title, width, height)

    if not hasattr(callback, "__call__"):
        raise ValueError("parameter 'callback' must be a function.")

    callback()

    stage.app.exec_()

def getColor(color):
    if isinstance(color, QtGui.QColor):
        return color
    elif not color:
        return QtCore.Qt.transparent
    else:
        colorObj = QtGui.QColor()
        colorObj.setNamedColor(color)

        return colorObj


def exit_app():
    print 'GG'

stage = Stage()
bits = np.ones((100,100),dtype=int)*255
bitmap = Bitmap(bits)
stage.addChild(bitmap)
init(10,'Test',400,400,exit_app)     
