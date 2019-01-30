"""
***********************************************************
File: planeFunctions.py
Author: Luke Burks
Date: April 2018

Provides secondary accessible functions for the backend of 
interface.py

***********************************************************
"""


__author__ = "Luke Burks"
__copyright__ = "Copyright 2018"
__credits__ = ["Luke Burks"]
__license__ = "GPL"
__version__ = "0.1.2"
__maintainer__ = "Luke Burks"
__email__ = "luke.burks@colorado.edu"
__status__ = "Development"

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *; 
from PyQt5.QtGui import *;
from PyQt5.QtCore import *;
from copy import deepcopy

def makeTruePlane(wind):

	wind.trueImage = QPixmap('../img/eastCampus_2017_2.jpg'); 
	wind.imgWidth = wind.trueImage.size().width(); 
	wind.imgHeight = wind.trueImage.size().height(); 

	wind.truePlane = wind.imageScene.addPixmap(wind.trueImage); 


def makeFogPlane(wind):
	fI = QPixmap('../img/eastCampus_1999_2.jpg')
	wind.fogImage = QImage(wind.imgWidth,wind.imgHeight,QtGui.QImage.Format_ARGB32);
	paintMask = QPainter(wind.fogImage);  
	paintMask.drawPixmap(0,0,fI);
	paintMask.end();

	wind.originalFog = fI.toImage(); 
	wind.fogPlane = wind.imageScene.addPixmap(QPixmap.fromImage(wind.fogImage)); 




def makeTransparentPlane(wind):
	
	testMap = QPixmap(wind.imgWidth,wind.imgHeight); 
	testMap.fill(QtCore.Qt.transparent); 
	return testMap; 


def defog(wind,points):

	for p in points:
		wind.fogImage.setPixelColor(p[0],p[1],QColor(0,0,0,0)); 

	wind.fogPlane.setPixmap(QPixmap.fromImage(wind.fogImage));


def refog(wind,points):

	for p in points:
		c = wind.originalFog.pixel(p[0],p[1]); 
		col = QColor(c)
		wind.fogImage.setPixelColor(p[0],p[1],col); 
	wind.fogPlane.setPixmap(QPixmap.fromImage(wind.fogImage));

def planeAddPaint(planeWidget,points=[],col=None,pen=None):

	pm = planeWidget.pixmap(); 

	painter = QPainter(pm); 
	if(pen is None):
		if(col is None):
			pen = QPen(QColor(0,0,0,255)); 
		else:
			pen = QPen(col); 
	painter.setPen(pen)
	
	for p in points:
		painter.drawPoint(p[0],p[1]); 
	painter.end(); 
	planeWidget.setPixmap(pm); 

def planeFlushPaint(planeWidget,points=[],col = None,pen=None):
	pm = planeWidget.pixmap(); 
	pm.fill(QColor(0,0,0,0)); 

	painter = QPainter(pm); 
	if(pen is None):
		if(col is None):
			pen = QPen(QColor(0,0,0,255)); 
		else:
			pen = QPen(col); 
	painter.setPen(pen)
	
	for p in points:
		painter.drawPoint(p[0],p[1]); 
	painter.end(); 
	planeWidget.setPixmap(pm); 


def planeFlushColors(planeWidget,points=[],cols=[]):
	pm = planeWidget.pixmap(); 
	pm.fill(QColor(0,0,0,0)); 

	p = points[::-1]; 

	painter = QPainter(pm); 
	for i in range(0,len(p)):
		pen = QPen(cols[i]); 
		pen.setWidth(5); 
		painter.setPen(pen); 
		painter.drawPoint(p[i][0],p[i][1]); 
	painter.end(); 
	planeWidget.setPixmap(pm); 



def paintPixToPix(planeWidget,newPM,opacity):
	pm = planeWidget.pixmap(); 
	pm.fill(QColor(0,0,0,0)); 

	painter = QPainter(pm); 
	# for i in range(0,pm.width()):
	# 	for j in range(0,pm.height()):
	# 		#pen = QPen(QColor(im.pixel(i,j))); 
	# 		col = QColor(im.pixel(i,j)).getRgb(); 
	# 		pen = QPen(QColor(col[0],col[1],col[2],100))
	# 		pen.setWidth(1); 
	# 		painter.setPen(pen); 
	# 		painter.drawPoint(i,j); 
	painter.setOpacity(opacity); 
	painter.drawPixmap(0,0,newPM); 
	painter.end(); 
	planeWidget.setPixmap(pm); 