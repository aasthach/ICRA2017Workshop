"""
***********************************************************
File: interface.py
Author: Luke Burks
Date: April 2018

Implements a PYQT5 interface that allows human aided 
target tracking through sketches, drone launches, 
and human push/robot pull semantic information

Using Model-View-Controller Architecture

Revamping for actual research

Version History (Sort of):
0.1.1: added robot movement
0.1.2: added automatic robot movement
0.2.0: added POMCP control, data collection, yaml config


***********************************************************
"""

__author__ = "Luke Burks"
__copyright__ = "Copyright 2018"
__credits__ = ["Luke Burks"]
__license__ = "GPL"
__version__ = "0.2.0"
__maintainer__ = "Luke Burks"
__email__ = "luke.burks@colorado.edu"
__status__ = "Development"


from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *; 
from PyQt5.QtGui import *;
from PyQt5.QtCore import *;
import sys,os

import numpy as np
import time
import yaml

import julia
import subprocess as sp

from matplotlib.backends.backend_qt4agg import FigureCanvas
from matplotlib.figure import Figure, SubplotParams
import matplotlib.pyplot as plt

from interfaceFunctions import *; 
from planeFunctions import *;
from problemModel import Model
from robotControllers import Controller; 
from juliaController import JuliaController


class SimulationWindow(QWidget):

	def __init__(self,cf):

		super(SimulationWindow,self).__init__()
		with open(cf,'r') as stream:
			self.params = yaml.load(stream); 

		self.setGeometry(1,1,1350,800)
		self.layout = QGridLayout(); 
		self.layout.setColumnStretch(0,2); 
		self.layout.setColumnStretch(1,2);
		#self.layout.setColumnStretch(2,.1); 
		# self.layout.setColumnStretch(3,1);
		# self.layout.setColumnStretch(4,1);
		self.setLayout(self.layout); 

		#DATA COLLECTION
		self.lastPush = []; 
		beliefType = self.params['Model']['belNum']; 
		pushing = self.params['Interface']['pushing']; 
		self.CONTROL_TYPE = self.params['Interface']['controlType']; 
		#self.SAVE_FILE = '../data/{}_bel{}_{}'.format(self.CONTROL_TYPE,beliefType,pushing,time.asctime().replace(' ','').replace(':','_')); 
		self.SAVE_FILE = None; 

		#Make Models
		self.trueModel = Model(self.params,trueModel=True);
		self.assumedModel = Model(self.params,trueModel=False); 
		self.TARGET_STATUS = 'loose'; 

		self.makeBreadCrumbColors(); 

		#Sketching Params
		self.sketchListen=False; 
		self.sketchingInProgress = False; 
		self.allSketches = {}; 
		self.allSketchNames = []; 
		self.allSketchPaths = []; 
		self.allSketchPlanes = {}; 
		self.sketchLabels = {}; 
		self.sketchDensity = 3; #radius in pixels of drawn sketch points
		self.NUM_SKETCH_POINTS = self.params['Interface']['numSketchPoints']; 

		#Drone Params
		self.droneClickListen = False; 
		self.DRONE_WAIT_TIME = self.params['Interface']['droneWaitTime']; 
		self.timeLeft = self.DRONE_WAIT_TIME; 
		self.DRONE_VIEW_RADIUS = self.params['Interface']['droneViewRadius'];

		#Controller Paramas
		self.CONTROL_FREQUENCY = self.params['Interface']['controlFreq']; #Hz
		
		if(self.CONTROL_TYPE == "MAP"):
			self.control = Controller(self.assumedModel); 
		elif(self.CONTROL_TYPE == "POMCP"):		
			self.control = sp.Popen(['python','-u','juliaBridge.py'],stdin = sp.PIPE,stdout = sp.PIPE,stderr=sp.STDOUT)


		
		self.makeMapGraphics();

		self.makeTabbedGraphics(); 

		self.populateInterface(); 
		self.connectElements(); 

		self.makeRobot();
		self.makeTarget();


		loadQuestions(self); 

		droneTimerStart(self); 

		if(self.CONTROL_TYPE != "Human"):
			controlTimerStart(self); 

		self.show()

	#Sets up a series of QColors to let the comet trail fade
	def makeBreadCrumbColors(self):
		self.breadColors = []; 
		num_crumbs = self.trueModel.BREADCRUMB_TRAIL_LENGTH; 

		for i in range(0,num_crumbs):
			alpha = 255*(i)/num_crumbs; 
			self.breadColors.append(QColor(255,0,0,alpha))


	#Listens for human control input
	def keyReleaseEvent(self,event):
		arrowEvents = [QtCore.Qt.Key_Up,QtCore.Qt.Key_Down,QtCore.Qt.Key_Left,QtCore.Qt.Key_Right]; 
		if(self.CONTROL_TYPE=='Human'):
			if(event.key() in arrowEvents):
				moveRobot(self,event.key()); 

	#Initializes robot position in layer
	def makeRobot(self):
		moveRobot(self,None); 
		 
	#Initializes target position in layer	
	def makeTarget(self):
		points = []; 
		rad = self.trueModel.TARGET_SIZE_RADIUS; 
		for i in range(-int(rad/2)+self.trueModel.robPose[0],int(rad/2)+self.trueModel.robPose[0]):
			for j in range(-int(rad/2) + self.trueModel.robPose[1],int(rad/2)+self.trueModel.robPose[1]):
				#if(i>0 and j>0 and i<self.imgHeight and j<self.imgWidth):
				tmp1 = min(self.imgWidth-1,max(0,i)); 
				tmp2 = min(self.imgHeight-1,max(0,j)); 
				points.append([tmp1,tmp2]); 
		planeAddPaint(self.truePlane,points,QColor(255,0,255,255)); 


	def makeMapGraphics(self):

		#Image View
		#************************************************************
		self.imageView = QGraphicsView(self); 
		self.imageScene = QGraphicsScene(self); 

		makeTruePlane(self); 

		makeFogPlane(self);

		#make sketchPlane
		self.sketchPlane = self.imageScene.addPixmap(makeTransparentPlane(self));


		#make robotPose plane
		self.robotPlane = self.imageScene.addPixmap(makeTransparentPlane(self));

		#make targetPose plane
		self.targetPlane = self.imageScene.addPixmap(makeTransparentPlane(self));

		#make click layer
		self.clickPlane = self.imageScene.addPixmap(makeTransparentPlane(self));

		#make comet trail layer
		self.trailLayer = self.imageScene.addPixmap(makeTransparentPlane(self)); 

		#Make goal layer
		self.goalLayer = self.imageScene.addPixmap(makeTransparentPlane(self)); 


		self.imageView.setScene(self.imageScene); 
		self.layout.addWidget(self.imageView,0,1,15,1); 





	def makeTabbedGraphics(self):
		
		self.tabs = QTabWidget(self); 

		#Belief Map
		#************************************************************
		#self.beliefMapWidget = QLabel(self); 
		self.beliefMapWidget = makeBeliefMap(self); 
		#self.beliefMapWidget.setPixmap(pm); 
		self.tabs.addTab(self.beliefMapWidget,'Belief'); 

		#Transitions Map
		#************************************************************
		self.assumedModel.transitionLayer = convertPixmapToGrayArray(self.fogPlane.pixmap()); 
		self.assumedModel.transitionLayer /= 255.0;
		self.assumedModel.transitionLayer = np.amax(self.assumedModel.transitionLayer)  - self.assumedModel.transitionLayer; 
		self.assumedModel.transitionLayer *= 10.0;
		self.assumedModel.transitionLayer -= 5.0; 

		###Saves the map to a pdf file
		# plt.figure(figsize = (4.3,7.5));  
		# plt.contourf(np.flipud(np.amax(self.assumedModel.transitionLayer)-np.transpose(self.assumedModel.transitionLayer)),cmap='seismic'); 
		# plt.axis('off'); 
		# plt.savefig('../img/assumed_Transitions.pdf',format='pdf',dpi=1000,bbox_inches='tight'); 

		self.trueModel.transitionLayer = convertPixmapToGrayArray(self.truePlane.pixmap());
		self.trueModel.transitionLayer /= 255.0;
		self.trueModel.transitionLayer = np.amax(self.trueModel.transitionLayer) - self.trueModel.transitionLayer; 
		self.trueModel.transitionLayer *= 10.0;
		self.trueModel.transitionLayer -= 5.0; 

		###Saves the map to a pdf file
		# plt.figure(figsize = (4.3,7.5)); 
		# plt.contourf(np.flipud(np.amax(self.assumedModel.transitionLayer)-np.transpose(self.trueModel.transitionLayer)),cmap='seismic'); 
		# plt.axis('off'); 
		# plt.savefig('../img/true_Transitions.pdf',format='pdf',dpi=1000,bbox_inches='tight'); 

		self.transMapWidget_true = QLabel(); 
		tm = makeModelMap(self,self.trueModel.transitionLayer); 
		self.transMapWidget_true.setPixmap(tm); 
		self.tabs.addTab(self.transMapWidget_true,'True Transitions'); 


		self.transMapWidget_assumed = QLabel(); 
		tm = makeModelMap(self,self.assumedModel.transitionLayer); 
		self.transMapWidget_assumed.setPixmap(tm); 
		self.tabs.addTab(self.transMapWidget_assumed,'Assumed Transitions'); 
		
	
		#Cost Map
		#************************************************************
		self.costMapWidget = QLabel(); 
		cm = makeModelMap(self,self.assumedModel.costLayer); 
		self.costMapWidget.setPixmap(cm); 
		self.tabs.addTab(self.costMapWidget,'Costs'); 



		self.layout.addWidget(self.tabs,0,0,15,1); 


	def populateInterface(self):

		sectionHeadingFont = QFont(); 
		sectionHeadingFont.setPointSize(20); 
		sectionHeadingFont.setBold(True); 


		#Sketching Section
		#**************************************************************
		sketchLabel = QLabel("Sketching");
		sketchLabel.setFont(sectionHeadingFont); 
		self.layout.addWidget(sketchLabel,1,2); 

		self.startSketchButton = QPushButton("Start\nSketch"); 
		self.layout.addWidget(self.startSketchButton,2,2); 

		self.sketchName = QLineEdit();
		self.sketchName.setPlaceholderText("Sketch Name");  
		self.layout.addWidget(self.sketchName,2,3,1,2); 


		sketchButtonLabel = QLabel("Sketch Parameters"); 
		self.layout.addWidget(sketchButtonLabel,3,2); 

		self.costRadioGroup = QButtonGroup(self); 
		self.targetRadio = QRadioButton("Target"); 
		self.costRadioGroup.addButton(self.targetRadio,id=1); 
		self.safeRadio = QRadioButton("Nominal"); 
		self.safeRadio.setChecked(True); 
		self.costRadioGroup.addButton(self.safeRadio,id=0); 
		self.dangerRadio = QRadioButton("Dangerous"); 
		self.costRadioGroup.addButton(self.dangerRadio,id=-1); 
		self.layout.addWidget(self.targetRadio,3,3); 
		self.layout.addWidget(self.safeRadio,4,3);
		self.layout.addWidget(self.dangerRadio,5,3);  

		self.speedRadioGroup = QButtonGroup(self); 
		self.fastRadio = QRadioButton("Fast"); 
		self.speedRadioGroup.addButton(self.fastRadio,id=1); 
		self.nomRadio = QRadioButton("Nominal"); 
		self.nomRadio.setChecked(True); 
		self.speedRadioGroup.addButton(self.nomRadio,id=0); 
		self.slowRadio = QRadioButton("Slow"); 
		self.speedRadioGroup.addButton(self.slowRadio,id=-1); 
		self.layout.addWidget(self.fastRadio,3,4);
		self.layout.addWidget(self.nomRadio,4,4); 
		self.layout.addWidget(self.slowRadio,5,4);  




		#Human Push Section
		#**************************************************************
		pushLabel = QLabel("Human Push"); 
		pushLabel.setFont(sectionHeadingFont);
		self.layout.addWidget(pushLabel,6,2); 

		self.positivityDrop = QComboBox(); 
		self.positivityDrop.addItem("Is"); 
		self.positivityDrop.addItem("Is not"); 
		self.layout.addWidget(self.positivityDrop,7,2); 

		self.relationsDrop = QComboBox();
		self.relationsDrop.addItem("Near"); 
		self.relationsDrop.addItem("North of"); 
		self.relationsDrop.addItem("South of");
		self.relationsDrop.addItem("East of");
		self.relationsDrop.addItem("West of");
		self.layout.addWidget(self.relationsDrop,7,3); 

		self.objectsDrop = QComboBox();
		self.objectsDrop.addItem("You"); 
		self.layout.addWidget(self.objectsDrop,7,4); 

		self.pushButton = QPushButton("Submit"); 
		self.pushButton.setStyleSheet("background-color: green"); 
		self.layout.addWidget(self.pushButton,7,5); 


		#Drone Launch Section
		#**************************************************************
		droneLabel = QLabel("Drone Controls"); 
		droneLabel.setFont(sectionHeadingFont);
		self.layout.addWidget(droneLabel,9,2); 

		self.updateTimerLCD = QLCDNumber(self); 
		self.updateTimerLCD.setSegmentStyle(QLCDNumber.Flat); 
		self.updateTimerLCD.setStyleSheet("background-color:rgb(255,0,0)"); 
		self.updateTimerLCD.setMaximumHeight(25);
		self.updateTimerLCD.setMinimumHeight(25);  
		self.layout.addWidget(self.updateTimerLCD,10,2); 

		self.droneButton = QPushButton("Launch\nDrone"); 
		self.layout.addWidget(self.droneButton,10,2); 


		#Robot Pull Section
		#**************************************************************
		pullLabel = QLabel("Robot Pull"); 
		pullLabel.setFont(sectionHeadingFont);
		self.layout.addWidget(pullLabel,12,2); 

		self.pullQuestion = QLineEdit("Robot Question");
		self.pullQuestion.setReadOnly(True); 
		self.pullQuestion.setAlignment(QtCore.Qt.AlignCenter); 
		self.layout.addWidget(self.pullQuestion,13,2,1,3); 

		self.yesButton = QPushButton("Yes");  
		self.yesButton.setStyleSheet("background-color: green"); 
		self.layout.addWidget(self.yesButton,14,2); 

		self.noButton = QPushButton("No");  
		self.noButton.setStyleSheet("background-color: red"); 
		self.layout.addWidget(self.noButton,14,4); 


	def connectElements(self):
		self.startSketchButton.clicked.connect(lambda:startSketch(self)); 

		self.droneButton.clicked.connect(lambda: launchDrone(self)); 

		self.yesButton.clicked.connect(lambda: getNewRobotPullQuestion(self)); 

		self.noButton.clicked.connect(lambda: getNewRobotPullQuestion(self)); 

		self.pushButton.clicked.connect(lambda: pushButtonPressed(self)); 

		self.imageScene.mousePressEvent = lambda event:imageMousePress(event,self); 
		self.imageScene.mouseMoveEvent = lambda event:imageMouseMove(event,self); 
		self.imageScene.mouseReleaseEvent = lambda event:imageMouseRelease(event,self);


		self.saveBeliefShortcut = QShortcut(QKeySequence("Ctrl+B"),self); 
		self.saveBeliefShortcut.activated.connect(self.saveBeliefs); 

		self.saveAllShortcut = QShortcut(QKeySequence("Ctrl+A"),self); 
		self.saveAllShortcut.activated.connect(self.saveAllThings);

	def saveBeliefs(self):
		np.save('../models/beliefs.npy',[self.assumedModel.belief]); 
		print("Saved Current Belief");

	def saveAllThings(self):
		print("Saving Things"); 

if __name__ == '__main__':

	configFile = 'config.yaml'; 

	app = QApplication(sys.argv); 
	ex = SimulationWindow(configFile); 
	sys.exit(app.exec_()); 


