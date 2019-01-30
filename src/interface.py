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
sys.path.append(''); 

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
from weightedGreedyController import WGController


class SimulationWindow(QWidget):

	def __init__(self,cf):

		super(SimulationWindow,self).__init__()
		with open(cf,'r') as stream:
			self.params = yaml.load(stream); 

		#self.setGeometry(1,1,1350,800)
		self.setGeometry(-10,-10,684,812)
		self.setFixedSize(684,812); 
		self.setStyleSheet("background-color:slategray;")
		self.layout = QGridLayout(); 
		#self.layout.setColumnStretch(0,2); 
		#self.layout.setColumnStretch(0,2);
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
		self.sketchLines = {}; 
		self.sketchDensity = self.params['Interface']['sketchDensity'];
		self.NUM_SKETCH_POINTS = self.params['Interface']['numSketchPoints']; 
		self.REFOGGING = self.params['Interface']['refogging']; 
		self.CHEAT_TARGET = self.params['Interface']['cheatTarget']; 

		#Drone Params
		self.droneClickListen = False; 
		self.DRONE_WAIT_TIME = self.params['Interface']['droneWaitTime']; 
		self.timeLeft = self.DRONE_WAIT_TIME; 
		self.DRONE_VIEW_RADIUS = self.params['Interface']['droneViewRadius'];


		#Controller Paramas
		self.CONTROL_FREQUENCY = self.params['Interface']['controlFreq']; #Hz
		self.QUESTION_FREQUENCY = self.params['Interface']['questionFreq']; #Hz
		
		if(self.CONTROL_TYPE == "MAP"):
			self.control = Controller(self.assumedModel); 
		elif(self.CONTROL_TYPE == 'WMAP'):
			self.control = WGController(self.assumedModel); 
		elif(self.CONTROL_TYPE == "POMCP"):		
			#self.control = sp.Popen(['python','-u','juliaBridge.py'],stdin = sp.PIPE,stdout = sp.PIPE,stderr=sp.STDOUT)
			self.control = JuliaController(self.assumedModel); 

		self.prevFogPoints = []; 

		#Robot Pull Params
		self.questionIndex = -1;

		
		self.makeMapGraphics();

		#self.makeTabbedGraphics(); 

		self.populateInterface(); 
		self.connectElements(); 

		self.makeRobots();
		#self.makeTarget();

		makeBeliefMap(self); 

		loadQuestions(self); 

		droneTimerStart(self); 

		if(self.CONTROL_TYPE != "Human"):
			controlTimerStart(self); 

		questionTimerStart(self); 

		self.show()

	#Sets up a series of QColors to let the comet trail fade
	def makeBreadCrumbColors(self):
		self.breadColors = []; 
		num_crumbs = self.trueModel.BREADCRUMB_TRAIL_LENGTH; 

		for i in range(0,num_crumbs):
			alpha = 255*(num_crumbs-i)/num_crumbs; 
			self.breadColors.append(QColor(255,0,0,alpha))


	#Listens for human control input
	def keyReleaseEvent(self,event):
		arrowEvents = [QtCore.Qt.Key_Up,QtCore.Qt.Key_Down,QtCore.Qt.Key_Left,QtCore.Qt.Key_Right]; 
		if(self.CONTROL_TYPE=='Human'):
			if(event.key() in arrowEvents):
				moveRobot(self,event.key()); 

	#Initializes robot position in layer
	def makeRobots(self):
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

		#make targetPose plane
		self.targetPlane = self.imageScene.addPixmap(makeTransparentPlane(self));

		makeFogPlane(self);

		#make sketchPlane
		self.sketchPlane = self.imageScene.addPixmap(makeTransparentPlane(self));


		#make robotPose plane
		self.robotPlane = self.imageScene.addPixmap(makeTransparentPlane(self));

		#make click layer
		self.clickPlane = self.imageScene.addPixmap(makeTransparentPlane(self));

		#make comet trail layer
		self.trailLayer = self.imageScene.addPixmap(makeTransparentPlane(self)); 

		#Make goal layer
		self.goalLayer = self.imageScene.addPixmap(makeTransparentPlane(self)); 

		self.beliefLayer = self.imageScene.addPixmap(makeTransparentPlane(self)); 
		

		self.imageView.setScene(self.imageScene); 


		self.layout.addWidget(self.imageView,0,0,15,1); 



	def populateInterface(self):

		sectionHeadingFont = QFont(); 
		sectionHeadingFont.setPointSize(20); 
		sectionHeadingFont.setBold(True); 


		#Sketching Section
		#**************************************************************
		sketchLabel = QLabel("Sketching");
		sketchLabel.setFont(sectionHeadingFont); 
		sketchLabel.setAlignment(Qt.AlignBottom | Qt.AlignHCenter); 
		sketchLayout = QGridLayout(); 
		sketchFrame = QFrame(); 
		sketchFrame.setFrameStyle(QFrame.Panel)
		sketchFrame.setFrameShadow(QFrame.Raised)
		#sketchFrame.setSizePolicy(QSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum))


		sketchFrame.setStyleSheet('border: 3px solid black')
		#sketchLayout.addWidget(sketchFrame,1,0,2,6); 
		sketchLayout.addWidget(sketchLabel,0,0); 
		#sketchFrame.resize(300,300)

		self.startSketchButton = QPushButton("Start\nSketch"); 
		sketchLayout.addWidget(self.startSketchButton,2,0); 

		self.sketchName = QLineEdit();
		self.sketchName.setPlaceholderText("Sketch Name");  
		sketchLayout.addWidget(self.sketchName,1,0,1,6); 

		self.layout.addLayout(sketchLayout,1,1,2,3);




		#Human Push Section
		#**************************************************************
		pushLabel = QLabel("Human Push"); 
		pushLabel.setFont(sectionHeadingFont);
		pushLayout = QGridLayout(); 
		pushLabel.setAlignment(Qt.AlignBottom | Qt.AlignHCenter); 
		pushFrame = QFrame(); 
		pushFrame.setStyleSheet('border: 1px solid black')
		#pushLayout.addWidget(pushFrame,1,0,2,3); 
		pushLayout.addWidget(pushLabel,0,0,1,2); 

		self.positivityDrop = QComboBox(); 
		self.positivityDrop.addItem("Is"); 
		self.positivityDrop.addItem("Is not"); 
		pushLayout.addWidget(self.positivityDrop,1,0); 

		self.relationsDrop = QComboBox();
		self.relationsDrop.addItem("Near"); 
		self.relationsDrop.addItem("North of"); 
		self.relationsDrop.addItem("South of");
		self.relationsDrop.addItem("East of");
		self.relationsDrop.addItem("West of");
		pushLayout.addWidget(self.relationsDrop,1,1); 

		self.objectsDrop = QComboBox();
		self.objectsDrop.addItem("You"); 
		pushLayout.addWidget(self.objectsDrop,1,2); 

		self.pushButton = QPushButton("Submit"); 
		self.pushButton.setStyleSheet("background-color: green"); 
		pushLayout.addWidget(self.pushButton,2,0); 

		self.layout.addLayout(pushLayout,3,1,3,4); 



		#Drone Launch Section
		#**************************************************************
		droneLabel = QLabel("Scout Controls"); 
		droneLabel.setFont(sectionHeadingFont);
		droneLayout = QGridLayout(); 
		droneLabel.setAlignment(Qt.AlignBottom | Qt.AlignHCenter); 
		droneFrame = QFrame(); 
		droneFrame.setStyleSheet('border: 1px solid black')
		droneLayout.addWidget(droneLabel,0,0); 
		#droneLayout.addWidget(droneFrame,1,0,2,3); 

		
		#self.layout.addWidget(droneLabel,9,1); 

		self.updateTimerLCD = QLCDNumber(self); 
		self.updateTimerLCD.setSegmentStyle(QLCDNumber.Flat); 
		self.updateTimerLCD.setStyleSheet("background-color:rgb(255,0,0)"); 
		self.updateTimerLCD.setMaximumHeight(25);
		self.updateTimerLCD.setMinimumHeight(25);  
		#self.layout.addWidget(self.updateTimerLCD,10,1); 
		droneLayout.addWidget(self.updateTimerLCD,1,0); 

		self.droneButton = QPushButton("Launch\nScout"); 
		#self.layout.addWidget(self.droneButton,10,1); 
		droneLayout.addWidget(self.droneButton,1,0); 
		self.layout.addLayout(droneLayout,6,1,2,3)


		#Robot Pull Section
		#**************************************************************
		pullLabel = QLabel("Robot Pull"); 
		pullLabel.setFont(sectionHeadingFont);
		pullLayout = QGridLayout(); 
		pullLabel.setAlignment(Qt.AlignBottom | Qt.AlignHCenter); 
		pullLayout.addWidget(pullLabel,0,0,1,2); 

		#self.layout.addWidget(pullLabel,12,1); 

		self.pullQuestion = QLineEdit("Awaiting Query");
		self.pullQuestion.setReadOnly(True); 
		self.pullQuestion.setAlignment(QtCore.Qt.AlignCenter); 
		f = self.pullQuestion.font(); 
		f.setPointSize(12); 
		self.pullQuestion.setFont(f); 
		pullLayout.addWidget(self.pullQuestion,1,0,1,4); 

		self.yesButton = QPushButton("Yes");  
		self.yesButton.setStyleSheet("background-color: green"); 
		pullLayout.addWidget(self.yesButton,2,0); 

		self.noButton = QPushButton("No");  
		self.noButton.setStyleSheet("background-color: red"); 
		pullLayout.addWidget(self.noButton,2,3); 

		self.layout.addLayout(pullLayout,9,1,2,4)


		#Belief Opacity Slider
		#**************************************************************
		sliderLayout = QGridLayout(); 
		self.beliefOpacitySlider = QSlider(Qt.Horizontal); 
		self.beliefOpacitySlider.setSliderPosition(30)
		self.beliefOpacitySlider.setTickPosition(QSlider.TicksBelow)
		self.beliefOpacitySlider.setTickInterval(10); 
		sliderLayout.addWidget(self.beliefOpacitySlider,0,0); 
		#self.layout.addWidget(self.beliefOpacitySlider,16,0,1,1); 
		belLabel = QLabel("Belief Opacity"); 
		belLabel.setAlignment(Qt.AlignLeft); 
		sliderLayout.addWidget(belLabel,0,1,1,2); 

		#Sketch Opacity Slider
		self.sketchOpacitySlider = QSlider(Qt.Horizontal); 
		self.sketchOpacitySlider.setSliderPosition(70); 
		self.sketchOpacitySlider.setTickPosition(QSlider.TicksBelow); 
		self.sketchOpacitySlider.setTickInterval(10); 
		sliderLayout.addWidget(self.sketchOpacitySlider,1,0); 
		#self.layout.addWidget(self.sketchOpacitySlider,17,0,1,1); 
		sketchOLabel = QLabel("Sketch Opacity"); 
		sketchOLabel.setAlignment(Qt.AlignLeft); 
		sliderLayout.addWidget(sketchOLabel,1,1,1,2); 

		self.layout.addLayout(sliderLayout,16,0,2,2); 


	def connectElements(self):
		self.startSketchButton.clicked.connect(lambda:startSketch(self)); 

		self.droneButton.clicked.connect(lambda: launchDrone(self)); 

		self.yesButton.clicked.connect(lambda: answerRobotPullQuestion(self,"Is")); 

		self.noButton.clicked.connect(lambda: answerRobotPullQuestion(self,"Is not")); 

		self.pushButton.clicked.connect(lambda: pushButtonPressed(self)); 

		self.imageScene.mousePressEvent = lambda event:imageMousePress(event,self); 
		self.imageScene.mouseMoveEvent = lambda event:imageMouseMove(event,self); 
		self.imageScene.mouseReleaseEvent = lambda event:imageMouseRelease(event,self);


		self.saveBeliefShortcut = QShortcut(QKeySequence("Ctrl+B"),self); 
		self.saveBeliefShortcut.activated.connect(self.saveBeliefs); 

		self.saveAllShortcut = QShortcut(QKeySequence("Ctrl+A"),self); 
		self.saveAllShortcut.activated.connect(self.saveAllThings);

		self.beliefOpacitySlider.valueChanged.connect(lambda: makeBeliefMap(self)); 
		self.sketchOpacitySlider.valueChanged.connect(lambda: redrawSketches(self)); 


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


