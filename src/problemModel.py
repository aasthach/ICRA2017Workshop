"""
***********************************************************
File: problemModel.py
Author: Luke Burks
Date: April 2018

Implements a Model class which contains information about 
rewards, transitions, and observations

Models may be either held or true

#Transition Layer defines the difference from nominal speed

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

from gaussianMixtures import Gaussian,GM; 
from softmaxModels import Softmax;

import matplotlib.pyplot as plt
import numpy as np; 

from interfaceFunctions import distance
from interfaceFunctions import mahalanobisDistance

class Model:

	def __init__(self,params,size = [437,754],trueModel = False,):

		self.truth = trueModel

		#Cop Pose
		self.copPose = params['Model']['copInitPose'];
		if(self.copPose == 'None'):
			self.copPose = [np.random.randint(0,437),np.random.randint(0,754)]; 


		self.ROBOT_VIEW_RADIUS = params['Model']['robotViewRadius']; 
		self.ROBOT_CATCH_RADIUS = params['Model']['robotCatchRadius']; 
		self.ROBOT_SIZE_RADIUS = params['Model']['robotSizeRadius']; 
		self.ROBOT_NOMINAL_SPEED = params['Model']['robotNominalSpeed']; 
		self.TARGET_NOMINAL_SPEED = params['Model']['targetSpeed']; 
		self.TARGET_SIZE_RADIUS = params['Model']['targetSizeRadius']; 

		self.TARGET_MOVEMENT_MODEL = params['Model']['targetMovementModel']; 

		self.MAX_BELIEF_SIZE = params['Model']['numRandBel']; 

		self.BREADCRUMB_TRAIL_LENGTH = params['Model']['breadCrumbLength']; 

		self.history = {'beliefs':[],'positions':[],'sketches':{},'humanObs':[]}; 

		belModel = params['Model']['belNum']

		#Make Target or Belief
		if(not self.truth):
			if(belModel == 'None'):
				self.belief = GM(); 

				for i in range(0,self.MAX_BELIEF_SIZE):
					self.belief.addNewG([np.random.randint(0,437),np.random.randint(0,754)],[[4000+500*np.random.normal(),0],[0,4000+500*np.random.normal()]],np.sqrt(np.random.random())); 
				self.belief.normalizeWeights(); 
			else:
				self.belief = np.load("../models/beliefs{}.npy".format(belModel))[0]


		self.robPose = params['Model']['targetInitPose'];
		if(self.robPose == 'None'):
			self.robPose = [np.random.randint(0,437),np.random.randint(0,754)]; 


		if(self.TARGET_MOVEMENT_MODEL == 'NCV'):
			self.robPose = [self.robPose[0],self.robPose[1],[-1,1][np.random.randint(0,2)]*self.TARGET_NOMINAL_SPEED,[-1,1][np.random.randint(0,2)]*self.TARGET_NOMINAL_SPEED]

		self.bounds = {'low':[0,0,-self.TARGET_NOMINAL_SPEED,-self.TARGET_NOMINAL_SPEED],'high':[437,754,self.TARGET_NOMINAL_SPEED,self.TARGET_NOMINAL_SPEED]}
		
		self.setupTransitionLayer(); 
		self.setupCostLayer(); 

		#TODO: Spatial Relations don't always map correctly, fix it....
		#self.spatialRealtions = {'Near':0,'South of':4,'West of':1,'North of':2,'East of':3}; 
		#self.spatialRealtions = {'Near':0,'South of':1,'West of':2,'North of':3,'East of':4}; 
		self.spatialRealtions = {'Near':0,'South of':3,'West of':4,'North of':1,'East of':2}; 

		self.sketches = {};
		self.sketchParams = {}; 

		self.prevPoses = []; 


		

		
	def setupCostLayer(self):
		self.costLayer = np.zeros(shape=(self.bounds['high'][0],self.bounds['high'][1]));

		x = self.robPose[0]
		y = self.robPose[1]; 

		for i in range(self.bounds['low'][0],self.bounds['high'][0]):
			for j in range(self.bounds['low'][1],self.bounds['high'][1]):
				if(not (i==x and y==j)):
					self.costLayer[i,j] = 1/np.sqrt((x-i)*(x-i) + (y-j)*(y-j)); 



	def setupTransitionLayer(self):
		self.transitionLayer = np.zeros(shape=(self.bounds['high'][0],self.bounds['high'][1]));

		if(self.truth):
			self.transitionLayer = np.load('../models/trueTransitions.npy'); 



	def transitionEval(self,x):
		if(x[0] > self.bounds['low'][0] and x[1] > self.bounds['low'][1] and x[0] < self.bounds['high'][0] and x[1] < self.bounds['high'][1]):
			return self.transitionLayer[x[0],x[1]]; 
		else:
			return -1e10;  

	def costEval(self,x):
		if(x[0] > self.bounds['low'][0] and x[1] > self.bounds['low'][1] and x[0] < self.bounds['high'][0] and x[1] < self.bounds['high'][1]):
			return self.rewardLayer[x[0],x[1]]; 
		else:
			return 0;  





	def makeSketch(self,vertices,name):
		pz = Softmax(); 
		vertices.sort(key=lambda x: x[1])

		pz.buildPointsModel(vertices,steepness=3); 
		self.sketches[name] = pz; 

		cent = [0,0,0,0]; 
		for i in range(0,len(vertices)):
			cent[0] += vertices[i][0]/len(vertices); 
			cent[1] += vertices[i][1]/len(vertices); 

		for i in range(0,len(vertices)):
			cent[2] = max(cent[2],np.abs(vertices[i][0]-cent[0]))
			cent[3] = max(cent[3],np.abs(vertices[i][1]-cent[1]))

		self.sketchParams[name] = cent; 



	def stateObsUpdate(self,name,relation,pos="Is"):
		if(name == 'You'):
			#Take Cops Position, builid box around it
			cp=self.copPose; 
			points = [[cp[0]-5,cp[1]-5],[cp[0]+5,cp[1]-5],[cp[0]+5,cp[1]+5],[cp[0]-5,cp[1]+5]]; 
			soft = Softmax()
			soft.buildPointsModel(points,steepness=3); 
		else:
			#If you want to actually pull the sketch, you'll still need to find "all classes of dir X"
			#soft = self.sketches[name]; 

			#Just make a hidden box with similar spatial extent
			cp = self.sketchParams[name]; 
			points = [[cp[0]-cp[2],cp[1]-cp[3]],[cp[0]+cp[2],cp[1]-cp[3]],[cp[0]+cp[2],cp[1]+cp[3]],[cp[0]-cp[2],cp[1]+cp[3]]];
			soft = Softmax(); 
			soft.buildPointsModel(points,steepness=3); 

		softClass = self.spatialRealtions[relation]; 

		if(pos=="Is"):
			self.belief = soft.runVBND(self.belief,softClass); 
			self.belief.normalizeWeights(); 
		else:
			tmp = GM();
			for i in range(0,5):
				if(i!=softClass):
					tmp.addGM(soft.runVBND(self.belief,i));
			tmp.normalizeWeights(); 
			self.belief=tmp; 
		if(self.belief.size > self.MAX_BELIEF_SIZE):
			self.belief.condense(self.MAX_BELIEF_SIZE); 
			self.belief.normalizeWeights()



	def stateLWISUpdate(self):

		cp=self.prevPoses[-1]; 
		prev = self.prevPoses[-2]; 
		theta = np.arctan2([cp[1]-prev[1]],[cp[0]-prev[0]]);
		#print(theta);  
		radius = self.ROBOT_VIEW_RADIUS; 
		points = [[cp[0]-radius,cp[1]-radius],[cp[0]+radius,cp[1]-radius],[cp[0]+radius,cp[1]+radius],[cp[0]-radius,cp[1]+radius]]; 
		soft = Softmax()
		soft.buildPointsModel(points,steepness=5);
		#soft.buildTriView(pose = [cp[0],cp[1],theta],length=10,steepness=5); 
		change = False; 
		post = GM(); 
		for g in self.belief:
			#if(distance(cp,g.mean) > self.ROBOT_VIEW_RADIUS+5):
			if(mahalanobisDistance(cp,g.mean,g.var) > 2):
				post.addG(g); 
			else:
				change = True; 
				tmp = soft.lwisUpdate(g,0,100,inverse=True);
				#self.bounds = {'low':[0,0],'high':[437,754]}
				tmp.mean[0] = max(self.bounds['low'][0]+1,tmp.mean[0]); 
				tmp.mean[1] = max(self.bounds['low'][1]+1,tmp.mean[1]); 
				tmp.mean[0] = min(self.bounds['high'][0]-1,tmp.mean[0]);
				tmp.mean[1] = min(self.bounds['high'][1]-1,tmp.mean[1]);  


				post.addG(tmp);
		self.belief = post; 
		self.belief.normalizeWeights(); 

		return change; 


	def stateDynamicsUpdate(self):

		if(self.truth):
			#update robber pose
			p = np.matrix(self.robPose); 
			s = self.TARGET_NOMINAL_SPEED; 
			if(self.TARGET_MOVEMENT_MODEL == 'Stationary'):
				STM = np.matrix([[1,0],[0,1]]); 
				var = np.matrix([[0.00000001,0],[0,0.00000001]]); 
				x = p.tolist()[0]
			elif(self.TARGET_MOVEMENT_MODEL == 'Random'):
				STM = np.matrix([[1,0],[0,1]]);
				var = np.matrix([[s**2, 0],[0,s**2]]); 
				x = np.random.multivariate_normal((STM*(p.T)).T.tolist()[0],var,size =1)[0].tolist();
			elif(self.TARGET_MOVEMENT_MODEL == 'NCV'):
				STM = np.matrix([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]); 
				var = np.matrix([[.1,0,0,0],[0,.1,0,0],[0,0,.25,0],[0,0,0,.25]]);

				x = np.random.multivariate_normal((STM*(p.T)).T.tolist()[0],var,size =1)[0].tolist();
			else:
				print("Please Configure a valid target model"); 
				exit(0); 


			if(x[0] < 0 or x[0] > self.bounds['high'][0]):
				if(self.TARGET_MOVEMENT_MODEL == 'NCV'):
					x[2] = -x[2]; 

			if(x[1] < 0 or x[1] > self.bounds['high'][1]):
				if(self.TARGET_MOVEMENT_MODEL == 'NCV'):
					x[3] = -x[3]; 			

			
			for i in range(0,len(x)):
				x[i] = max(self.bounds['low'][i]+self.TARGET_SIZE_RADIUS/2,x[i]); 
				x[i] = min(self.bounds['high'][i]-self.TARGET_SIZE_RADIUS/2,x[i]); 



			x[0] = int(x[0]); 
			x[1] = int(x[1]); 

			self.robPose = x; 

		if(not self.truth):
			#update belief
			if(self.TARGET_MOVEMENT_MODEL == 'Stationary'):
				pass;
			else:
				s = self.TARGET_NOMINAL_SPEED;
				var = np.array([[s**2, 0],[0,s**2]]); 
				for g in self.belief:
					g.var = (np.array(g.var) + var).tolist(); 





