"""
***********************************************************
File: juliaController.py
Author: Luke Burks
Date: May 2018

Implements a subclasses Controller which calls the 
JuliaPOMDP packages implementation for online planning
and control.
Also implements a recreation function which accounts for 
model changes

***********************************************************
"""

__author__ = "Luke Burks"
__copyright__ = "Copyright 2018"
__credits__ = ["Luke Burks"]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = "Luke Burks"
__email__ = "luke.burks@colorado.edu"
__status__ = "Development"


import julia; 
from julia import Main
import numpy as np

import matplotlib.pyplot as plt

class JuliaMAPController():


	def __init__(self,model=None):
		self.model = model; 

		j = julia.Julia(); 


		j.include("BlindMAP.jl"); 
		self.getAct = j.eval("getAct"); 
		create = j.eval("create"); 
		create(); 


	def getActionKey(self):

		act = self.getAct(0); 
		bel = Main.parseBel;

		return act; 

	def testJulia(self):

		#ping julia bridge
		# act = self.getAct(0); 
		# bel = Main.parseBel; 
		# self.belItOut(bel); 
		moves = ["Left","Right","Up","Down"]; 

		pose = [200,200,400,400]; 
		for i in range(0,100):
			#obs = int(np.random.choice([0,1],p=[.9,.1])); 
			#print(obs); 
			obs = 0; 
			if(np.sqrt((pose[0]-pose[2])**2 + (pose[1]-pose[3])**2) < 25):
				obs = 1; 

			act = self.getAct(obs); 
			bel = Main.parseBel;
			#print(act); 
			if(act == 0):
				pose[0] -= 10; 
			elif(act == 1):
				pose[0] += 10; 
			elif(act == 2):
				pose[1] += 10; 
			elif(act == 3):
				pose[1] -= 10; 

			pose[2] = np.random.normal(pose[2],8)
			pose[3] = np.random.normal(pose[3],8); 

			print(pose); 
			print(moves[act]); 
			print(""); 

			self.belItOut(bel,pose); 
		return act; 

	
	def getQuestionIndex(self):
		return np.random.randint(0,len(self.model.sketches.keys())*5 + 5); 


	def belItOut(self,bel,pose):

		parts = np.array(bel); 

		#cop
		copParts = parts[:,0:2]; 
		robParts = parts[:,2:4];


		low = [0,0]; 
		high = [437,754]; 
		res = 100; 
		[h,_,_,_] = plt.hist2d(robParts[:,0],robParts[:,1],range=[[0,437],[0,754]],bins=[100,100]); 
		x, y = np.mgrid[low[0]:high[0]:(float(high[0]-low[0])/res), low[1]:high[1]:(float(high[1]-low[1])/res)]

		plt.contourf(x,y,h); 
		plt.scatter(pose[0],pose[1],c='g');
		plt.scatter(pose[2],pose[3],c='r') 
		plt.pause(0.1); 


if __name__ == '__main__':
	c = JuliaMAPController(); 
	print(c.getActionKey()); 