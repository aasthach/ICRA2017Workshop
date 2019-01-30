"""
***********************************************************
File: weightedGreedyController.py
Author: Luke Burks
Date: January 2019

A controller that goes to the point of highest probability,
weighted by the distance

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


import numpy as np

class WGController():


	def __init__(self,model=None):
		self.model = model; 

		#Ratio of consideration for value vs distance
		#infinite: pure value
		#zero:     pure distance
		#one:      equal consideration
		self.eFudge = 4/3; 


	def distance(self,p1,p2):
		return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2); 

	def getActionKey(self):
		means = self.model.belief.getMeans(); 
		pose = self.model.copPose;

		#dists = [self.distance(means[i],pose) for i in range(0,len(means))];
		#evals = [self.model.belief.pointEval(means[i]) for i in range(0,len(means))]; 

		dists = []; 
		evals = []; 
		for i in range(0,len(means)):
			dists.append(self.distance(pose,means[i])); 
			evals.append(self.model.belief.pointEval(means[i])); 


		dists = np.array(dists)/np.max(dists); 
		evals = np.array(evals)/np.max(evals); 

		

		best = -10; 
		bestInd = 0; 
		for i in range(0,len(dists)):
			tmp = self.eFudge*evals[i] - dists[i];
			if(tmp > best):
				best = tmp
				bestInd = i; 
		goal = means[bestInd]; 
		#print(goal,evals[i],dists[i],i); 
		#print(evals); 

		if(abs(goal[0]-pose[0]) > abs(goal[1]-pose[1])):
			if(pose[0] < goal[0]):
				return 3; 
			else:
				return 2;
		else:
			if(pose[1] < goal[1]):
				return 1; 
			else:
				return 0;

		
		return act; 


	def getQuestionIndex(self):
		return np.random.randint(0,4); 

if __name__ == '__main__':
	a = 0; 

