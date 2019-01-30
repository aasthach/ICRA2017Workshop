"""
***********************************************************
File: robotControllers.py
Author: Luke Burks
Date: May 2018

Implements a Controller class which makes changes to a 
model

***********************************************************
"""

__author__ = "Luke Burks"
__copyright__ = "Copyright 2018"
__credits__ = ["Luke Burks"]
__license__ = "GPL"
__version__ = "0.1.1"
__maintainer__ = "Luke Burks"
__email__ = "luke.burks@colorado.edu"
__status__ = "Development"

import numpy as np


class Controller:

	def __init__(self,model):
		self.model = model; 



	def getActionKey(self):
		goal = self.model.belief.findMAPN(); 
		pose = self.model.copPose; 

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

	def getActionKey_Myopic(self):
		goal = self.model.belief.findMAPN(); 
		pose = self.model.copPose; 

	
	def getQuestionIndex(self):
		return np.random.randint(0,4); 