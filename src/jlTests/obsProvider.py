#Sends observations, gets actions and beliefs

import julia
import numpy as np
import matplotlib.pyplot as plt

from julia import Main


j = julia.Julia(); 

j.include("actionProvider.jl"); 

getAct = j.eval("getAct"); 


allBel = []; 

for i in range(0,20):
	obs = np.random.randint(-20,0); 


	act = getAct(float(obs)); 
	bel = Main.parseBel; 

	print(act,np.mean(bel)); 

	plt.hist(bel,range=(-20,20));
	plt.pause(0.1); 