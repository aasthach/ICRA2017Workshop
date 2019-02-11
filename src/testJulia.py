import julia

from julia import Base
from julia.Base import sin
from julia import Main

import numpy as np

Main.theVar = [1,2,3]; 

a = np.array([2,3,4]); 
theSin = Main.eval("sin.(theVar)")

print(theSin+a); 



j = julia.Julia(); 


j.include("RTDuneJulia.jl"); 
getAct = j.eval("getAct"); 

print(getAct(np.array([5.,5.]))); 