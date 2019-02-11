using POMDPs
using BasicPOMCP
using POMDPModels
using POMDPModelTools
using POMDPSimulators
using D3Trees
using Random
using ParticleFilters


pomdp = LightDark1D()
N = 1000; 
solver = POMCPSolver(tree_queries=1000, c=10.0, rng=MersenneTwister(1))
planner = solve(solver, pomdp)


up = SIRParticleFilter(pomdp,N);

b0 = POMDPModels.LDNormalStateDist(-15.0,5.0);


global bel
global parseBel
parseBel = []; 
global action
bel = initialize_belief(up,b0); 


function getAct(o::Float64)

	global bel
	global action
	action,info = action_info(planner,bel)
	newBel,u = update_info(up,bel,action,o)
	bel = newBel

	pythonifyBel(bel); 

	return action

end


function pythonifyBel(bel)
	
	global parseBel
	parseBel = []; 
	parts = particles(bel) 

	for i in 1:n_particles(bel)
		push!(parseBel,particle(bel,i).y); 
	end
	
	
end
