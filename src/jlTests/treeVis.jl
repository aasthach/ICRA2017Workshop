using POMDPs
using BasicPOMCP
using POMDPModels
using POMDPModelTools
using POMDPSimulators
using D3Trees
using Random
using ParticleFilters

#pomdp = BabyPOMDP()
#pomdp = TigerPOMDP()
#solver = POMCPSolver(tree_queries=1000, c=10.0, rng=MersenneTwister(1))
#planner = solve(solver, pomdp)

###Uncomment here to show tree
#a, info = action_info(planner, initialstate_distribution(pomdp), tree_in_info=true)
#inbrowser(D3Tree(info[:tree], init_expand=3),"firefox")


#for (s,a,o) in stepthrough(pomdp,planner,"sao",max_steps=10)
	#println("State was $s")
#end

#history = sim(pomdp,max_steps=10) do obs
#	println("Observation was $obs.")
#	return TIGER_OPEN_LEFT
#end






pomdp = LightDark1D()
N = 1000; 
solver = POMCPSolver(tree_queries=1000, c=10.0, rng=MersenneTwister(1))
planner = solve(solver, pomdp)


up = SIRParticleFilter(pomdp,N);

b0 = POMDPModels.LDNormalStateDist(-15.0,5.0);

global oSet
oSet = [-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20]; 

global bel
global prevAct
global allBel
allBel = []; 

prevAct = 1; 
bel = initialize_belief(up,b0); 

sim(pomdp,max_steps=20,show_progress=false) do o
	global bel
	global allBel
	global prevAct
	global oSet
	o = oSet[1]; 
	deleteat!(oSet,1);
	#println(o,oSet);
	#println(o); 
	newBel,u = update_info(up,bel,prevAct,o)
	prevAct,info = action_info(planner,newBel)
	bel = newBel
	push!(allBel,bel)
	println(prevAct)

	return prevAct;
end


using Plots
using Reel

frames = Frames(MIME("image/png"),fps=4)
for b in allBel
    ys = [s.y for s in particles(b)]
    nbins = round(Int, (maximum(ys)-minimum(ys))*2)
    push!(frames, histogram(ys,
                            xlim=(-20,20),
                            ylim=(0,1000),
                            nbins=nbins,
                            label="",
                            title="Particle Histogram")
                            
    )
end
write("hist.gif", frames)
