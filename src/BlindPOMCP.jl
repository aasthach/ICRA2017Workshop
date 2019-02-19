import Base: ==, +, *, -


using POMDPs
using POMDPModels
using BasicPOMCP
using POMDPPolicies
using POMDPSimulators
using ParticleFilters
import ParticleFilters: BasicParticleFilter, UnweightedParticleFilter, SIRParticleFilter
using Distributions
using Random
import POMDPs: initialstate_distribution, actions, n_actions, reward, generate_s, discount, isterminal, transition, observation, generate_o
using POMDPModelTools
import POMDPModelTools: obs_weight



mutable struct POMCP_Blind_POMDP <: POMDPs.POMDP{Vector{Float64},Int,Int}
    discount_factor::Float64
    found_r::Float64
    lost_r::Float64
    step_size::Float64
    movement_cost::Float64
end


global robotViewRadius


function initialstate_distribution(pomdp::POMCP_Blind_POMDP,s::Vector{Int64})

	numMixands = 10000; 
	mixands = MvNormal[]; 
	for i=1:numMixands
		tmp = MvNormal([s[1],s[2],rand(10:427),rand(10:744)],[0.01,0.01,rand(5:10),rand(5:10)])
		push!(mixands,tmp); 
	end


	return MixtureModel(mixands); 

end

###
function initialstate_distribution(pomdp::POMCP_Blind_POMDP)
	
	s=[250,50]
	numMixands = 5000; 
	mixands = MvNormal[]; 
	#for i=1:numMixands
		#tmp = MvNormal([s[1],s[2],rand(10:427),rand(10:744)],[0.01,0.01,rand(5:10),rand(5:10)])
		#push!(mixands,tmp); 
	#end
	tmp = MvNormal([s[1],s[2],200,50],[0.01,0.01,rand(5:10),rand(5:10)])
	push!(mixands,tmp); 

	return MixtureModel(mixands); 

end
###

function generate_o(::POMCP_Blind_POMDP, s::Vector{Float64}, a::Int64, sp::Vector{Float64}, rng::AbstractRNG)

	global robotViewRadius

	if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < robotViewRadius/2
		return 2
	elseif sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < robotViewRadius
		return 1; 
	else
		return 0; 
	end

end



function obs_weight(::POMCP_Blind_POMDP, s::Vector{Float64}, a::Int64, sp::Vector{Float64}, o::Int64)

	global robotViewRadius

	if o==2
		if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < robotViewRadius/2
			return .75; 
		elseif sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < robotViewRadius
			return .20; 
		else
			return .05
		end
	elseif o==1
		if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < robotViewRadius/2
			return .05; 
		elseif sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < robotViewRadius
			return .90; 
		else
			return .05
		end
	else
		if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < robotViewRadius/2
			return .01; 
		elseif sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < robotViewRadius
			return .04; 
		else
			return .95
		end
	end


end


function generate_s(p::POMCP_Blind_POMDP, s::Vector{Float64}, a::Int, rng::AbstractRNG)
	sig_0 = [1.,1.,8.,8.]

	d= MvNormal(sig_0)
    noise = rand(d::MvNormal) 

    noise[1] = 0; 
    noise[2] = 0; 

    if a == 2
        s= s + [-10.,0.,0.,0.] + noise
    elseif a == 3
        s= s + [10.,0.,0.,0.] + noise
    elseif a == 1
    	s= s + [0.,10.,0.,0.] + noise
    elseif a == 0
    	s= s + [0.,-10.,0.,0.] + noise
    else 
    	s= s + noise
    end

    s[1] = max(0,s[1])
    s[1] = min(437,s[1]); 
    s[2] = max(0,s[2])
    s[2] = min(754,s[2]);
    s[3] = max(0,s[3])
    s[3] = min(437,s[3]); 
    s[4] = max(0,s[4])
    s[4] = min(754,s[4]);

    return s   
    

end



function reward(p::POMCP_Blind_POMDP, s::Vector{Float64}, a::Int)
	if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < 25
		return 5; 
	else
		return 0; 
	end

end;


POMCP_Blind_POMDP() = POMCP_Blind_POMDP(0.9, 5.0, 0.0, 1.0, 0.0)

convert_s(::Type{A}, s::Vector{Float64}, p::POMCP_Blind_POMDP) where A<:AbstractVector = eltype(A)[s.status, s.y]
convert_s(::Type{Vector{Float64}}, s::A, p::POMCP_Blind_POMDP) where A<:AbstractVector = Vector{Float64}(Int64(s[1]), s[2])


global pomdp
global up
global solver
global planner
global bel
global parseBel
global action
global objectModels
global planner

objectModels = Any[]

discount(p::POMCP_Blind_POMDP) = p.discount_factor
actions(p::POMCP_Blind_POMDP) = [0,1,2,3]
n_actions(p::POMCP_Blind_POMDP) = length(actions(p))

function create(copPose::Vector{Int64})
	
	global pomdp
	global up
	global solver
	global planner
	global bel
	global robotViewRadius
	global planner

	robotViewRadius = 50; 

	#println("copPose"); 
	#println(copPose); 

	discount(p::POMCP_Blind_POMDP) = p.discount_factor
	actions(::POMCP_Blind_POMDP) = [0,1,2,3]
	n_actions(p::POMCP_Blind_POMDP) = length(actions(p))


	pomdp =  POMCP_Blind_POMDP(0.9,5.0,0.0,1.0,0.0)
	b0 = initialstate_distribution(pomdp,copPose);


	N = 100000; 
	

	#println(mode(bel))

	solver = POMCPSolver(max_time=1.0, c=10.0,max_depth=50,tree_queries=N)
	planner = solve(solver,pomdp); 
	up = updater(planner); 
	bel = initialize_belief(up,b0); 

end



function getAct(o::Vector{Int64})

	global bel
	global action
	global planner
	global up
	
	MAP = mode(bel); 
	#VOI(bel); 
	print("MAP: ")
	println(MAP)


	action,info = action_info(planner,bel)
	print("Action: ")
	println(action); 

	newBel,u = update_info(up,bel,action,o[5])
	bel = newBel
	
	print("New Map: "); 
	MAP = mode(bel); 
	println(MAP); 

	pythonifyBel(bel); 

	return action

end


function pythonifyBel(bel)
	
	global parseBel
	parseBel = []; 
	parts = particles(bel) 

	for i in 1:n_particles(bel)
		push!(parseBel,particle(bel,i)); 
	end
	
	
end

function addModel(a::Any)
	
	global objectModels

	push!(objectModels,a); 


end


function VOI(bel)

	global up
	global robotViewRadius
	
	action = 0; 


	belNo,u = update_info(up,bel,action,0)
	belYes,u = update_info(up,bel,action,1)

	VOINO = 0; 
	for i in 1:n_particles(belNo)
		s = particle(belNo,i); 
		if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < robotViewRadius
			VOINO += 5; 
		end
	end

	VOIYes = 0; 
	for i in 1:n_particles(belYes)
		s = particle(belYes,i); 
		if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < robotViewRadius
			VOIYes += 5; 
		end
	end

	println("VOI of No:")
	println(VOINO)

	println("VOI of Yes:")
	println(VOIYes)

	println("Average:")
	println((VOINO+VOIYes)/2); 

end



function testSim()
	
	numSims = 100; 
	simLength = 100; 
	N=10000
	allSims = []; 
	exploreConstant = 10; 
	timeGiven = 1; 
	RNG = MersenneTwister(1)
	global robotViewRadius
	robotViewRadius = 25; 

	pomdp = POMCP_Blind_POMDP(0.9,5.0,0.0,1.0,0.0); 

	up = SIRParticleFilter(pomdp,10000,RNG);
	upf = UnweightedParticleFilter(pomdp,10000,RNG); 


	println("Loaded Problem"); 

	solver = POMCPSolver(max_time = timeGiven,c=exploreConstant,max_depth=1000); 
	planner = solve(solver,pomdp); 

	hr = HistoryRecorder(max_steps=simLength,show_progress=false)

	#println("Loaded Sim"); 

	r_pomcp = simulate(hr, pomdp, planner, up);
	
	totalR = []
	for step in eachstep(r_pomcp, "(s, a, r, sp, o)")    
	    #println("reward $(step.r) received when state $(step.sp) was reached after action $(step.a) was taken in state $(step.s). Obs: $(step.o)")
	   	println("$(step.r)")
	   	push!(totalR,step.r)
	end
	println("The Total Reward was: $(sum(totalR))")

end


#create()
#getAct(0)

#testSim()


