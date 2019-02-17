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



mutable struct MAP_Blind_POMDP <: POMDPs.POMDP{Vector{Float64},Int,Int}
    discount_factor::Float64
    found_r::Float64
    lost_r::Float64
    step_size::Float64
    movement_cost::Float64
end




##########TODO###############
function initialstate_distribution(pomdp::MAP_Blind_POMDP)

	numMixands = 5000; 
	copPose = [200,200]
	mixands = MvNormal[]; 
	for i=1:numMixands
		tmp = MvNormal([copPose[1],copPose[2],rand(10:427),rand(10:744)],[0.01,0.01,rand(5:10),rand(5:10)])
		push!(mixands,tmp); 
	end


	return MixtureModel(mixands); 

end

function generate_o(::MAP_Blind_POMDP, s::Vector{Float64}, a::Int64, sp::Vector{Float64}, rng::AbstractRNG)

	if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < 25
		return 1; 
	else
		return 0; 
	end

end



function obs_weight(::MAP_Blind_POMDP, s::Vector{Float64}, a::Int64, sp::Vector{Float64}, o::Int64)

	if o==1
		if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < 25
			return .9; 
		else
			return .1; 
		end
	else
		if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < 25
			return .1; 
		else
			return .9; 
		end

	end


end


function generate_s(p::MAP_Blind_POMDP, s::Vector{Float64}, a::Int, rng::AbstractRNG)
	sig_0 = [sqrt(0.00001),sqrt(0.00001),1.,1.]
	#sig_stationary = [sqrt(0.00001),sqrt(0.00001)]
	d= MvNormal(sig_0)
    noise = rand(d::MvNormal) 


    if a == 0
        s= s + [-10.,0.,0.,0.] + noise
    elseif a == 1
        s= s + [10.,0.,0.,0.] + noise
    elseif a == 2
    	s= s + [0.,10.,0.,0.] + noise
    elseif a == 3
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



function reward(p::MAP_Blind_POMDP, s::Vector{Float64}, a::Int)
	if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < 25
		return 5; 
	else
		return 0; 
	end

end;


MAP_Blind_POMDP() = MAP_Blind_POMDP(0.9, 5.0, 0.0, 1.0, 0.0)

convert_s(::Type{A}, s::Vector{Float64}, p::MAP_Blind_POMDP) where A<:AbstractVector = eltype(A)[s.status, s.y]
convert_s(::Type{Vector{Float64}}, s::A, p::MAP_Blind_POMDP) where A<:AbstractVector = Vector{Float64}(Int64(s[1]), s[2])


global pomdp
global up
global solver
global planner
global bel
global parseBel
global action

function create()
	
	global pomdp
	global up
	global solver
	global planner
	global bel



	discount(p::MAP_Blind_POMDP) = p.discount_factor
	actions(::MAP_Blind_POMDP) = [0,1,2,3,4]
	n_actions(p::MAP_Blind_POMDP) = length(actions(p))


	pomdp = MAP_Blind_POMDP(); 
	b0 = initialstate_distribution(pomdp);


	N = 100000; 
	RNG = MersenneTwister(1)
	up = SIRParticleFilter(pomdp,N,RNG);
	bel = initialize_belief(up,b0); 

	solver = POMCPSolver(tree_queries=1000, c=10.0, rng=RNG)
	planner = solve(solver,pomdp); 

end



function getAct(o::Int64)

	global bel
	global action
	
	MAP = mode(bel); 
	#VOI(bel); 
	#println(MAP)

	#left/right or up/down
	if abs(MAP[1]-MAP[3]) > abs(MAP[2]-MAP[4])

		#left or right
		if MAP[1] > MAP[3]
			action=0
		else
			action=1
		end

	else

		#up or down
		if MAP[2] > MAP[4]
			action=3
		else
			action=2
		end
	end

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
		push!(parseBel,particle(bel,i)); 
	end
	
	
end



function VOI(bel)

	global up

	action = 0; 


	belNo,u = update_info(up,bel,action,0)
	belYes,u = update_info(up,bel,action,1)

	VOINO = 0; 
	for i in 1:n_particles(belNo)
		s = particle(belNo,i); 
		if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < 25
			VOINO += 5; 
		end
	end

	VOIYes = 0; 
	for i in 1:n_particles(belYes)
		s = particle(belYes,i); 
		if sqrt((s[1]-s[3])*(s[1]-s[3]) + (s[2]-s[4])*(s[2]-s[4])) < 25
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



#create()
#getAct(0)