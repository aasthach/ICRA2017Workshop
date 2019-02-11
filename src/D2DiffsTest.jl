
import Base: ==, +, *, -


using POMDPs
using POMDPModels
using BasicPOMCP
using POMDPPolicies
using POMDPSimulators
using ParticleFilters
import ParticleFilters: BasicParticleFilter, UnweightedParticleFilter
using Distributions
using Random
import POMDPs: initialstate_distribution, actions, n_actions, reward, generate_s, discount, isterminal, transition, observation, generate_o
using POMDPModelTools
import POMDPModelTools: obs_weight

mutable struct D2Diffs <: POMDPs.POMDP{Vector{Float64},Int,Int}
    discount_factor::Float64
    found_r::Float64
    lost_r::Float64
    step_size::Float64
    movement_cost::Float64
end




D2Diffs() = D2Diffs(0.9, 5.0, 0.0, 1.0, 0.0)

discount(p::D2Diffs) = p.discount_factor

initialstate_distribution(pomdp::D2Diffs) = MvNormal([10.,10.])


function generate_o(::D2Diffs, s::Vector{Float64}, a::Int64, sp::Vector{Float64}, rng::AbstractRNG)
	
	#println(sp); 

	if sqrt(sp[1]*sp[1] + sp[2]*sp[2]) < 1
		return 4
	elseif abs(sp[1]) > abs(sp[2])
		if sp[1] < 0
			return 0
		else
			return 1
		end
	else
		if sp[2] > 0
			return 2
		else
			return 3
		end
	end
end


#function obs_weight(::D2Diffs, s::Vector{Float64}, a::Int64, sp::Vector{Float64}, o::Int64)

	#return 1; 

#end


function generate_s(p::D2Diffs, s::Vector{Float64}, a::Int, rng::AbstractRNG)
	sig_0 = [1.,1.]
	#sig_stationary = [sqrt(0.00001),sqrt(0.00001)]
	d= MvNormal(sig_0)
    noise = rand(d::MvNormal) 


    if a == 0
        s= s + [-1.,0.] + noise
    elseif a == 1
        s= s + [1.,0.] + noise
    elseif a == 2
    	s= s + [0.,1.] + noise
    elseif a == 3
    	s= s + [0.,-1.] + noise
    else 
    	s= s + noise
    end

    s[1] = max(-10,s[1])
    s[1] = min(10,s[1]); 
    s[2] = max(-10,s[2])
    s[2] = min(10,s[2]);

    return s   
    

end


function reward(p::D2Diffs, s::Vector{Float64}, a::Int)
   	if sqrt(s[1]*s[1] + s[2]*s[2]) < 1
		return 5
	else
		return 0
	end
end;


convert_s(::Type{A}, s::Vector{Float64}, p::D2Diffs) where A<:AbstractVector = eltype(A)[s.status, s.y]
convert_s(::Type{Vector{Float64}}, s::A, p::D2Diffs) where A<:AbstractVector = Vector{Float64}(Int64(s[1]), s[2])



mutable struct DummyHeuristic1DPolicy <: POMDPs.Policy
    thres::Float64
end
DummyHeuristic1DPolicy() = DummyHeuristic1DPolicy(0.1)

mutable struct SmartHeuristic1DPolicy <: POMDPs.Policy
    thres::Float64
end
SmartHeuristic1DPolicy() = SmartHeuristic1DPolicy(0.1)



function action(p::SmartHeuristic1DPolicy, b::B) where {B}
	if sqrt(s[1]*s[1] + s[2]*s[2]) < 1
		return 4
	elseif abs(s[1]) > abs(s[2])
		if s[1] < 0
			return 0
		else
			return 1
		end
	else
		if s[2] > 0
			return 2
		else
			return 3
		end
	end
end




actions(::D2Diffs) = [0,1,2,3,4]
n_actions(p::D2Diffs) = length(actions(p))




numSims = 100; 
simLength = 100; 
allSims = []; 
exploreConstant = 10; 
timeGiven = .1; 

for i in 1:numSims

	#println("Loaded Imports"); 
	println("Starting simulation $i of $numSims"); 
	pomdp = D2Diffs(0.9, 5.0, 0.0, 1.0, 0.0)

	#println("Loaded Problem"); 

	solver = POMCPSolver(max_time = timeGiven,c=exploreConstant,max_depth=1000); 

	#println("Loaded Solver"); 

	planner = solve(solver, pomdp); 
	#rand_policy = RandomPolicy(pomdp);

	#println("Loaded Planner"); 

	#pf = SIRParticleFilter(pomdp,10000); 
	RNG = MersenneTwister()
	upf = UnweightedParticleFilter(pomdp,10000,RNG); 
	#println("Loaded Filter"); 

	#rollout_sim = RolloutSimulator(max_steps=1000); 
	hr = HistoryRecorder(max_steps=simLength,show_progress=false)

	#println("Loaded Sim"); 

	r_pomcp = simulate(hr, pomdp, planner, upf);
	push!(allSims,r_pomcp); 
	#r_rand = simulate(hr, pomdp, rand_policy, pf, MvNormal([4.,4.]));

	totalR = []
	for step in eachstep(r_pomcp, "(s, a, r, sp, o)")    
	    #println("reward $(step.r) received when state $(step.sp) was reached after action $(step.a) was taken in state $(step.s). Obs: $(step.o)")
	   	push!(totalR,step.r)
	end
	println("The Total Reward was: $(sum(totalR))")



	#allBels = []; 


	#for step in eachstep(r_pomcp, "(s, a, r, sp, b)")  
	#	x0 = []; 
	#	y0 = []; 
	#	for a in step.b.particles
	#		push!(x0,a[1])
	#		push!(y0,a[2])
	#	end
	#	push!(allBels,[x0,y0]); 
	#end



	
	#open("beliefTest.txt","w") do f
	#	writedlm(f,allBels); 
	#end

end

using DelimitedFiles
open("tenthSecondPOMCP_2_c10.txt","w") do f
	writedlm(f,allSims); 
end