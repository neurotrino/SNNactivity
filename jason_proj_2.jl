using PyPlot
using Distributions
using GeoStats
using Optim
using Stats
using StatsBase
using Plots
using Distances
using LightGraphs
using Iterators
using JLD2
using OpenCL
using FileIO
using HypothesisTests
using ParallelAccelerator
using Dierckx

###### TO DO #########
# FIX COMBINADICS



#################################################################################################################################################
######################################################################## SIMULATION #############################################################
#################################################################################################################################################


### HERE IS THE CODE DEFINING NETWORK BEHAVIOR
################################################ DEFINING UNITS
const metre=1e2;const meter=1e2;const cm=metre/1e2;const mm=metre/1e3;const um=metre/1e6;const nm=metre/1e9;
const second=1e3;const ms=second/1e3;const Hz=1/second;const voltage=1e3;const mV=voltage/1e3;
const ampere=1e6;const mA=ampere/1e3;const uA=ampere/1e6;const nA=ampere/1e9;
const farad=1e6;const uF=ufarad=farad/1e6;
const siemens=1e3;const mS=msiemens=siemens/1e3; const nS=nsiemens=siemens/1e9;
#################################################
################################################# THESE ARE NEURON PARAMETERS THAT DONT CHANGE
################################################# SO WHY NOT JUST DEFINE THEM HERE ALSO SIMULATION PARAMS
const Ne = 4000
const Ni = 1000
const N = Ne+Ni
@everywhere const run_total = 1000; # total run time

const num_poisson = 3000 # characterize input
const num_projected = 500;
@everywhere const stim_in = 30; # time stimulis is input for
const poisson_probability = .18
const poisson_mean_strength = -5.*10^-5.
const poisson_varriance = .5
const poisson_spikerate = 20
const poisson_max = 1 # how much to increment gP with spike
const current_in =  30 # duration of current input
const current_max = 0

@everywhere const dt=.1
const C =.000281*ufarad
const gL = .00003 * msiemens
const EL = -70.6 * mV
const VT = -50.4 * mV
const DeltaT = 2 * mV
const Vcut = VT + 5 * DeltaT
const tauw = 144 * ms
const a = .000004 * msiemens
const b = 0.0805*nA
const Vr = -70.6*mV
const EE = 0*mV
const EI = -75*mV
const taue = 10*ms
const taui = 3*ms
const taup = 3*ms # talk to brendan (time integration constant for Poisson input units?)
const hard_refrac = 0*ms
const refrac_max = Int(floor(hard_refrac/dt)) # number of time steps in hard refraction

const log_mean =  -5.*10^-5.
const log_sigma = .5

#################################################
##################### THESE DEFINE EQUATIONS
Fw(vm,gE,w,gI,t) = (a.*(vm - EL) - w)./tauw
Fvm(vm,gE,w,gI,gP,t,I)= ( -gL.*(vm-EL) + gL.*DeltaT.*exp.((vm-VT)./DeltaT) - gE.*(vm-EE) - gI.*(vm-EI) - gP.*(vm-EE)  + I - w )./C
FgE(vm,gE,w,gI,t)= -gE./taue
FgI(vm,gE,w,gI,t)= -gI./taui
####################



###################### POISSON INPUT POPULATION 1 is 1 ms
function poisson_spiketrain(firing_rate,dt,input_time,total_time)
	t = dt:dt:total_time;
	spike_train = zeros(1,length(t));
	spikes = [];
	times = 0
	while(times<total_time) # Create Spike Train
		ISI = -log(rand())/(firing_rate/1000) # added 1000 to account for ms time scale
		times = times + ISI
		if (length(spikes) == 0)
			spikes = [spikes;ISI]
		else
			spikes = [spikes;ISI+times]
		end
		if (spikes[end]>total_time)
			spikes = spikes[1:end-1]
		end
	end
	for i = 1:length(spikes)

		index = Int(round(spikes[i]/dt)) # descritize to dt
		if(index == 0 )
			index = 1
		end
		spike_train[index] = 1
	end
	return spike_train;
end
######################

###################### POISSON INPUT POPULATION 1 is 1 ms
function poisson_spiketrain_to_inputstim(spike_train,time_constant,max,dt)
	input = zeros(Int64(sum(spike_train)),length(spike_train))
	starts = find(spike_train);
	for i = 1:length(starts)
		for j = starts[i]:length(spike_train)
			input[i,j] = max*exp.(-(j-starts[i])*dt*time_constant);
		end
	end
	input = sum(input,1);
	return input;
end
######################

###################### POISSON generation of all stimuli
function poisson_stimulis_gen(N,maximum_rate,dt,input_time,time_constant,max,total_time)

	t = dt:dt:total_time;
	stims = zeros(N,length(t))
	for i = 1:N
		train = poisson_spiketrain(maximum_rate*rand(1)[1],dt,input_time,total_time);
		stims[i,:] = poisson_spiketrain_to_inputstim(train,time_constant,max,dt);
	end
	return stims;
end
######################

###################### POISSON network generation ER
function poisson_network(p,N,Ne,mu,sigma)
	network = zeros(N,Ne)
	d = LogNormal(mu,sigma)
	for i = 1:N
		for j = 1:Ne
			if(rand()[1]<p)
				network[i,j] = rand(d,1)[1]
			end
		end
	end
	return network;
end
######################

####################### THIS IS THE NETWORK TYPE DEFINITION
type Network500
	t
	dt
	delay
	N
	vm
	vm_var_e
	vm_var_i
	vm_var
	vm_var_e_pop
	vm_var_i_pop
	vm_var_pop
	vm_var_e_M2
	vm_var_i_M2
	vm_var_M2
	vm_var_e_pop_M2
	vm_var_i_pop_M2
	vm_var_pop_M2
	w
	gE
	gI
	gP
	I
	we
	wi
	vmRecord
	gERecord
	wRecord
	gIRecord
	vmTemp
	gETemp
	wTemp
	gITemp
	tSpike
	i
	refrac
	Network500()=new()
end
######################

###################################################### integration happens here
##############################################

function integrate!(net::Network500,poisson_stim,i,projected)
	#net.gP = [zeros(2000); poisson_stim[:,Int(i)]]
	if(i*dt<current_in)
		#net.I = rand(N)*current_max*nA
		net.gP[projected] = poisson_stim[:,Int(i)]
		net.I = ones(N)*current_max*nA
	else
		net.I = zeros(N)
		net.gP[projected] = net.gP[projected]*exp.(-taup*net.dt);
		# advance via euler's method
	end
	net.refrac[find(net.refrac)] += 1 #increment everything by one time step
	net.refrac[find(x->x>(refrac_max+1),net.refrac)] = 0
	refrac = find(net.refrac)

	temp_gE = net.gE
	temp_gE[refrac] = 0

	temp_gI = net.gI
	temp_gI[refrac] = 0

	temp_gP = net.gP
	temp_gP[refrac] = 0

	temp_I  = net.I
	temp_I[refrac]  = 0

	Δvm=net.dt*Fvm(net.vm,temp_gE,net.w,temp_gI,temp_gP,net.t,temp_I)
	ΔgE=net.dt*FgE(net.vm,net.gE,net.w,net.gI,net.t)
	Δw=net.dt*Fw(net.vm,net.gE,net.w,net.gI,net.t)
	ΔgI=net.dt*FgI(net.vm,net.gE,net.w,net.gI,net.t)
	net.vm+=Δvm
	net.gE+=ΔgE
	net.w+=Δw
	net.gI+=ΔgI
	net.t+=net.dt
	if i == 1
		net.vm_var_e = net.vm[1:Ne]
		net.vm_var_i = net.vm[Ne+1:N]
		net.vm_var = net.vm
		net.vm_var_e_pop = mean(net.vm[1:Ne])
		net.vm_var_i_pop = mean(net.vm[Ne+1:N])
		net.vm_var_pop = mean(net.vm)
	end
	null,net.vm_var_e,net.vm_var_e_M2 = update([i,net.vm_var_e,net.vm_var_e_M2],net.vm[1:Ne])
	null,net.vm_var_i,net.vm_var_i_M2 = update([i,net.vm_var_i,net.vm_var_i_M2],net.vm[Ne+1:N])
	null,net.vm_var,net.vm_var_M2 = update([i,net.vm_var,net.vm_var_M2],net.vm)
	null,net.vm_var_e_pop,net.vm_var_e_pop_M2 = update([i,net.vm_var_e_pop,net.vm_var_e_pop_M2],mean(net.vm[1:Ne]))
	null,net.vm_var_i_pop,net.vm_var_i_pop_M2 = update([i,net.vm_var_i_pop,net.vm_var_i_pop_M2],mean(net.vm[Ne+1:N]))
	null,net.vm_var_pop,net.vm_var_pop_M2 = update([i,net.vm_var_pop,net.vm_var_pop_M2],mean(net.vm))
end

######################## THIS IS REALLY JUST A WRAPPER FOR SIMULATION CALLS ALL SUBPROCESSES
function run!(net::Network500,iter,poisson_stim,projected)
  counter = 0
  for k=1:iter
    #record!(net)
    spike!(net)
    synapse!(net)
    integrate!(net,poisson_stim,k,projected)
    reset!(net)
    if(length(net.i)>(Ne/2))
    	k = iter
    	burst = 1
    	print("QUIT BURSTING")
    	return burst
    end
    if(length(net.i) == 0)
    	counter += 1
    	if (counter*dt > 200)
    		print("dead","\n\n")
    		k = iter
    		return counter
    	end
    else
    	counter = 0
    end
  end
end
################################

###################### DEFINE RECORDING HERE CHANGE INDICIES IF YOU WANT DIFFERENT NEURONS
function record!(net::Network500)
	net.vmRecord=[net.vmRecord net.vm[[1,2,3,4,5,6]]]
	net.gERecord=[net.gERecord net.gE[[1,2,3,4,5,6]]]
	net.wRecord=[net.wRecord net.w[[1,2,3,4,5,6]]]
	net.gIRecord=[net.gIRecord net.gI[[1,2,3,4,5,6]]]
end
######################

################################ here we definie a spike
function spike!(net::Network500)
	isExceedThreshold=net.vm.>Vcut
	isSpike=isExceedThreshold
	net.i=find(isSpike)
	for spikeNeuron in net.i
		net.tSpike[spikeNeuron]=[net.tSpike[spikeNeuron];net.t]
	end
	net.refrac[net.i] = 1# set counter to 1 for recently spike neurons
end
##################################

########################## this is a reset for a spike
function reset!(net::Network500)
	net.vm[net.i]=Vr
	net.w[net.i]+=b
end
###########################

########################## this is a synapse defintion, how we update after spike!
function synapse!(net::Network500)
	net.gE+=sum(net.we[:,net.i],2)
	net.gI+=sum(net.wi[:,net.i],2)
end
##########################

#################################################################################################################################################
######################################################################## SCORE AND OPT ##########################################################
#################################################################################################################################################

########################## convert multi trial spike train to ISI list

function score_time_ignited(spike_list_set) #this does not allow for bursting
	max_vals = zeros(length(spike_list_set))
	for i = 1:length(spike_list_set)[1]
		if(isempty(spike_list_set[i]))
			max_vals[i] = 0
		else
			max_vals[i] = maximum(spike_list_set[i])
		end
	end

	return maximum(max_vals-stim_in)/(run_total-stim_in)
end

function clustering_coef(adj_mat)
#=
	if(isempty(adj_mat))
		return NaN
	end
	if(size(adj_mat)[1] > 4000)
		return NaN
	end

	temp = adj_mat
	temp[find(temp)] = 1
	temp = Bool.(temp)
	triangles = 0
	dim = size(adj_mat)[1]
	print(dim,"\n\n")

	for i = 1:dim
		for j = 1:dim
			for k = 1:dim
				if((temp[i,j]==true) & (temp[j,k]==true) & (temp[i,k]==true))
					triangles += 1
				end
			end
		end
	end

	paths = sum(temp^2)-trace(temp^2)
	return triangles/paths
=#

	if(isempty(adj_mat))
		return NaN
	end

	dim = size(adj_mat)[1]

	#temp = (adj_mat).^(1/3)
	temp = (adj_mat).^(1/3)

	d_in = temp'*ones(dim)
	d_out = temp*ones(dim)
	d_bi = diag(temp^2)
	d_tot = d_in+d_out
	all_motifs_CC = diag(((temp+temp')^3))./(2*(d_tot.*(d_tot-1)-2*d_bi))
	return mean(all_motifs_CC)
end

function cluster_over_time(spike_list_set,adj_mat;plot=false)
	bins = 2 # 1ms
	time_starts  = 0:bins:(run_total-bins) #### estimate rate parameter
	time_ends    = bins:bins:run_total
	cluster_vals = zeros(length(time_starts))
	for i = 1:length(time_starts)
		active = []
		for j = 1:N
			if(~isempty(find(x-> (x>time_starts[i]) & (x<time_ends[i]), spike_list_set[j])))
				active = [active; j]
			end
		end
		cluster_vals[i] = clustering_coef(adj_mat[active,active])
	end
	cluster_overall = clustering_coef(adj_mat)
	if(plot)
		figure()
		PyPlot.plot((time_starts+time_ends)/2,cluster_vals)
		figure()
		PyPlot.plot((time_starts+time_ends)/2,cluster_vals/cluster_overall)
	end
	return cluster_vals
end

function clustering_doiron_range(range_vals)
	cluster_vals_ee = zeros(length(range_vals))
	cluster_Vals_ei = zeros(length(range_vals))
	p_ie = .31
	p_ei = .22
	p_ee = .2
	cluster = 50

	for i = 1:length(range_vals)

		R = range_vals[i]

		p_ee_out = (p_ee*cluster)/(R+cluster-1)
		p_ee_in  = p_ee_out*R
		p_ei_out =  (p_ei*cluster)/(R+cluster-1)
		p_ei_in  =  p_ei_out*R

		a = doiron_hetero(Ne, Ne, p_ee_in, p_ee_out, 1,1,cluster)
		#d = [doiron_hetero(Ni, Ne, p_ei_in,p_ei_out,1,1,50)	zeros(Ni,Ni); zeros(Ne,Ni)]
		cluster_vals_ee[i] = clustering_coef(a)
		print(i)
		print("\n")
	end
	figure()
	PyPlot.plot(range_vals,cluster_vals_ee)
end


function combinadic(n, k, m)
    #represent n with 3
    indices = []
    #if n < 1 || k < 1
    #    return indices
    #
    print(n,"\t",k,"\t",m,"\n")
    n_choose_k = binomial(n, k)
    if m >= n_choose_k
        return indices
    end
    m = n_choose_k - 1 - m
    guess = n - 1
    for i = k:-1:1
        take = binomial(guess, i)
        while take > m
            guess = guess - 1
            take = binomial(guess, i)
        end
        m = m - take
        push!(indices, n - 1 - guess)
    end
    return indices
end


function is_isomorphic(adj_mat1,adj_mat2)
	#this assumes 3x3 which works becasue thats all we care about
	isomorphic = false
	mat1 = [1 0 0; 0 0 1; 0 1 0]
	mat2 = [0 1 0; 1 0 0; 0 0 1]
	mat3 = [0 0 1; 1 0 0; 0 1 0]
	mat4 = [0 0 1; 0 1 0; 1 0 0]
	mat5 = [0 1 0; 0 0 1; 1 0 0]
	if(adj_mat1 == adj_mat2)
		is_isomorphic = true
	elseif(adj_mat1 == mat1*adj_mat2*mat1')
		is_isomorphic = true
	elseif(adj_mat1 == mat2*adj_mat2*mat2')
		is_isomorphic = true
	elseif(adj_mat1 == mat3*adj_mat2*mat3')
		is_isomorphic = true
	elseif(adj_mat1 == mat4*adj_mat2*mat4')
		is_isomorphic = true
	elseif(adj_mat1 == mat5*adj_mat2*mat5')
		is_isomorphic = true
	end
	return isomorphic
end

function motif_identity(adj_mat)
	# this uses the convention found in song et al 2005
	edges_num = sum(adj_mat)
	edges     = find(adj_mat)
	out_deg   = sum(adj_mat,1)
	in_deg    = sum(adj_mat,2)
	if(edges_num == 0)
		motif = 1
	elseif(edges_num == 1)
		motif = 2
	elseif(edges_num == 5)
		motif = 15
	elseif(edges_num == 6)
		motif = 16
	elseif(edges_num == 2)
		if((edges == [2;4]) | (edges == [3;7]) | (edges == [6;8])) # plain bi directional
			motif = 3
		elseif(  (out_deg == [2 0 0]) | (out_deg == [0 2 0]) | (out_deg == [0 0 2]))
			motif = 4
		elseif(  (in_deg == [2 0 0]) | (in_deg == [0 2 0]) | (in_deg == [0 0 2]))
			motif = 5
		else
			motif = 6
		end
	elseif(edges_num == 3)
		if(  (out_deg == [1 1 1]) & (in_deg == [1 1 1]) )
			motif = 11
		elseif( (out_deg == [1 1 1]) )
			motif = 7
		elseif( (in_deg == [1 1 1]) )
			motif = 8
		else
			motif = 10
		end
	else
		if(  (in_deg == [0 2 2]) | (in_deg == [2 0 2]) | (in_deg == [2 2 0]) )
			motif = 12
		elseif(  (out_deg == [0 2 2]) | (out_deg == [2 0 2]) | (out_deg == [2 2 0]) )
			motif = 14
		elseif(is_isomorphic(adj_mat,[0 1 0; 1 0 1; 0 1 0]))
			motif = 9
		else
			motif = 13
		end
	end
	return motif
end

function motif_baseline(adj_mat)
	motif_counts = zeros(16) # 16 motifs
	num_samples = 10000
	dim = size(adj_mat)[1]
	motif_size = 3
	if(binomial(dim,motif_size)<num_samples)
		num_samples = binomial(dim-1,motif_size)
	end
	subset = sample(1:binomial(dim-1, motif_size),num_samples,replace=false)
	for i in 1:num_samples
		combo = combinadic(dim-1, motif_size, subset[i])+[1;1;1] # extra ones account for one base indexing
		subgraph = adj_mat[combo,combo]
		motif_counts[motif_identity(subgraph)] +=1
	end
	motif_counts = motif_counts/sum(motif_counts) #normalize
	return motif_counts
end



function motifs_overtime(spike_list_set,time_start,time_end,bins,adj_mat)
	time_starts  = time_start:bins:(time_end-bins) #### estimate rate parameter
	time_ends    = time_start+bins:bins:time_end
	motif_vals = zeros(length(time_starts),16)
	normalize_motif_vals = zeros(length(time_starts),16)
	for i = 1:length(time_starts)
		active = []
		for j = 1:N
			if(~isempty(find(x-> (x>time_starts[i]) & (x<time_ends[i]), spike_list_set[j])))
				active = [active; j]
			end

		end
		print(length(active))
		print("\n")
		motif_vals[i,:] = motif_baseline(adj_mat[active,active])
	end
	motif_overall = motif_baseline(adj_mat)
	for i = 1:16
		normalize_motif_vals[:,i] = motif_vals[:,i]/motif_overall[i]
	end
	ts = (time_starts+time_ends)/2
	return motif_vals, normalize_motif_vals,ts
end

function clustered_motifs(adj_mat) # revision as of 20180728

	if(isempty(adj_mat)) # with the way calculating recruitment, unlikely to be the case.
		return NaN,NaN,NaN,NaN,NaN
	elseif(isempty(find(adj_mat.!=0)))
		return NaN,NaN,NaN,NaN,NaN
	end

	W = adj_mat'; # Wij convention in Fagiolo is the transpose of ours

	dim = size(adj_mat)[1];

	if(!isempty(find(W.!=0)))
		temp = (W).^(1/3);
		d_in = zeros(dim,);
		d_out = zeros(dim,);
		d_bi = zeros(dim,);
		for ii = 1:dim # find in, out, and bi-degrees for each unit in the network
			in_nodes = find(W[:,ii].!=0); # the units that project in to ii
			out_nodes = find(W[ii,:].!=0); # the units that ii projects out to
			bi_nodes = intersect(in_nodes,out_nodes); # bidirectional ones
			d_in[ii] = length(in_nodes);
			d_out[ii] = length(out_nodes);
			d_bi[ii] = length(bi_nodes);
		end

		#=d_in = W'*ones(dim);
		d_out = adj_mat'*ones(dim);
		d_bi = diag(adj_mat);=#
		d_tot = d_in+d_out;

		denom = (d_in.*d_out-d_bi);

		# possible that for individual units, there would be 0's for degrees and thus for denom.
		# this would lead to NaNs

		cycle_CC = diag(W^3)./denom;
		middleman_CC = diag(W*W'*W)./denom;
		# thus for some units, we would have NaNs for cycle_CC and middleman_CC in a timebin
		# if the out or in degree is only 1, then it would be moot as well
		fanin_CC = diag(W'*W^2)./(d_in.*(d_in-1));
		fanout_CC = diag(W^2*W')./(d_out.*(d_out-1));

		all_motifs_CC = diag((W+W')^3)./(2*(d_tot.*(d_tot-1)-2*d_bi));

		#=cycle_CC  = diag(temp_squared*temp)./denom;
		middleman_CC = diag(temp*temp_prime*temp)./denom;
		fanin_CC  = diag(temp_prime*temp_squared)./(d_in.*(d_in-1));
		fanout_CC  = diag(temp_squared*temp_prime)./(d_out.*(d_out-1));
		all_motifs_CC = diag(((temp+temp_prime)^3))./(2*(d_tot.*(d_tot-1)-2*d_bi));
		=#
	end


	# perhaps we in fact do not want to set NaNs to zeros - we want to leave them out of population mean calculation
	# but this also wouldn't be the reason we're getting 0's for whole-timesteps.
#=
	for ii = 1:size(cycle_CC)[1]
		if isnan(cycle_CC[ii])
			# could be NaN, for example, if the denom is zero, that is no in or out-degrees for unit ii
			cycle_CC[ii] = 0;
		end
		if isnan(middleman_CC[ii])
			middleman_CC[ii] = 0;
		end
		if isnan(fanin_CC[ii])
			fanin_CC[ii] = 0;
		end
		if isnan(fanout_CC[ii])
			fanout_CC[ii] = 0;
		end
		if isnan(all_motifs_CC[ii])
			all_motifs_CC[ii] = 0;
		end
	end
=#
	# we want to return NaNs for some units
	return cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC
end


function clustered_motifs_overtime(spike_list_set,adj_mat;bins=2,plot=true)#5ms)
	time_starts  = 0:bins:(run_total-bins) #### estimate rate parameter
	time_ends    = bins:bins:run_total

	cycle_cc_over_time = zeros(length(time_starts))
	middleman_cc_over_time = zeros(length(time_starts))
	fanin_cc_over_time  = zeros(length(time_starts))
	fanout_cc_over_time = zeros(length(time_starts))
	all_motifs_cc_over_time = zeros(length(time_starts))

	for i = 1:length(time_starts)
		active = []
		for j = 1:N
			if(~isempty(find(x-> (x>time_starts[i]) & (x<time_ends[i]), spike_list_set[j]) ) )
				active = [active; j]
			end

		end
		cyc, mid, fan_in, fan_out = clustered_motifs(adj_mat[active,active])
		cycle_cc_over_time[i] = cyc
		middleman_cc_over_time[i]= mid
		fanin_cc_over_time[i]  = fan_in
		fanout_cc_over_time[i] = fan_out

	end
	#motif_overall = motif_baseline(adj_mat)
	#for i = 1:16
	#	normalize_motif_vals[:,i] = motif_vals[:,i]/motif_overall[i]
	#end
	ts = (time_starts+time_ends)/2
	if(plot)
		figure()
		PyPlot.plot(ts,cycle_cc_over_time,color="blue")
		figure()
		PyPlot.plot(ts,middleman_cc_over_time,color="red")
		figure()
		PyPlot.plot(ts,fanin_cc_over_time,color="orange")
		figure()
		PyPlot.plot(ts,fanout_cc_over_time,color="green")
	end

	return cycle_cc_over_time, middleman_cc_over_time,fanin_cc_over_time,fanin_cc_over_time
end

function binary_raster_gen(spike_set,discretize_dt)
	bins = 0:discretize_dt:run_total
	raster = zeros(N,length(bins)-1)
	conversion_factor = discretize_dt/run_total
	for i = 1:N
		discrete_train = unique(round.(spike_set[i]/discretize_dt))
		discrete_train = discrete_train[discrete_train.>0]

		for j = 1:length(discrete_train)
			raster[i,Int(discrete_train[j])] = 1
		end
	end
	return raster
end

function shuffle_null_raster_gen(spike_set,discretize_dt)
	bins = 0:discretize_dt:run_total
	raster = zeros(N,length(bins)-1)
	conversion_factor = discretize_dt/run_total
	shuffle_null = randperm(N)
	for i = 1:N
		discrete_train = unique(round.(spike_set[shuffle_null[i]]/discretize_dt))
		discrete_train = discrete_train[discrete_train.>0]

		for j = 1:length(discrete_train)
			raster[i,Int(discrete_train[j])] = 1
		end
	end
	return raster
end

function poisson_raster_gen(discretize_dt,rate_e,rate_i)
	bins = 0:discretize_dt:run_total
	raster = zeros(N,length(bins)-1)
	for i = 1:Ne
		for j = 1:length(bins)-1
			if rand() < (discretize_dt * rate_e/1000)
				raster[i,j] = 1
			end
		end
	end
	for i = (Ne+1):N
		for j = 1:length(bins)-1
			if rand() < (discretize_dt * rate_i/1000)
				raster[i,j] = 1
			end
		end
	end
	return raster
end

function motif_functional_graphs(graph,raster,discretize_dt)

	#connections = find(graph) # find locations with weights
	#graph[connections] = 1 # make graph binary

	cycle_over_time = zeros(size(raster)[2]-1)
	middleman_over_time = zeros(size(raster)[2]-1)
	fanin_over_time = zeros(size(raster)[2]-1)
	fanout_over_time = zeros(size(raster)[2]-1)
	all_motifs_over_time = zeros(size(raster)[2]-1)

	for i = 1:(size(raster)[2]-1)
		print(i,"\n\n")
		functional_graph = zeros(N,N)
		bin_1 = raster[:,i]
		bin_2 = raster[:,i+1]

		pre_spike = find(bin_1)
		post_spike = find(bin_2)

		for j = 1:length(pre_spike)
			inds = intersect(post_spike,find(graph[:,j]))
			functional_graph[inds ,j] = 1
		end

		valid_nodes = intersect(find(sum(functional_graph,1)),find(sum(functional_graph,2)))
		functional_graph = functional_graph[valid_nodes,valid_nodes]
		cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = clustered_motifs(functional_graph)
		#all_motifs_CC = clustered_motifs(functional_graph)
		cycle_over_time[i] = mean(cycle_CC)
		middleman_over_time[i] = mean(middleman_CC)
		fanin_over_time[i] = mean(fanin_CC)
		fanout_over_time[i] = mean(fanout_CC)
		all_motifs_over_time[i] = mean(all_motifs_CC)
		#print(cycle_CC,"\t",middleman_CC,"\t",fanin_CC,"\t",fanout_CC,"\t",all_motifs_CC,"\n\n")


	end

	fanin_over_time_safe = fanin_over_time[find(!isnan,fanin_over_time)]
	fanout_over_time_safe = fanout_over_time[find(!isnan,fanout_over_time)]
	middleman_over_time_safe = middleman_over_time[find(!isnan,middleman_over_time)]
	cycle_over_time_safe = cycle_over_time[find(!isnan,cycle_over_time)]
	all_motifs_over_time_safe = all_motifs_over_time[find(!isnan,all_motifs_over_time)]

	spl1 = Spline1D(find(!isnan,fanin_over_time).*discretize_dt, fanin_over_time_safe; w=ones(length(fanin_over_time_safe)), k=3, bc="nearest", s=0.0)
	spl2 = Spline1D(find(!isnan,fanout_over_time).*discretize_dt, fanout_over_time_safe; w=ones(length(fanout_over_time_safe)), k=3, bc="nearest", s=0.0)
	spl3 = Spline1D(find(!isnan,middleman_over_time).*discretize_dt, middleman_over_time_safe; w=ones(length(middleman_over_time_safe)), k=3, bc="nearest", s=0.0)
	spl4 = Spline1D(find(!isnan,cycle_over_time).*discretize_dt, cycle_over_time_safe; w=ones(length(cycle_over_time_safe)), k=3, bc="nearest", s=0.0)
	spl5 = Spline1D(find(!isnan,all_motifs_over_time).*discretize_dt, all_motifs_over_time_safe; w=ones(length(all_motifs_over_time_safe)), k=3, bc="nearest", s=0.0)

	fanin_interp_inds = find(isnan,fanin_over_time)
	for i in 1:length(fanin_interp_inds)
		fanin_over_time[fanin_interp_inds[i]] = Dierckx.evaluate(spl1, discretize_dt*fanin_interp_inds[i])
	end

	fanout_interp_inds = find(isnan,fanout_over_time)
	for i in 1:length(fanout_interp_inds)
		fanout_over_time[fanout_interp_inds[i]] = Dierckx.evaluate(spl2, discretize_dt*fanout_interp_inds[i])
	end

	middleman_interp_inds = find(isnan,middleman_over_time)
	for i in 1:length(middleman_interp_inds)
		middleman_over_time[middleman_interp_inds[i]] = Dierckx.evaluate(spl3, discretize_dt*middleman_interp_inds[i])
	end

	cycle_interp_inds = find(isnan,cycle_over_time)
	for i in 1:length(cycle_interp_inds)
		cycle_over_time[cycle_interp_inds[i]] = Dierckx.evaluate(spl4, discretize_dt*cycle_interp_inds[i])
	end

	all_motifs_interp_inds = find(isnan,all_motifs_over_time)
	for i in 1:length(all_motifs_interp_inds)
		all_motifs_over_time[all_motifs_interp_inds[i]] = Dierckx.evaluate(spl5, discretize_dt*all_motifs_interp_inds[i])
	end

	safe_in = find(!isnan,fanin_over_time)
	safe_out = find(!isnan,fanout_over_time)
	safe_mid = find(!isnan,middleman_over_time)
	safe_cycle = find(!isnan,cycle_over_time)
	safe_all_motifs = find(!isnan,all_motifs_over_time)

	fanin_over_time = fanin_over_time[safe_in[1]:safe_in[end]]
	fanout_over_time = fanout_over_time[safe_out[1]:safe_out[end]]
	middleman_over_time = middleman_over_time[safe_mid[1]:safe_mid[end]]
	cycle_over_time = cycle_over_time[safe_cycle[1]:safe_cycle[end]]
	all_motifs_over_time = all_motifs_over_time[safe_all_motifs[1]:safe_all_motifs[end]]

	print("fanin mean\t",mean(fanin_over_time),"\n\n")
	print("fanout mean\t",mean(fanout_over_time),"\n\n")
	print("mid mean\t",mean(middleman_over_time),"\n\n")
	print("cycle mean\t",mean(cycle_over_time),"\n\n")
	print("all_motifs mean\t",mean(all_motifs_over_time),"\n\n")

	fanin_over_time = fanin_over_time - mean(fanin_over_time)
	fanout_over_time = fanout_over_time - mean(fanout_over_time)
	middleman_over_time = middleman_over_time - mean(middleman_over_time)
	cycle_over_time = cycle_over_time - mean(cycle_over_time)
	all_motifs_over_time = all_motifs_over_time - mean(all_motifs_over_time)


	#figure()
	#PyPlot.plot(fanin_over_time,fanout_over_time)
	#xlabel("fanin")
	#ylabel("fanout")

	#figure()
	#PyPlot.plot(middleman_over_time,fanout_over_time)
	#xlabel("middleman")
	#ylabel("fanout")

	#figure()
	#PyPlot.plot(fanin_over_time,middleman_over_time)
	#xlabel("fanin")
	#ylabel("fanout")

	#figure()
	#PyPlot.plot3D(fanin_over_time,fanout_over_time,middleman_over_time,linewidth=.01)
	#xlabel("fanin")
	#ylabel("fanout")
	#zlabel("middleman")

	#figure()
	#PyPlot.plot(fanin_over_time)
	#ylabel("fanin")

	#figure()
	#PyPlot.plot(fanout_over_time)
	#ylabel("fanout")

	#figure()
	#PyPlot.plot(middleman_over_time)
	#ylabel("middleman")

	fanin_fft = fft(fanin_over_time)
	fanin_fft = fanin_fft[1:Int(floor(length(fanin_fft)/2))]
	fanin_phase = atan2.(imag(fanin_fft),real(fanin_fft))
	fanin_components = sqrt.(imag(fanin_fft).^2+real(fanin_fft).^2)

	fanout_fft = fft(fanout_over_time)
	fanout_fft = fanout_fft[1:Int(floor(length(fanout_fft)/2))]
	fanout_phase = atan2.(imag(fanout_fft),real(fanout_fft))
	fanout_components = sqrt.(imag(fanout_fft).^2+real(fanout_fft).^2)

	middleman_fft = fft(middleman_over_time)
	middleman_fft = middleman_fft[1:Int(floor(length(middleman_fft)/2))]
	middleman_phase = atan2.(imag(middleman_fft),real(middleman_fft))
	middleman_components = sqrt.(imag(middleman_fft).^2+real(middleman_fft).^2)

	cycle_fft = fft(cycle_over_time)
	cycle_fft = cycle_fft[1:Int(floor(length(cycle_fft)/2))]
	cycle_phase = atan2.(imag(cycle_fft),real(cycle_fft))
	cycle_components = sqrt.(imag(cycle_fft).^2+real(cycle_fft).^2)

	all_motifs_fft = fft(all_motifs_over_time)
	all_motifs_fft = middleman_fft[1:Int(floor(length(all_motifs_fft)/2))]
	all_motifs_phase = atan2.(imag(all_motifs_fft),real(all_motifs_fft))
	all_motifs_components = sqrt.(imag(all_motifs_fft).^2+real(all_motifs_fft).^2)

	freqs = zeros(length(middleman_fft))
	Fs = 1/(discretize_dt/1000)
	for n = 1:length(middleman_fft)
		freqs[n] = (n-1) * Fs / length(middleman_fft)
	end




	#figure()
	#PyPlot.plot(fanin_components,fanout_components)
	#xlabel("fanin_freq")
	#ylabel("fanout_freq")
	#figure()
	#PyPlot.plot(middleman_components,fanout_components)
	#xlabel("middleman_freq")
	#ylabel("fanout_freq")
	#figure()
	#PyPlot.plot(fanin_components,middleman_components)
	#xlabel("fanin_freq")
	#ylabel("fanout_freq")
	#figure()
	#PyPlot.plot3D(fanin_components,fanout_components,middleman_components,linewidth=.01)
	#xlabel("fanin_freq")
	#ylabel("fanout_freq")
	#zlabel("middleman_freq")

	figure()
	PyPlot.plot(freqs,fanin_components,linewidth = .5)
	xlabel("freq fanin")

	figure()
	PyPlot.plot(freqs,fanout_components, linewidth = .5)
	xlabel("freq fanout")

	figure()
	PyPlot.plot(freqs,middleman_components, linewidth = .5)
	xlabel("freq middleman")

	figure()
	PyPlot.plot(freqs,cycle_components, linewidth = .5)
	xlabel("freq cyc")

	figure()
	PyPlot.plot(freqs,all_motifs_components, linewidth = .5)
	xlabel("freq all")

	figure()
	PyPlot.loglog(freqs,fanin_components,linewidth = .5)
	xlabel("freq fanin log log")

	figure()
	PyPlot.loglog(freqs,fanout_components, linewidth = .5)
	xlabel("freq fanout log log")

	figure()
	PyPlot.loglog(freqs,middleman_components, linewidth = .5)
	xlabel("freq middleman log log")

	figure()
	PyPlot.loglog(freqs,cycle_components, linewidth = .5)
	xlabel("freq cyc log log")

	figure()
	PyPlot.loglog(freqs,all_motifs_components, linewidth = .5)
	xlabel("freq all log log")

	#figure()
	#PyPlot.plot(fanin_phase,fanout_phase)
	#xlabel("fanin_phase")
	#ylabel("fanout_phase")
	#figure()
	#PyPlot.plot(middleman_phase,fanout_phase)
	#xlabel("middleman_phase")
	#ylabel("fanout_phase")
	#figure()
	#PyPlot.plot(fanin_phase,middleman_phase)
	#xlabel("fanin_phase")
	#ylabel("fanout_phase")
	#figure()
	#PyPlot.plot3D(fanin_phase,fanout_phase,middleman_phase,linewidth=.01)
	#xlabel("fanin_phase")
	#ylabel("fanout_phase")
	#zlabel("middleman_phase")

	#figure()
	#PyPlot.plot(fanin_phase)
	#xlabel("fanin_phase")
	#figure()
	#PyPlot.plot(fanout_phase)
	#xlabel("fanout_phase")
	#figure()
	#PyPlot.plot(fanout_phase)
	#xlabel("middleman_phase")
	return 1
end



function raster_gen(spike_list_set,mat)
	num_steps = Int(floor(run_total/dt))
	raster = zeros(num_steps,N)
	e_exp  = exp(-(0:dt:30)/taue)
	i_exp  = exp(-(0:dt:30)/taui)
	for i = 1:Ne
		#weight = sum(mat[i,:])
		spikes = spike_list_set[i]
		for j = 2:length(spikes)
			if((Int(floor(spikes[j]/dt))+length(e_exp)-1)<=num_steps)
				raster[Int(floor(spikes[j]/dt)):Int(floor(spikes[j]/dt))+length(e_exp)-1,i] += e_exp#weight*e_exp
			else
				raster[Int(floor(spikes[j]/dt)):end,i] += e_exp[1:length(raster[Int(floor(spikes[j]/dt)):end,i])]
			end
		end
	end

	for i = Ne+1:N
		#weight = sum(mat[i,:])
		spikes = spike_list_set[i]
		for j = 2:length(spikes)
			if((Int(floor(spikes[j]/dt))+length(i_exp)-1)<=num_steps)
				raster[Int(floor(spikes[j]/dt)):Int(floor(spikes[j]/dt))+length(e_exp)-1,i] += i_exp#weight*i_exp
			else
				raster[Int(floor(spikes[j]/dt)):end,i] += i_exp[1:length(raster[Int(floor(spikes[j]/dt)):end,i])]
			end
		end
	end
	raster = raster'
	figure()
	pcolormesh(raster)
	figure()
	pcolormesh(mat)
	return raster
end



function rate_est(spike_set) # similar to rate est score, but uses last spike
	num_spikes = 0
	max_vals = zeros(Ne)

	for i = 1:Ne
		post_stim  = (spike_set[i])[find(spike_set[i] .> stim_in)] # only count spikes after stimulis
		if(isempty(post_stim))
			max_vals[i] = 0
		else

			max_vals[i] = maximum(post_stim)
		end
		num_spikes += length(post_stim)
	end
	max_vals = maximum(max_vals)-stim_in # ignore stimulis input period
	spikes_per = (num_spikes/Ne)
	rate_e = 1000*spikes_per/max_vals#divide by 1000 bc ms conversion

	num_spikes = 0
	max_vals = zeros(Ni)
	for i = 1:Ni
		post_stim   = (spike_set[Ne+i])[find(spike_set[Ne+i] .> stim_in)] # only count spikes after stimulis
		if(isempty(post_stim))
			max_vals[i] = 0
		else
			max_vals[i] = maximum(post_stim)
		end
		num_spikes += length(post_stim)
	end
	max_vals = maximum(max_vals)-stim_in # ignore stimulis input period
	spikes_per = (num_spikes/Ni)
	rate_i = 1000*spikes_per/max_vals#divide by 1000 bc ms conversion

	if(rate_e < 0)
		rate_e = -1
	elseif(rate_i < 0)
		rate_i = -1
	end
	return rate_e,rate_i
end

function rate_est_active(spike_set) # same as above, but only using the units which were active
	num_spikes = 0
	max_vals = zeros(Ne)

	active_e = Ne;
	active_i = Ni;
	# which we will decrement every time we find a neuron isempty(post_stim)

	for i = 1:Ne
		post_stim  = (spike_set[i])[find(spike_set[i] .> stim_in)] # only count spikes after stimulis
		if(isempty(post_stim))
			max_vals[i] = 0
			active_e = active_e - 1; # one less active excitatory unit
		else
			max_vals[i] = maximum(post_stim)
		end
		num_spikes += length(post_stim)
	end
	max_vals = maximum(max_vals)-stim_in # ignore stimulis input period
	spikes_per = (num_spikes/active_e)
	rate_e = 1000*spikes_per/max_vals #divide by 1000 bc ms conversion

	num_spikes = 0
	max_vals = zeros(Ni)
	for i = 1:Ni
		post_stim   = (spike_set[Ne+i])[find(spike_set[Ne+i] .> stim_in)] # only count spikes after stimulis
		if(isempty(post_stim))
			max_vals[i] = 0
			active_i = active_i - 1; # one less active inhibitory unit
		else
			max_vals[i] = maximum(post_stim)
		end
		num_spikes += length(post_stim)
	end
	max_vals = maximum(max_vals)-stim_in # ignore stimulis input period
	spikes_per = (num_spikes/active_i)
	rate_i = 1000*spikes_per/max_vals#divide by 1000 bc ms conversion

	if(rate_e < 0)
		rate_e = -1
	elseif(rate_i < 0)
		rate_i = -1
	end
	return rate_e,rate_i
end

@everywhere function synch_score(spikes_1::Array{Float64},spikes_2::Array{Float64})
	tau = 5.0::Float64 #ms
	cut = 50.0::Float64 # ms, point at which we stop considering val, it becomes small

	synch_vals_1 = zeros(Float64,Int((run_total+cut)/dt))
	synch_vals_2 = zeros(Float64,Int((run_total+cut)/dt))
	kernel = sqrt(2.0/tau)*exp.(-(0.0:dt:cut)/tau)::Array{Float64}

	ind1 = Int64.(ceil.(spikes_1/dt))
	ind2 = Int64.(ceil.(spikes_2/dt))
	ind1 = ind1[ind1.>stim_in .& (ind1.<stim_in + 100.0)]
	ind2 = ind2[ind2.>stim_in .& (ind2.<stim_in + 100.0)]

	for i = 1:length(ind1)
		for j = 0:length(kernel)-1
			@inbounds @fastmath synch_vals_1[ind1[i]+j] += kernel[j+1]
		end
	end

	for i = 1:length(ind2)
		for j = 0:length(kernel)-1
			@inbounds @fastmath synch_vals_2[ind2[i]+j] += kernel[j+1]
		end
	end

	score = (synch_vals_1-synch_vals_2).^2
	score = sum(score*dt)
	return score
end


function population_synch_score_e_i(spike_set::Array{Array{Float64,1},1})
	score_e = 0.0::Float64
	score_i = 0.0::Float64
	score_e_full = 0.0::Float64
	score_i_full = 0.0::Float64

	for i = 1:(Ne-1) # do excitatory
		print(i,"\n")
		score_e = @parallel (+) for j = (i+1):Ne
			synch_score(spike_set[i],spike_set[j])
		end
		score_e_full = score_e_full + score_e
	end

	score_e_full = score_e_full*(2/(Ne*(Ne-1)))

	for i = (Ne+1):N
		print(i,"\n")
		score_i = @parallel (+) for j = i:N
			synch_score(spike_set[i],spike_set[j])
		end
		score_i_full = score_i_full + score_i
	end

	score_i_full = score_i_full*(2/(Ni*(Ni-1)))

	return score_e_full, score_i_full
end

function psth(spike_set,bin_width,time_total;plot=false)
    time_start = 0:bin_width:(time_total-bin_width)
    time_ends  = time_start + bin_width
    psth_data_e = zeros(length(time_start),Ne)
    psth_data_i = zeros(length(time_start),Ni)
    for i = 1:length(time_start)
        for j = 1:Ne
            psth_data_e[i,j] = length(find(x-> (x>time_start[i]) & (x<time_ends[i]),spike_set[j]))
        end
        for j = 1:Ni
            psth_data_i[i,j] = length(find(x-> (x>time_start[i]) & (x<time_ends[i]),spike_set[j+Ne]))
        end
    end

    psth_data_e = mean(psth_data_e,2)/(bin_width/1000)
    psth_data_i = mean(psth_data_i,2)/(bin_width/1000)
    if(plot)
        figure()
        PyPlot.plot((time_start+time_ends)/2,psth_data_e,color="blue")
        PyPlot.plot((time_start+time_ends)/2,psth_data_i,color="red")
    end
    return psth_data_e,psth_data_i
end

function branching_score(spike_set;plot=false)
	window = .1 # 4 millisecond windows
	time_start = stim_in:window:(run_total-window)
    time_ends  = time_start + window
    avalanche_data = zeros(length(time_start))
    for i = 1:length(time_start)
    	spike_count = 0
    	for j = 1:N
    		spikes = length(find(x-> (x>time_start[i]) & (x<time_ends[i]),spike_set[j]))
    		spike_count += spikes
    	end
    	avalanche_data[i] = spike_count
    end
    avalanche_sizes = []
    event = false
    event_size = 0
    for i = 1:(length(avalanche_data)-1)

    	if((avalanche_data[i]==0) & ~event)
    		event = true
    		event_size = 0
    	end

    	if(event & (avalanche_data[i] != 0))
    		event_size += 1
    	end
    	if(event & (avalanche_data[i] == 0))
    		event = false
    		append!(avalanche_sizes,event_size)
    	end
    end
    print("\n\n\n",avalanche_sizes)
    return length(avalanche_sizes)
end

function update(existingAggregate, newValue)
    count_online, mean_online, M2 = existingAggregate
    count_online = count_online + 1
    delta = newValue .- mean_online
    mean_online = mean_online .+ delta ./ count_online
    delta2 = newValue .- mean_online
    M2 = M2 .+ delta .* delta2
    return count_online, mean_online, M2
end

function finalize_var(existingAggregate)
    count_online, mean_online, M2 = existingAggregate
    variance =  M2/count_online
    sampleVariance = M2/(count_online-1)
    if count_online < 2
        return NaN
    else
        return mean_online, variance, sampleVariance
    end
end

function branching_param(tSpike,W,bin)
	# edited as of 20180816 to be correct!
	nbins = convert(Int64,run_total/bin);
	spike_set = [];
	active_inds = [];
	for ii = 1:length(tSpike)
		post_stim = (tSpike[ii])[find(tSpike[ii].>stim_in)];
		if !isempty(post_stim)
			push!(spike_set,tSpike[ii]);
			push!(active_inds,ii);
		end
	end

	e_edges = find(active_inds.<=Ne);
	i_edges = find(active_inds.>Ne);

	Nactive = length(spike_set);
	W_active = W[active_inds,active_inds];

	units_bscore = zeros(Nactive,);

	Spikes_binned = zeros(Nactive,nbins);
	# don't shuffle the correspondence of the spikes
	for ii = 1:Nactive
		# find the indices for neuron ii for which we'll put a 1 in Spikes_binned
		if !isempty(spike_set[ii,])
			indices = floor.(spike_set[ii]);
			indices = indices[find(indices.<=nbins)];
			indices = indices[find(indices.>0)]; # get rid of the first zero
			for jj = 1:length(indices)
				Spikes_binned[ii,convert(Int64,indices[jj])] = 1;
			end
		end
	end

	span = collect(Int64,5/bin:20/bin);

	# for every group of subsequent timesteps, determine the number of ancestors and the number of descendants
	numA = zeros(nbins-maximum(span)); # number of ancestors for each bin
	numD = zeros(nbins-maximum(span)); # number of descendants for each ancestral bin
	d = zeros(nbins-maximum(span)); # the number of electrode descendants per ancestor
	for tt = 1:length(numA)
		# that is, we look at time tt for nA and time tt+5ms:tt+20ms for nD
		# this is going to be a little hairy...
		numA[tt] = length(find(Spikes_binned[:,tt].==1));
		numD[tt] = length(find(Spikes_binned[:,tt+span]));
		d[tt] = round.(numD[tt]/numA[tt]); # the ratio of descendants to ancestors
	end

	dratio = filter(!isnan,d);
	filter!(!isinf,dratio);
	#filter!(!iszero,dratio);
	dratio = unique(dratio);
	sort!(dratio);
	pd = zeros(length(dratio),);
	Na = sum(numA); # the total number of ancestors over all bins
	#norm = (Nactive-1)/(Nactive - Na); # correction for refractoriness in next time bin (we do not have refractoriness)

#=
	uniqD = unique(numD);
	pd = zeros(length(uniqD)); # probability of observing d descendants
	for ii = 1:length(uniqD)
		idx = find(numD.==uniqD[ii]);
		if !isempty(idx)
			nad = sum(numA[idx]);
			pd[ii] = (nad/Na);
		end
	end
=#

	i = 1;
	for ii = 1:length(dratio)
		# find the total number of ancestors in all avalanches where there were d descendants
		idx = find(d.==dratio[ii]);
		if !isempty(idx)
			nad = sum(numA[idx]);
			# from that, get the likelihood of d descendants
			pd[i] = (nad/Na);
		end
		i += 1;
	end

	net_bscore = sum(dratio.*pd)/length(span);

	# do the above again using subspikes where we are only looking at e_edges and i_edges
	subspikes_e = Spikes_binned[e_edges,:];
	numA = zeros(nbins-maximum(span)); # number of ancestors for each bin
	numD = zeros(nbins-maximum(span)); # number of descendants for each ancestral bin
	d = zeros(nbins-maximum(span)); # the number of electrode descendants per ancestor
	for tt = 1:length(numA)
		# that is, we look at time tt for nA and time tt+5ms:tt+20ms for nD
		# this is going to be a little hairy...
		numA[tt] = length(find(subspikes_e[:,tt].==1));
		numD[tt] = length(find(subspikes_e[:,tt+span]));
		d[tt] = round.(numD[tt]/numA[tt]); # the ratio of descendants to ancestors
	end
	dratio = filter(!isnan,d);
	filter!(!isinf,dratio);
	#filter!(!iszero,dratio);
	dratio = unique(dratio);
	sort!(dratio);
	pd = zeros(length(dratio),);
	Na = sum(numA); # the total number of ancestors over all bins
	#norm = (Nactive-1)/(Nactive - Na); # correction for refractoriness in next time bin (we do not have refractoriness)
	i = 1;
	for ii = 1:length(dratio)
		# find the total number of ancestors in all avalanches where there were d descendants
		idx = find(d.==dratio[ii]);
		if !isempty(idx)
			nad = sum(numA[idx]);
			# from that, get the likelihood of d descendants
			pd[i] = (nad/Na);
		end
		i += 1;
	end
	net_bscore_e = sum(dratio.*pd)/length(span);


	subspikes_i = Spikes_binned[i_edges,:];
	numA = zeros(nbins-maximum(span)); # number of ancestors for each bin
	numD = zeros(nbins-maximum(span)); # number of descendants for each ancestral bin
	d = zeros(nbins-maximum(span)); # the number of electrode descendants per ancestor
	for tt = 1:length(numA)
		# that is, we look at time tt for nA and time tt+5ms:tt+20ms for nD
		# this is going to be a little hairy...
		numA[tt] = length(find(subspikes_i[:,tt].==1));
		numD[tt] = length(find(subspikes_i[:,tt+span]));
		d[tt] = round.(numD[tt]/numA[tt]); # the ratio of descendants to ancestors
	end
	dratio = filter(!isnan,d);
	filter!(!isinf,dratio);
	#filter!(!iszero,dratio);
	dratio = unique(dratio);
	sort!(dratio);
	pd = zeros(length(dratio),);
	Na = sum(numA); # the total number of ancestors over all bins
	#norm = (Nactive-1)/(Nactive - Na); # correction for refractoriness in next time bin (we do not have refractoriness)
	i = 1;
	for ii = 1:length(dratio)
		# find the total number of ancestors in all avalanches where there were d descendants
		idx = find(d.==dratio[ii]);
		if !isempty(idx)
			nad = sum(numA[idx]);
			# from that, get the likelihood of d descendants
			pd[i] = (nad/Na);
		end
		i += 1;
	end
	net_bscore_i = sum(dratio.*pd)/length(span);



    return net_bscore_e, net_bscore_i, net_bscore;
end


function score(param;trials=0,generated_net=0,plot=false,generated_stim=0,generated_projected=0,saved_ic=0) # structured as p_ie, p_ei, generates and scores network
	 # connection probabilities

	 #args param is a param set of length 4 with [p_ie p_ei p_ii R]
	 #trials is number of trials to run for
	 #net is optinal argument for passing precomputed network
	 #plot is optional flag for more plot output
	 #generated stim is optinal arg for inputing predone poisson stim as gP_full




	p_ie = param[1]
	p_ei = param[2]
	p_ii = param[3]
	R    = param[4]
	p_ee = .2
	if(generated_projected==0)
		projected = randperm(N)[1:num_projected]
	else
		projected = generated_projected
	end


	cluster = 50 # number of clusters

	p_ee_out = (p_ee*cluster)/(R+cluster-1)
	p_ee_in  = p_ee_out*R

	p_ei_out =  (p_ei*cluster)/(R+cluster-1)
	p_ei_in  =  p_ei_out*R
	#print(p_ei_out)
	#print("\n")
	#print(p_ei_in)
	#print("\n")
	#print(p_ee_out)
	#print("\n")
	#print(p_ee_in)

	# hard refrac - take no input
	# add clustering to
	# add leak




	LOG_RAND_sigma = 1
	LOG_RAND_mu = log(1) - 0.5*(LOG_RAND_sigma*LOG_RAND_sigma) # this condition ensures that the mean of the new distributoin = 1

	LOG_RAND_sigmaInh = 0.1  # suggesting we hold these constant and only vary excitatory connections
	LOG_RAND_muInh = log(1) - 0.5*(LOG_RAND_sigma*LOG_RAND_sigmaInh)# (so we aren't scaling total weight as we explore heavy-tailedness)

	if(generated_net == 0)
		Network_holder = Network_Generation(Ne,Ni,p_ee_in,p_ee_out,cluster,p_ii,p_ie,p_ei_in,p_ei_out,log_mean,log_sigma,log_mean,log_sigma,log_mean,log_sigma,log_mean,log_sigma);
		we = Network_holder[1]*1*nS
		wi = Network_holder[2]*10*nS
	else
		we = [generated_net[:,1:Ne] zeros(N,Ni)]
		wi = [zeros(N,Ne) generated_net[:,(Ne+1):N]]
	end

	#we = zeros(4000,4000)
	#wi = zeros(4000,4000)
	#we = sprand(2400,2400,.11)*4*nS
	#wi = sprand(2400,2400,.11)*10*nS
	t=0
	delay=0
	if saved_ic == 0
		vm= -60*ones(N)+10*rand(N)
		ic = copy(vm)
	else
		vm = saved_ic
		ic = saved_ic
	end
	w=zeros(N)
	gE=zeros(N)
	gI=zeros(N)
	#I= ones(N)*.4*nA


	## set up poisson inputs here

	if(generated_stim==0)
		wp = poisson_network(poisson_probability,num_poisson,num_projected,poisson_mean_strength,poisson_varriance)*5*nS
		stim = poisson_stimulis_gen(num_poisson,poisson_spikerate,dt,stim_in,taup,poisson_max,stim_in) # give 10 ms of decay time before disregarding
		gP_full = wp'*stim;
	else
		gP_full = generated_stim
		wp = 0
		stim = 0
	end



	##

	net=Network500()
	net.t=t
	net.dt=dt
	net.delay=delay
	net.N=N
	net.vm=vm
	net.w=w
	net.gE=gE
	net.gI=gI
	net.gP=zeros(N)
	net.I=I
	net.we =we
	net.wi =wi
	net.refrac = zeros(N)

	net.vm_var_e = zeros(Ne)
	net.vm_var_i = zeros(Ni)
	net.vm_var = zeros(N)
	net.vm_var_e_M2 = zeros(Ne)
	net.vm_var_i_M2 = zeros(Ni)
	net.vm_var_M2 = zeros(N)
	net.vm_var_e_pop = 0
	net.vm_var_i_pop = 0
	net.vm_var_pop = 0
	net.vm_var_e_pop_M2 = 0
	net.vm_var_i_pop_M2 = 0
	net.vm_var_pop_M2 = 0

	#net.vmRecord=Array{Float64,2}(6,0)
    #net.gERecord=Array{Float64,2}(6,0)
	#net.wRecord=Array{Float64,2}(6,0)
	#net.gIRecord=Array{Float64,2}(6,0)

	## score the newtork

	net.tSpike=Array{Vector{Float64}}(net.N)
	fill!(net.tSpike,[-0])
	run!(net,run_total/dt,gP_full,projected) # time in ms

	if(plot)
		t,neuron=rasterPlot(net)
		figure()
		PyPlot.scatter(t,neuron,s=.1)
		#raster_gen(net.tSpike,we+wi)
	end
	#string_name = print("image","_",ind1,"_",ind2,"_",ind3,".svg")
	#PyPlot.savefig(string_name)
	#close()
	isi = net.tSpike
	#func_net = functional_net(net.tSpike)
	#func_net = func_net/(trials+1)
	#net_proj = we + wi
	#cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = clustered_motifs(net_proj[sort(projected),sort(projected)])

	net.vm_var_e,net.vm_var_e_M2,vm_var_e_sample = finalize_var([run_total/dt,net.vm_var_e,net.vm_var_e_M2])
	net.vm_var_i,net.vm_var_i_M2,vmi_var_i_sample = finalize_var([run_total/dt,net.vm_var_i,net.vm_var_i_M2])
	net.vm_var,net.vm_var_M2,vm_var_sample = finalize_var([run_total/dt,net.vm_var,net.vm_var_M2])
	net.vm_var_e_pop,net.vm_var_e_pop_M2,vm_var_e_pop_sample = finalize_var([run_total/dt,net.vm_var_e_pop,net.vm_var_e_pop_M2])
	net.vm_var_i_pop,net.vm_var_i_pop_M2,vm_var_i_pop_sample = finalize_var([run_total/dt,net.vm_var_i_pop,net.vm_var_i_pop_M2])
	net.vm_var_pop,net.vm_var_pop_M2,vm_var_pop_sample = finalize_var([run_total/dt,net.vm_var_pop,net.vm_var_pop_M2])


	score = zeros(9)
	score[1],score[2] = rate_est(net.tSpike)
	score[3] = score_time_ignited(net.tSpike)
	score[4] = net.vm_var_e_pop_M2/mean(net.vm_var_e_M2)#synch e
	score[5] = net.vm_var_i_pop_M2/mean(net.vm_var_i_M2)#synch i
	score[6] = net.vm_var_pop_M2/mean(net.vm_var_M2)#synch tot
	bin = 5;
	score[7],score[8],score[9] = branching_param(net.tSpike,generated_net,bin)#branc_e, branch_i,branch_total
	print(score,"\n\n")

	#discretize_dt = 5
	#real_raster = binary_raster_gen(net.tSpike,discretize_dt)# 5ms
	#motif_functional_graphs((we/1*nS)[1:Ne],real_raster,discretize_dt)
	#null_poisson_raster = poisson_raster_gen(discretize_dt,score[1],score[2]) # 5 ms
	#motif_functional_graphs((we/1*nS)[1:Ne],null_poisson_raster,discretize_dt)
	#shuffle_null_raster = shuffle_null_raster_gen(net.tSpike,discretize_dt)
	#motif_functional_graphs((we/1*nS)[1:Ne],shuffle_null_raster,discretize_dt)


	#e,i
	#null_spikes=Array{Vector{Float64}}(net.N)
	#fill!(null_spikes,[-0])
	#for i = 1:Ne
	#	net.tSpike[i]=
	#end


	#motif_functinoal_graphs(null_spikes,we+wi,1)
	#score[4] = cycle_cc_mean
	#score[5] = middle_cc_mean
	#score[6] = out_cc_mean
	#score[7] = in_cc_mean
	#score[8] = cycle_cc_median
	#score[9] = middle_cc_median
	#score[10] = out_cc_median
	#score[11] = in_cc_median
	#score[12] = all_mean
	#score[13] = all_median
	#try
	#	score[14] = sum(abs.(eigs(we/(1*nS)+wi/(10*nS))))
	#catch
	#	score[14] = 0
	#end
	#score[4] = branching_score(net.tSpike)
	#tic()
	#score[4],score[5] = population_synch_score_e_i(isi)
	#toc()
	#score[3] = 1 #poisson_likelihood_sum_counts(isi,run_total,100,10) fix this later if necessary
	#score[4] = skewness(histo_counts)

	#adj_mat  = we+wi
	#adj_mat[find(adj_mat)] = 1
	#adj_mat_ee = we
	#adj_mat_ee[find(adj_mat_ee)] = 1
	#adj_mat_ii = wi
	#adj_mat_ii[find(adj_mat_ii)] = 1
	#if(plot)
	#	synch_vals, ts = synch_over_time(adj_mat,isi,current_in,run_total,2.5)
	#	figure()
	#	PyPlot.plot(ts,synch_vals)
	#	title("synch total z score")
	#	xlabel("time")
	#	ylabel("zscore")
	#end
	#synch_vals_ee, ts = synch_over_time(adj_mat_ee,isi,current_in,run_total,5)
	#figure()
	#PyPlot.plot(ts,synch_vals)
	#title("synch excitatory z score")
	#xlabel("time")
	#ylabel("zscore")

	#synch_vals_ii, ts = synch_over_time(adj_mat_ii,isi,current_in,run_total,5)
	#figure()
	#PyPlot.plot(ts,synch_vals)
	#title("synch inhibitory z score")
	#xlabel("time")
	#ylabel("zscore")

	#motif_over_time,motif_over_time_normalized,ts = motifs_overtime(isi,current_in,run_total,2.5,adj_mat)

	#for i = 1:16
	#	figure()
	#	PyPlot.plot(ts,motif_over_time[:,i])
	#	xlabel("time")
	#	ylabel("probability")
	#	name = string("motif ",i," prob over time")
	#	title(name)
	#	motif_overall = motif_baseline(adj_mat)
	#	figure()
	#	PyPlot.plot(ts,motif_over_time_normalized[:,i])
	#	xlabel("time")
	#	ylabel("multiplier over time")
	#	name = string("motif ",i," multiple of whole graph prob over time")
	#	title(name)
	#end

	# generate cross correlations

	#for i in [10 13 14] #number of motifs

	#	corr    = xcorr(motif_over_time_normalized[:,i],synch_vals)
		#corr_ee = xcorr(motif_over_time_normalized[:,i],synch_vals_ee)
		#corr_ii = xcorr(motif_over_time_normalized[:,i],synch_vals_ii)

		#abs_corr    = xcorr(motif_over_time_normalized[:,i],abs(synch_vals))
		#abs_corr_ee = xcorr(motif_over_time_normalized[:,i],abs(synch_vals_ee))
		#abs_corr_ii = xcorr(motif_over_time_normalized[:,i],abs(synch_vals_ii))

	#	lags = (-(length(corr)-1)/2):1:((length(corr)-1)/2)

	#	figure()
	#	PyPlot.plot(lags,corr)
	#	name = string("xcorr motif ",i," with poisson z score null (all)")
	#	ylabel("correlation")
	#	xlabel("lags")
	#	title(name)

		#figure()
		#PyPlot.plot(lags,corr_ee)
		#name = string("xcorr motif ",i," with poisson z score null (excite)")
		#ylabel("correlation")
		#xlabel("lags")
		#title(name)

		#figure()
		#PyPlot.plot(lags,corr_ii)
		#name = string("xcorr motif ",i," with poisson z score null (inhib)")
		#ylabel("correlation")
		#xlabel("lags")
		#title(name)

		#figure()
		#PyPlot.plot(lags,abs_corr)
		#name = string("xcorr motif ",i," with poisson z score null (all,abs)")
		#ylabel("correlation")
		#xlabel("lags")
		#title(name)

		#figure()
		#PyPlot.plot(lags,abs_corr_ee)
		#name = string("xcorr motif ",i," with poisson z score null (excite,abs)")
		#ylabel("correlation")
		#xlabel("lags")
		#title(name)

		#figure()
		#PyPlot.plot(lags,abs_corr_ii)
		#name = string("xcorr motif ",i," with poisson z score null (inhib,abs)")
		#ylabel("correlation")
		#xlabel("lags")
		#title(name)
	#end

	#cluster_vals = cluster_over_time(isi,current_in,run_total,5,we+wi)
	#poisson_nul_hist(isi,current_in,run_total,20*1000,5)
	#figure()
	#null_counts = poisson_nul_hist(isi,trials,current_in,run_total,20*1000,5)
	#PyPlot.plt[:hist](null_counts[1:end-1],25,normed=true)
	#print(histo_counts)
	#PyPlot.plt[:hist](histo_counts,25,normed=true)


	#return score,	wp , stim, gP_full, we+wi, net.tSpike, projected, cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC
	#ISI_hist(net.tSpike)
	return score, net.tSpike, projected, we+wi, gP_full, ic
	#, func_net# cluster_vals #Network_holder#, net#, mat
	#return net.tSpike, we+wi

end

function ISI_hist(spikes)
	t_Ne = []
	t_Ni = []
	for i = 1:Ne
		spike_set = spikes[i]
		if (length(spike_set) > 1)
			t_Ne = append!(t_Ne,spike_set[2:length(spike_set)]-spike_set[1:length(spike_set)-1])
		end
	end

	for i = Ne+1:N
		spike_set = spikes[i]
		if (length(spike_set) > 1)
			t_Ni = append!(t_Ni,spike_set[2:length(spike_set)]-spike_set[1:length(spike_set)-1])
		end
	end
	figure()
	xlabel("Time (ms)")
	title("excitatory ISIs")
	PyPlot.plt[:hist](t_Ne,800,normed=true)
	figure()
	xlabel("Time (ms)")
	title("inhibitory ISIs")
	PyPlot.plt[:hist](t_Ni,800,normed=true)

end

function clean_dir(path,run_number,thresh)
    path_string = string(path,run_number)
    score_path = string(path_string,"/scores/")
    list = readdir(score_path)
    spikes_path = string(path_string,"/spikes/")
    list2 = readdir(spikes_path)
	poisson_path = string(path_string,"/poisson_in_data/")
    list3 = readdir(poisson_path)
	network_path = string(path_string,"/network/")
    list4 = readdir(network_path)
	analysis_path=string(path_string,"/analysis/")
    for i = 1:length(list)
        cd(score_path)
        data = load(list[i])["scores"]
        if(data[1]>thresh && data[3]<.95)
            rm(list[i])
            cd(spikes_path)
            rm(list2[i])
            cd(poisson_path)
            rm(list3[i])
            cd(network_path)
            rm(list4[i])
        end
    end
    return 0
end


function score_logger(run_number)
	save_counter = 0;
	path_string = string("D:/qing/",run_number)

	score_path = string(path_string,"/scores/")
	poisson_path = string(path_string,"/poisson_in_data/")
	network_path = string(path_string,"/network/")
	spikes_path = string(path_string,"/spikes/")
	analysis_path=string(path_string,"/analysis/")

	mkdir(path_string)
	mkdir(score_path)
	mkdir(poisson_path)
	mkdir(network_path)
	mkdir(spikes_path)
	mkdir(analysis_path)

	Constants = Array[[Ne], [Ni], [Ne+Ni],[log_mean], [log_sigma], [run_total],
	[num_poisson],[num_projected],[stim_in],[poisson_probability], [poisson_mean_strength],
	[poisson_varriance], [poisson_spikerate], [poisson_max], [current_in], [current_max],[dt],[C],[gL],[EL],
	[VT],[DeltaT],[Vcut],[tauw],[a],[b],[Vr],[EE],[EI],[taue],[taui],[taup],[hard_refrac],[refrac_max]]

	Constants_names = ["Ne", "Ni", "Ne+Ni","log_mean","log_sigma","run_total",
	"num_poisson","num_projected","stim_in","poisson_probability", "poisson_mean_strength",
	"poisson_varriance", "poisson_spikerate", "poisson_max", "current_in", "current_max","dt","C","gL","EL",
	"VT","DeltaT","Vcut","tauw","a","b","Vr","EE","EI","taue","taui","taup","hard_refrac","refrac_max"]


	constants_file_path = string(path_string,"/Constants.jld")

	save(constants_file_path, "Constants", Constants)
	save(constants_file_path, "Constant Names", Constants_names)
	for i = 1:10000000
		score_vals, spikes, projected, graph, gP_full = score([.31,.22, .3,2])
		if score_vals[1] < 5 && score_vals[3] > .2
			poisson_file_path = string(poisson_path,"PoissonData",i,".jld")
			JLD.save(poisson_file_path, "poisson_stim", gP_full,
			"input_neurons",projected)

			network_file_path = string(network_path,"NetworkData",i,".jld")
			JLD.save(network_file_path, "network", graph)

			score_file_path = string(score_path,"ScoreData",i,".jld")
			JLD.save(score_file_path, "scores", score_vals)

			spikes_file_path = string(spikes_path,"SpikeData",i,".jld")
			JLD.save(spikes_file_path, "spikes", spikes)
		end
		#poisson_file_path = string(poisson_path,"PoissonData",counter,".jld")
		#psth_data_e,psth_data_i = psth(spikes,2,run_total;plot=false)
		#clustering = cluster_over_time(spikes,network)

		#result_file_path = string(score_path,"ResultData",counter,".jld")

		#spikes_file_path = string(spikes_path,"SpikesData",counter,".jld")
		#analysis_file_path = string(analysis_path,"Analysis",counter,".jld")
		#save(result_file_path, "score_vals",score_vals)
		#save(network_file_path, "network", network)
		#save(poisson_file_path, "poisson_stim",poisson_stim)
		#save(spikes_file_path, "spikes",spikes)
		#save(analysis_file_path, "PSTH_e",psth_data_e,"PSTH_i",psth_data_i)#"cluster_vals",clustering)
		print(i)
		print("\n")
	end
	return 0
end

function score_logger_fixed_network(run_number)
	counter = 0;

	path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
	score_path = string(path_string,"/scores/")
	poisson_path = string(path_string,"/poisson_in_data/")
	network_path = string(path_string,"/network/")
	spikes_path = string(path_string,"/spikes/")
	analysis_path=string(path_string,"/analysis/")
	projected_path=string(path_string,"/projected/")

	mkdir(path_string)
	mkdir(score_path)
	mkdir(poisson_path)
	mkdir(network_path)
	mkdir(spikes_path)
	mkdir(analysis_path)
	mkdir(projected_path)

	Constants = Array[[Ne], [Ni], [Ne+Ni],[log_mean], [log_sigma], [run_total],
	[num_poisson],[num_projected],[stim_in],[poisson_probability], [poisson_mean_strength],
	[poisson_varriance], [poisson_spikerate], [poisson_max], [current_in], [current_max],[dt],[C],[gL],[EL],
	[VT],[DeltaT],[Vcut],[tauw],[a],[b],[Vr],[EE],[EI],[taue],[taui],[taup],[hard_refrac],[refrac_max]]

	Constants_names = ["Ne", "Ni", "Ne+Ni","log_mean","log_sigma","run_total",
	"num_poisson","num_projected","stim_in","poisson_probability", "poisson_mean_strength",
	"poisson_varriance", "poisson_spikerate", "poisson_max", "current_in", "current_max","dt","C","gL","EL",
	"VT","DeltaT","Vcut","tauw","a","b","Vr","EE","EI","taue","taui","taup","hard_refrac","refrac_max"]


	constants_file_path = string(path_string,"/Constants.jld")

	save(constants_file_path, "Constants", Constants)
	save(constants_file_path, "Constant Names", Constants_names)
	fixed_net = JLD.load("/Users/kyle/Desktop/Jason_Data/12/network/NetworkData23.jld")
	fixed_net = fixed_net["network"]
	while(true)
		score_vals, poisson_net, poisson_spikes, poisson_stim, network, spikes, projected_neurons, cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = score([.31,.22, .3,2], generated_net = fixed_net)
		counter = counter + 1;

		#poisson_file_path = string(poisson_path,"PoissonData",counter,".jld")
		#JLD.save(poisson_file_path, "poisson_net", poisson_net,"poisson_spikes", poisson_spikes,"poisson_stim", poisson_stim)

		#network_file_path = string(network_path,"NetworkData",counter,".jld")
		#JLD.save(network_file_path, "network", network)

		#psth_data_e,psth_data_i = psth(spikes,2,run_total;plot=false)
		#clustering = cluster_over_time(spikes,network)

		result_file_path = string(score_path,"ResultData",counter,".jld")
		projected_file_path = string(projected_path,"ProjectedData",counter,".jld")
		analysis_file_path = string(analysis_path,"Analysis",counter,".jld")
		save(result_file_path, "score_vals",score_vals)
		#save(network_file_path, "network", network)
		#save(spikes_file_path, "spikes",spikes)
		save(projected_file_path,"projected",projected_neurons)
		save(analysis_file_path, "cycle_CC",cycle_CC,"middleman_CC",middleman_CC,"fanin_CC",fanin_CC,"fanout_CC",fanout_CC,"all_motifs_CC",all_motifs_CC)#"cluster_vals",clustering)
		print(counter)
		print("\n")
	end
	return 0
end

function score_logger_poisson_fixed(run_number)
	counter = 0;

	path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
	score_path = string(path_string,"/scores/")
	poisson_path = string(path_string,"/poisson_in_data/")
	network_path = string(path_string,"/network/")
	spikes_path = string(path_string,"/spikes/")
	analysis_path=string(path_string,"/analysis/")
	projected_path=string(path_string,"/projected/")

	mkdir(path_string)
	mkdir(score_path)
	mkdir(poisson_path)
	mkdir(network_path)
	mkdir(spikes_path)
	mkdir(analysis_path)
	mkdir(projected_path)

	Constants = Array[[Ne], [Ni], [Ne+Ni],[log_mean], [log_sigma], [run_total],
	[num_poisson],[num_projected],[stim_in],[poisson_probability], [poisson_mean_strength],
	[poisson_varriance], [poisson_spikerate], [poisson_max], [current_in], [current_max],[dt],[C],[gL],[EL],
	[VT],[DeltaT],[Vcut],[tauw],[a],[b],[Vr],[EE],[EI],[taue],[taui],[taup],[hard_refrac],[refrac_max]]

	Constants_names = ["Ne", "Ni", "Ne+Ni","log_mean","log_sigma","run_total",
	"num_poisson","num_projected","stim_in","poisson_probability", "poisson_mean_strength",
	"poisson_varriance", "poisson_spikerate", "poisson_max", "current_in", "current_max","dt","C","gL","EL",
	"VT","DeltaT","Vcut","tauw","a","b","Vr","EE","EI","taue","taui","taup","hard_refrac","refrac_max"]


	constants_file_path = string(path_string,"/Constants.jld")

	save(constants_file_path, "Constants", Constants)
	save(constants_file_path, "Constant Names", Constants_names)
	fixed_poisson = JLD.load("/Users/kyle/Desktop/Jason_Data/14/poisson_in_data/PoissonData1694.jld")
	fixed_poisson = fixed_poisson["poisson_stim"]
	while(true)
		score_vals, poisson_net, poisson_spikes, poisson_stim, network, spikes, projected_neurons, cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = score([.31,.22, .3,2], generated_stim = fixed_poisson)
		counter = counter + 1;

		#poisson_file_path = string(poisson_path,"PoissonData",counter,".jld")
		#JLD.save(poisson_file_path, "poisson_net", poisson_net,"poisson_spikes", poisson_spikes,"poisson_stim", poisson_stim)

		#network_file_path = string(network_path,"NetworkData",counter,".jld")
		#JLD.save(network_file_path, "network", network)

		#psth_data_e,psth_data_i = psth(spikes,2,run_total;plot=false)
		#clustering = cluster_over_time(spikes,network)

		result_file_path = string(score_path,"ResultData",counter,".jld")
		projected_file_path = string(projected_path,"ProjectedData",counter,".jld")
		analysis_file_path = string(analysis_path,"Analysis",counter,".jld")
		save(result_file_path, "score_vals",score_vals)
		#save(network_file_path, "network", network)
		#save(spikes_file_path, "spikes",spikes)
		save(projected_file_path,"projected",projected_neurons)
		save(analysis_file_path, "cycle_CC",cycle_CC,"middleman_CC",middleman_CC,"fanin_CC",fanin_CC,"fanout_CC",fanout_CC,"all_motifs_CC",all_motifs_CC)#"cluster_vals",clustering)
		print(counter)
		print("\n")
	end
	return 0
end


function score_logger_vary_poisson(run_number,file_number,file_directory)
	counter = 1;
	path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)

	score_path = string(path_string,"/scores/")
	poisson_path = string(path_string,"/poisson_in_data/")
	network_path = string(path_string,"/network/")

	mkdir(path_string)
	mkdir(score_path)
	mkdir(poisson_path)
	mkdir(network_path)



	Constants = Array[[Ne], [Ni], [Ne+Ni],[log_mean], [log_sigma], [run_total],
	[num_poisson],[num_projected],[stim_in],[poisson_probability], [poisson_mean_strength],
	[poisson_varriance], [poisson_spikerate], [poisson_max], [current_in], [current_max],[dt],[C],[gL],[EL],
	[VT],[DeltaT],[Vcut],[tauw],[a],[b],[Vr],[EE],[EI],[taue],[taui],[taup],[hard_refrac],[refrac_max]]

	Constants_names = ["Ne", "Ni", "Ne+Ni","log_mean","log_sigma","run_total",
	"num_poisson","num_projected","stim_in","poisson_probability", "poisson_mean_strength",
	"poisson_varriance", "poisson_spikerate", "poisson_max", "current_in", "current_max","dt","C","gL","EL",
	"VT","DeltaT","Vcut","tauw","a","b","Vr","EE","EI","taue","taui","taup","hard_refrac","refrac_max"]


	constants_file_path = string(path_string,"/Constants.jld")

	JLD.save(constants_file_path, "Constants", Constants)
	JLD.save(constants_file_path, "Constant Names", Constants_names)

    load_string = string("/Users/kyle/desktop/Jason_Data/",file_directory,"/scores/ResultData",file_number,".jld")
    predone_network = JLD.load(load_string)["network"]
    predone_to_project = JLD.load(load_string)["projected_neurons"]


	while(true)
		score_vals, poisson_net, poisson_spikes, poisson_stim, network, spikes, projected_neurons = score([.31,.22, .3,2],generated_net=predone_network,generated_projected=predone_to_project)
		counter = counter + 1;

		#poisson_file_path = string(poisson_path,"PoissonData",counter,".jld")
		#JLD.save(poisson_file_path, "poisson_net", poisson_net,"poisson_spikes", poisson_spikes,"poisson_stim", poisson_stim)

		network_file_path = string(network_path,"NetworkData",counter,".jld")
		JLD.save(network_file_path, "network", network)

		result_file_path = string(score_path,"ResultData",counter,".jld")
		JLD.save(result_file_path, "score_vals",score_vals)
	end
	return 0
end
#################################################################################################################################################
######################################################################## NETWORK GEN ############################################################
#################################################################################################################################################

# modified for rectangular matrices -- note: you need to remove autapses in
#       postprocessing, modifief furthor replaceing while loop with mods
#  BC Apr-11-2014, KB 12-15-15
# modified again for usage in julia

function smallworld_bc_v2(N_out, N_in, connectionProb, rewireProb, log_mean, log_sig) # warning, this should break if connectionProb == 1

	# N_out := number of rows
	# N_in := number of columns
	# with lattice size determined by connectionProb and N_in
	rectangularity = N_in / N_out; # compute compensatory offset in lattice
	adjmat_sw = zeros(N_out,N_in);
	lattice_halfwidth = round(Int, 0.5 * connectionProb * N_in)+1;
	d = LogNormal(log_mean, log_sig)

	for i_row = 1:N_out  #generate lattice

    	for j_offset = -lattice_halfwidth:1:lattice_halfwidth

	       	idx = round(Int,i_row*rectangularity) + j_offset;
        	idx = mod(idx-1,N_in)+1; #-1 shifts 1 unit left +1 raises one unit
        	adjmat_sw[i_row,idx] = rand(d,1)[1];
    	end
	end

	for i_pre = 1:N_out # rewire
    	for i_post = 1:N_in
        	if adjmat_sw[i_pre, i_post] != 0 && rand(1)[1] < rewireProb # if there is a connection and it qualifies for rewiring

            	if rand(1)[1] < 0.5
                	indicies = find(adjmat_sw[:,i_post]-1);
                	if(!isempty(indicies))
                    	newCandidate = rand(1:length(indicies));
                    	newCandidate = indicies[newCandidate];

                    	adjmat_sw[newCandidate,i_post] = adjmat_sw[i_pre, i_post]; # rewire
                    	adjmat_sw[i_pre, i_post] = 0; # disconnect old edge
                	end
            	else
                	indicies = find(adjmat_sw[i_pre,:]-1);
                	if(!isempty(indicies))
                    	newCandidate = rand(1:length(indicies));
                    	newCandidate = indicies[newCandidate];
                    	adjmat_sw[i_pre, newCandidate] = adjmat_sw[i_pre, i_post]; # rewire#                    	adjmat_sw[i_pre, i_post] = 0; # disconnect old edge
                		adjmat_sw[i_pre, i_post] = 0;
                	end
            	end
        	end
    	end
	end
	if(N_out == N_in)
		for i = 1:N_in
			adjmat_sw[i,i] = 0;
		end
	end
	return adjmat_sw
end

# keeping around to see logic and arguments without going into src

function er_graph(N_out, N_in, p, log_mean, log_sig, autap)
	mat = zeros(N_out,N_in)
	d = LogNormal(log_mean, log_sig)
	for i = 1:N_out
		for j = 1:N_in
			if(autap)
				if((rand()<p) & (i != j))
					mat[i,j] = rand(d)
				end
			else
				if(rand()<p)
					mat[i,j] = rand(d)
				end
			end
		end
	end
	return mat
end

function doiron_homo(N, p_in, p_out, log_mean, log_sig, cluster_size)
	mat = er_graph(N, N, p_out, log_mean, log_sig, false)
	d = LogNormal(log_mean, log_sig)
	cluster_start = 1:cluster_size:N
	cluster_end = cluster_start + cluster_size - 1;
	for i = 1:length(cluster_start)
		mat[cluster_start[i]:cluster_end[i],cluster_start[i]:cluster_end[i]] = er_graph(cluster_size, cluster_size, p_in, log_mean, log_sig, true)
	end
	return mat
end

function doiron_hetero(N_out,N_in, p_in, p_out, log_mean, log_sig, num_clusters)
	mat = er_graph(N_out, N_in, p_out, log_mean, log_sig, false)
	membership1 = rand(1:num_clusters,1,N_out) # give each neuron two clusters
	membership2 = rand(1:num_clusters,1,N_out)
	d = LogNormal(log_mean, log_sig)
	for i = 1:num_clusters
		region1 = find(x-> x==i,membership1)
		region2 = find(x-> x==i,membership2)
		region  = unique([region1; region2])
		mat[region,region] = er_graph(length(region), length(region), p_in, log_mean, log_sig, true)
	end
	return mat
end

function adj_mat_to_Graph(adj_mat)

	G = Graph(size(adj_mat)[1])
	to_add = find(adj_mat);
	print(length(to_add))
	print("\n")
	for i = 1:length(to_add)
		print(i)
		print("\n")
		points = ind2sub(zeros(size(adj_mat)[1],size(adj_mat)[1]),to_add[i])
		add_edge!(G, points[1], points[2])
	end
	return G
end

function Network_Generation( Ne,Ni,p_ee_in,p_ee_out,cluster,p_ii,p_ie,p_ei_in,p_ei_out,log_mean_ee,log_sig_ee,log_mean_ei,log_sig_ei,log_mean_ie,log_sig_ie,log_mean_ii,log_sig_ii)
	a = doiron_hetero(Ne, Ne, p_ee_in, p_ee_out, log_mean_ee,log_sig_ee,cluster )
	b = er_graph(Ni, Ni, p_ii, log_mean_ii,log_sig_ii,true)
	c = er_graph(Ne, Ni, p_ie, log_mean_ie,log_sig_ie,false)
	d = doiron_hetero(Ni, Ne, p_ei_in,p_ei_out, log_mean_ei,log_sig_ei,cluster)
	we = [a zeros(Ne,Ni);d zeros(Ni,Ni)]
	wi = [zeros(Ne,Ne) c; zeros(Ni,Ne) b]
	return we, wi
end


function grid_search(p_ie_range,p_ei_range;num_runs = 10)
	#p_ie_range = .3090:.0001:.313 #tiny grid
	#p_ei_range = .216:.0001:.220 #tiny grid
	#p_ie_range = .29:.001:.37#big grid
	#p_ei_range = .16:.001:.24
	e_rate_vals = zeros(num_runs,length(p_ie_range),length(p_ei_range))
	i_rate_vals = zeros(num_runs,length(p_ie_range),length(p_ei_range))
	time_vals = zeros(num_runs,length(p_ie_range),length(p_ei_range))
	#score[1],score[2] = rate_est(net.tSpike)
	#score[3] = score_time_ignited(net.tSpike)
	for i = 1:num_runs
		print("\n",i,"\n")
		for j = 1:length(p_ie_range)
			for k = 1:length(p_ei_range)
				vals = score([p_ie_range[j],p_ei_range[k], .3,2]) #[p_ie p_ei p_ii R]
				e_rate_vals[i,j,k] = vals[1]
				i_rate_vals[i,j,k] = vals[2]
				time_vals[i,j,k] = vals[3]
			end
		end
	end
	return e_rate_vals,i_rate_vals,time_vals
end

function line_search_ei_fixed()
	num_runs = 300
	p_ie_range = .3090:.00005:.313
	p_ei = .218 #tiny grid

	e_rate_vals = zeros(num_runs,length(p_ie_range))
	i_rate_vals = zeros(num_runs,length(p_ie_range))
	time_vals = zeros(num_runs,length(p_ie_range))
	#score[1],score[2] = rate_est(net.tSpike)
	#score[3] = score_time_ignited(net.tSpike)
	for i = 1:num_runs
		print("\n",i,"\n")
		for j = 1:length(p_ie_range)
			vals = score([p_ie_range[j],p_ei, .3,2]) #[p_ie p_ei p_ii R]
			e_rate_vals[i,j] = vals[1]
			i_rate_vals[i,j] = vals[2]
			time_vals[i,j] = vals[3]
		end
	end
	return e_rate_vals,i_rate_vals,time_vals
end

function line_search_ie_fixed()
	num_runs = 300
	p_ie = .311
	p_ei_range = .216:.00005:.220 #tiny grid

	e_rate_vals = zeros(num_runs,length(p_ei_range))
	i_rate_vals = zeros(num_runs,length(p_ei_range))
	time_vals = zeros(num_runs,length(p_ei_range))
	#score[1],score[2] = rate_est(net.tSpike)
	#score[3] = score_time_ignited(net.tSpike)
	for i = 1:num_runs
		print("\n",i,"\n")
		for j = 1:length(p_ei_range)
			vals = score([p_ie,p_ei_range[j], .3,2]) #[p_ie p_ei p_ii R]
			e_rate_vals[i,j] = vals[1]
			i_rate_vals[i,j] = vals[2]
			time_vals[i,j] = vals[3]
		end
	end
	return e_rate_vals,i_rate_vals,time_vals
end
#=
function line_search()
	num_runs = 500
	p_ei = .218
	p_ie = .311
	e_rate_vals = zeros(num_runs,length(p_ie_range),length(p_ei_range))
	i_rate_vals = zeros(num_runs,length(p_ie_range),length(p_ei_range))
	time_vals = zeros(num_runs,length(p_ie_range),length(p_ei_range))
	#score[1],score[2] = rate_est(net.tSpike)
	#score[3] = score_time_ignited(net.tSpike)
	for i = 1:num_runs
		print("\n",i,"\n")
		for j = 1:length(p_ie_range)
			for k = 1:length(p_ei_range)
				vals = score([p_ie_range[j],p_ei_range[k], .3,2]) #[p_ie p_ei p_ii R]
				e_rate_vals[i,j,k] = vals[1]
				i_rate_vals[i,j,k] = vals[2]
				time_vals[i,j,k] = vals[3]
			end
		end
	end
	return e_rate_vals,i_rate_vals,time_vals
end
=#
#################################################################################################################################################
######################################################################## VIS TOOLS ##############################################################
#################################################################################################################################################


function raster(p)
  fire = p.records[:fire]
  x, y = Float32[], Float32[];
  for t = eachindex(fire)
    for n in find(fire[t])
      push!(x, t)
      push!(y, n)
    end
  end
  x, y
end

function raster(P::Array)
  y0 = Int[0]
  X = Float32[]; Y = Float32[]
  for p in P
    x, y = raster(p)
    append!(X, x)
    append!(Y, y + sum(y0))
    push!(y0, p.N)
  end
  plt = scatter(X, Y, m = (1, :black), leg = :none,
              xaxis=("t", (0, Inf)), yaxis = ("neuron",))
  y0 = y0[2:end-1]
  !isempty(y0) && hline!(plt, cumsum(y0), linecolor = :red)
  return plt
end

function vecplot(p, sym)
  v = getrecord(p, sym)
  y = hcat(v...)'
  x = 1:length(v)
  plot(x, y, leg = :none,
      xaxis=("t", extrema(x)),
      yaxis=(string(sym), extrema(y)))
end

function vecplot(P::Array, sym)
  plts = [vecplot(p, sym) for p in P]
  plot(plts..., layout = (length(plts), 1))
end

function windowsize(p)
  A = sum.(p.records[:fire]) / length(p.N)
  W = round(Int, 0.5p.N / mean(A)) # filter window, unit=1
end

function density(p, sym)
  X = getrecord(p, sym)
  t = 1:length(X)
  xmin, xmax = extrema(vcat(X...))
  edge = linspace(xmin, xmax, 50)
  c = center(edge)
  ρ = [fit(Histogram, x, edge).weights |> float for x in X] |> x -> hcat(x...)
  ρ = smooth(ρ, windowsize(p), 2)
  ρ ./= sum(ρ, 1)
  p = @gif for t = 1:length(X)
    bar(c, ρ[:, t], leg = false, xlabel = string(sym), yaxis = ("p", extrema(ρ)))
  end
  is_windows() && run(`powershell start $(p.filename)`)
  is_unix() && run(`xdg-open $(p.filename)`)
  p
end

function rateplot(p, sym)
  r = getrecord(p, sym)
  R = hcat(r...)
end

function rateplot(P::Array, sym)
  R = vcat([rateplot(p, sym) for p in P]...)
  y0 = [p.N for p in P][2:end-1]
  plt = heatmap(flipdim(R, 1), leg = :none)
  !isempty(y0) && hline!(plt, cumsum(y0), line = (:black, 1))
  plt
end

function activity(p)
  A = sum.(p.records[:fire]) / length(p.N)
  W = windowsize(p)
  A = smooth(A, W)
end

function activity(P::Array)
  A = activity.(P)
  t = 1:length(P[1].records[:fire])
  plot(t, A, leg=:none, xaxis=("t",), yaxis=("A", (0, Inf)))
end

function if_curve(model, current; neuron = 1, dt = 0.1ms, duration = 1second)
  E = model(neuron)
  monitor(E, [:fire])
  f = Float32[]
  for I = current
    clear_records(E)
    E.I = [I]
    SNN.sim!([E], []; dt = dt, duration = duration)
    push!(f, activity(E))
  end
  plot(current, f)
end

function rasterPlot(net)
  t=[]
  neuron=[]
  for n=1:length(net.tSpike)
    append!(t,net.tSpike[n])
    append!(neuron,n*ones(length(net.tSpike[n])))
  end
  t=convert(Array{Float64},t)
  neuron=convert(Array{Float64},neuron)
  idx=find(t.>0)
  t=t[idx]
  neuron=neuron[idx]
  return t,neuron
end
