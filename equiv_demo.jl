#= This script will attempt to demonstrate the equivalence or non-equivalence
(of some sort) between:
1. Average population motif CC timecourses on the recruitment network for one of
   our SNNs.
2. Population motif CC timecourse calculated from the topology of the same SNN,
   using the spikes as 'votes' of each unit's CC contribution to the population
   mean at each timestep.
=#

include("/Users/maclean lab/Documents/qing/SNNactivity/jason_proj_2.jl");

# create some network, run it, look at its spikes and topology
function main()
    tSpike,graph = gen_net();
    # from tSpike, create a matrix of 1's and 0's.
    nbins = run_total/dt-1;
    Spikes_binned = zeros(N,nbins);
    for ii = 1:N
        # find the indices for neuron ii for which we'll put a 1 in Spikes_binned
        if !isempty(tSpike[ii,])
            indices = floor.(tSpike[ii,]./dt);
            for jj = 2:length(indices) # first value is '0'
                Spikes_binned[ii,convert(Int64,indices[jj])] = 1;
            end
        end
        # in the end we'll get a '1' in the right [neuron,timebin] array slot for each spike of each neuron
    end

    # perform first option.
    cycle_recruit_CC, middle_recruit_CC, fanin_recruit_CC, fanout_recruit_CC, all_recruit_CC = motifs_through_time(graph,Spikes_binned);
    # plot

    # perform second option.
    cycle_static_CC, middle_static_CC, fanin_static_CC, fanout_static_CC, all_static_CC = get_CC_static(Spikes_binned,graph);

    # compare the timecourses from the second method with the motif_over_time_safe variables from the first method
    ts = 1:1:nbins-1;
    PyPlot.plot(ts,fanin_timecourse);
    PyPlot.plot!(ts,fanin_over_time_safe);
end

function get_CC_recruitment(Spikes_binned,graph)
    cycle_over_time = zeros(size(Spikes_binned)[2]-1);
	middleman_over_time = zeros(size(Spikes_binned)[2]-1);
	fanin_over_time = zeros(size(Spikes_binned)[2]-1);
	fanout_over_time = zeros(size(Spikes_binned)[2]-1);
	all_motifs_over_time = zeros(size(Spikes_binned)[2]-1);
    for i = 1:(size(Spikes_binned)[2]-1)
		recruitment_graph = zeros(N,N);
		bin_1 = Spikes_binned[:,i];
		bin_2 = Spikes_binned[:,i+1];

		pre_spike = find(bin_1.==1);
		post_spike = find(bin_2.==1);

        for jj = 1:length(pre_spike)
            # find the intersection between places in the graph where jj projects
            # to other units (we'll get the index of those columns) and the post_spike units.
            # thus we will get place&time where there is a real connection and where that real connection led to sequential spikes
            inds = intersect(post_spike,find(graph[:,jj]));
            if !isempty(inds)
                recruitment_graph[inds,jj] = graph[inds,jj];
            end
        end

		#valid_nodes = intersect(find(sum(functional_graph,1)),find(sum(functional_graph,2)));
		#functional_graph = functional_graph[valid_nodes,valid_nodes];
		cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = clustered_motifs(recruitment_graph);
		cycle_over_time[i] = mean(cycle_CC);
		middleman_over_time[i] = mean(middleman_CC);
		fanin_over_time[i] = mean(fanin_CC);
		fanout_over_time[i] = mean(fanout_CC);
		all_motifs_over_time[i] = mean(all_motifs_CC);
	end
    # clean up
    fanin_over_time_safe = fanin_over_time[find(!isnan,fanin_over_time)];
    fanout_over_time_safe = fanout_over_time[find(!isnan,fanout_over_time)];
    middleman_over_time_safe = middleman_over_time[find(!isnan,middleman_over_time)];
    cycle_over_time_safe = cycle_over_time[find(!isnan,cycle_over_time)];
    all_motifs_over_time_safe = all_motifs_over_time[find(!isnan,all_motifs_over_time)];

    return cycle_over_time_safe, middleman_over_time_safe, fanin_over_time_safe, fanout_over_time_safe, all_motifs_over_time_safe;
end

function get_CC_static(Spikes_binned,graph,nbins)
    cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = clustered_motifs(graph);
    # look at spikes and calculate a population average for each motif at every timestep
    #=step = 5;
    nbins = convert(Int64,floor.(run_total/step)-1);
    time = step*[1:1:nbins]-step+1;
    time = time[1];=#
    cycle_timecourse = zeros(nbins,);
    middle_timecourse = zeros(nbins,);
    fanin_timecourse = zeros(nbins,);
    fanout_timecourse = zeros(nbins,);
    all_motifs_timecourse = zeros(nbins,);
    for ii = 1:nbins
        cycle_mu = [];
        middleman_mu = [];
        fanin_mu = [];
        fanout_mu = [];
        all_motifs_mu = [];
        for jj = 1:N
            # determine if unit jj spiked in this time step ii
            # if so, add each of its motif CCs to the population average for motifs then
            if Spikes_binned[jj,ii] == 1
                push!(cycle_mu,cycle_CC[jj]);
                push!(middleman_mu,middleman_CC[jj]);
                push!(fanin_mu,fanin_CC[jj]);
                push!(fanout_mu,fanout_CC[jj]);
                push!(all_motifs_mu,all_motifs_CC[jj]);
            end
        end
        # find the population average for each motif at timestep ii
        if !isempty(cycle_mu)
            cycle_timecourse[ii] = mean(cycle_mu);
        end
        if !isempty(middleman_mu)
            middle_timecourse[ii] = mean(middleman_mu);
        end
        if !isempty(fanin_mu)
            fanin_timecourse[ii] = mean(fanin_mu);
        end
        if !isempty(fanout_mu)
            fanout_timecourse[ii] = mean(fanout_mu);
        end
        if !isempty(all_motifs_mu)
            all_motifs_timecourse[ii] = mean(all_motifs_mu);
        end
    end
    # should visualize the distribution of clustering coefficients (amongst units, amongst timepoints)
    return cycle_timecourse, middle_timecourse, fanin_timecourse, fanout_timecourse, all_motifs_timecourse;
end

function gen_net()
    param = [.31,.22,.3,2];
    p_ie = param[1]
    p_ei = param[2]
    p_ii = param[3]
    R    = param[4]
    p_ee = .2

    projected = randperm(N)[1:num_projected]

    cluster = 50 # number of clusters

    p_ee_out = (p_ee*cluster)/(R+cluster-1)
    p_ee_in  = p_ee_out*R

    p_ei_out =  (p_ei*cluster)/(R+cluster-1)
    p_ei_in  =  p_ei_out*R

    LOG_RAND_sigma = 1
    LOG_RAND_mu = log(1) - 0.5*(LOG_RAND_sigma*LOG_RAND_sigma) # this condition ensures that the mean of the new distributoin = 1

    LOG_RAND_sigmaInh = 0.1  # suggesting we hold these constant and only vary excitatory connections
    LOG_RAND_muInh = log(1) - 0.5*(LOG_RAND_sigma*LOG_RAND_sigmaInh)# (so we aren't scaling total weight as we explore heavy-tailedness)

    Network_holder = Network_Generation(Ne,Ni,p_ee_in,p_ee_out,cluster,p_ii,p_ie,p_ei_in,p_ei_out,log_mean,log_sigma,log_mean,log_sigma,log_mean,log_sigma,log_mean,log_sigma);
    we = Network_holder[1]*1*nS
    wi = Network_holder[2]*10*nS

    t=0
    delay=0
    vm= -60*ones(N)+10*rand(N)
    w=zeros(N)
    gE=zeros(N)
    gI=zeros(N)

    wp = poisson_network(poisson_probability,num_poisson,num_projected,poisson_mean_strength,poisson_varriance)*5*nS
    stim = poisson_stimulis_gen(num_poisson,poisson_spikerate,dt,stim_in,taup,poisson_max,stim_in) # give 10 ms of decay time before disregarding
    gP_full = wp'*stim;

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
    run!(net,run_total/dt,gP_full,projected)
    return net.tSpike,we+wi;
end

function compute_recruitment2(W,s_all,tt)
    #= This function returns the recruitment network, which contains the Wij
    value of each real connection that was functionally active in a sequential
    way.
    Since we are discontinuous through time, we will have BLOCKS. Each block
    will have its own recruitment network.
    =#
    Recruitment = zeros(size(W)[1],size(W)[2]); # dimensions of synaptic weight matrix through time
    Nunits = size(W)[1];
    for ii = 1:Nunits
        eidx = find(W[:,ii].!=0);
        edge_vals = W[eidx,ii];
        Nedges = length(eidx);
        # ^ this'll give us the synaptic connections
        # go through every time step in s_all1 and find when ii spiked at t and any
        # of the units in eidx spiked at t+1.
        # find if neuron ii spiked at this time
        if s_all[ii,tt] != 0
            for jj = 1:Nedges # for every neuron jj that is synaptically connected to ii
                if s_all[eidx[jj],tt]==1 # if neuron jj spiked at time t+1
                    Recruitment[eidx[jj],ii] = W[eidx[jj],ii]; # add that synaptic weight to the recruitment network matrix
                    # otherwise recruitment Wij remains at zero
                    # remember that i is the neuron receiving projection, j sending projection
                end
            end
        end
    end
    return Recruitment;
end

function compute_recruitment3(W,s_all,tt)
    #= This function returns the recruitment network, which contains the Wij
    value of each real connection that was functionally active in a sequential
    way.
    Since we are discontinuous through time, we will have BLOCKS. Each block
    will have its own recruitment network.
    =#
    Recruitment = zeros(size(W)[1],size(W)[2]); # dimensions of synaptic weight matrix through time
    Nunits = size(W)[1];
    for ii = 1:Nunits
        eidx = find(W[:,ii].!=0);
        edge_vals = W[eidx,ii];
        Nedges = length(eidx);
        # ^ this'll give us the synaptic connections
        # go through every time step in s_all1 and find when ii spiked at t and any
        # of the units in eidx spiked at t+1.
        # find if neuron ii spiked at this time
        if s_all[ii][tt] != 0
            for jj = 1:Nedges # for every neuron jj that is synaptically connected to ii
                if s_all[eidx[jj]][tt]==1 # if neuron jj spiked at time t+1
                    Recruitment[eidx[jj],ii] = W[eidx[jj],ii]; # add that synaptic weight to the recruitment network matrix
                    # otherwise recruitment Wij remains at zero
                    # remember that i is the neuron receiving projection, j sending projection
                end
            end
        end
    end
    return Recruitment;
end

function motifs_through_time(W,Spikes_binned,nbins,bin)
    # revision as of 20180814 - removed nanfilt, made nans into zeros instead, so population mean reflects all the units with 0 denom
    cycle_over_time = zeros(size(Spikes_binned)[2]-1);
	middleman_over_time = zeros(size(Spikes_binned)[2]-1);
	fanin_over_time = zeros(size(Spikes_binned)[2]-1);
	fanout_over_time = zeros(size(Spikes_binned)[2]-1);
	all_motifs_over_time = zeros(size(Spikes_binned)[2]-1);
    isempty_recruit = zeros(convert(Int64,nbins-1),);
    for tt = 1:convert(Int64,nbins-1)
        recruit = compute_recruitment_delay(W,Spikes_binned,tt,bin);
        # determine whether the recruitment network is all zeros
        # - that is one of the only ways I can imagine there being 0's for all motif CC's in a time bin
        # if it is all zeros, account for it in a separate variable,
        # and don't bother with calling clustered_motifs.

        if isempty(find(recruit.!=0))
            # if none of the values in the recruitment network are nonzero, then motifs are all zero,
            cycle_over_time[tt] = 0;
            middleman_over_time[tt] = 0;
            fanin_over_time[tt] = 0;
            fanout_over_time[tt] = 0;
            all_motifs_over_time[tt] = 0;
            # and indicate this fact in isempty_recruit variable.
            isempty_recruit[tt] = 1;
        else

            # after getting the recruitment graph for this timebin, you can now determine
            # how many units are actually active in the recruitment graph, and then calculate
            # motifs using only that subgraph.

            activeidx = [];
            rowidx = [];
            colidx = [];
            for ii = 1:size(recruit)[1]
                if !isempty(find(recruit[:,ii].!=0))
                    push!(activeidx,ii);
                    push!(colidx,ii);
                elseif !isempty(find(recruit[ii,:].!=0))
                    push!(activeidx,ii);
                    push!(rowidx,ii);
                end
            end
            subrecruit = recruit[activeidx,activeidx];


            cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = clustered_motifs(subrecruit);

            cycle_CC[find(isnan.(cycle_CC))] = 0;
            middleman_CC[find(isnan.(middleman_CC))] = 0;
            fanin_CC[find(isnan.(fanin_CC))] = 0;
            fanout_CC[find(isnan.(fanout_CC))] = 0;
            all_motifs_CC[find(isnan.(all_motifs_CC))] = 0;

            cycle_over_time[tt] = mean(cycle_CC);
            middleman_over_time[tt] = mean(middleman_CC);
            fanin_over_time[tt] = mean(fanin_CC);
            fanout_over_time[tt] = mean(fanout_CC);
            all_motifs_over_time[tt] = mean(all_motifs_CC);
        end
    end
    return cycle_over_time,middleman_over_time,fanin_over_time,fanout_over_time,all_motifs_over_time,isempty_recruit;
end

function motifs_through_time_nanfilt(W,Spikes_binned,nbins,bin)
    # revision as of 20180730 - added !isnan filter to calculate population mean motif CC
    cycle_over_time = zeros(size(Spikes_binned)[2]-1);
	middleman_over_time = zeros(size(Spikes_binned)[2]-1);
	fanin_over_time = zeros(size(Spikes_binned)[2]-1);
	fanout_over_time = zeros(size(Spikes_binned)[2]-1);
	all_motifs_over_time = zeros(size(Spikes_binned)[2]-1);
    isempty_recruit = zeros(convert(Int64,nbins-1),);
    for tt = 1:convert(Int64,nbins-1)
        recruit = compute_recruitment_delay(W,Spikes_binned,tt,bin);
        # determine whether the recruitment network is all zeros
        # - that is one of the only ways I can imagine there being 0's for all motif CC's in a time bin
        # if it is all zeros, account for it in a separate variable,
        # and don't bother with calling clustered_motifs.

        if isempty(find(recruit.!=0))
            # if none of the values in the recruitment network are nonzero, then motifs are all zero,
            cycle_over_time[tt] = 0;
            middleman_over_time[tt] = 0;
            fanin_over_time[tt] = 0;
            fanout_over_time[tt] = 0;
            all_motifs_over_time[tt] = 0;
            # and indicate this fact in isempty_recruit variable.
            isempty_recruit[tt] = 1;
        else

            # after getting the recruitment graph for this timebin, you can now determine
            # how many units are actually active in the recruitment graph, and then calculate
            # motifs using only that subgraph.

            activeidx = [];
            rowidx = [];
            colidx = [];
            for ii = 1:size(recruit)[1]
                if !isempty(find(recruit[:,ii].!=0))
                    push!(activeidx,ii);
                    push!(colidx,ii);
                elseif !isempty(find(recruit[ii,:].!=0))
                    push!(activeidx,ii);
                    push!(rowidx,ii);
                end
            end
            subrecruit = recruit[activeidx,activeidx];


            cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = clustered_motifs(subrecruit);
            cycle_over_time[tt] = mean(filter(!isnan, cycle_CC));
            middleman_over_time[tt] = mean(filter(!isnan, middleman_CC));
            fanin_over_time[tt] = mean(filter(!isnan, fanin_CC));
            fanout_over_time[tt] = mean(filter(!isnan, fanout_CC));
            all_motifs_over_time[tt] = mean(filter(!isnan, all_motifs_CC));
        end
    end
    return cycle_over_time,middleman_over_time,fanin_over_time,fanout_over_time,all_motifs_over_time,isempty_recruit;
end

function compute_recruitment_delay(W,s_all,tt,bin) # revision as of 20180809
    #=
    with delay, meaning taking all spikes 20-30ms after the first spike to calculate the recruitment
    =#
    Recruitment = zeros(size(W)); # dimensions of synaptic weight matrix through time
    Nunits = size(W)[1];
    # for this timebin, find the units that spiked at all
    active = find(s_all[:,tt].!=0);
    for ii = 1:length(active)
        eidx = find(W[:,active[ii]].!=0); # all the units which active[ii] projects to in the graph
        #edge_vals = W[eidx,active[ii]]; # get the values of those edges
        Nedges = length(eidx);
        # ^ this'll give us the synaptic connections
        # go through every unit in eidx and find whether any of them spiked btwn
        # the start and end bins relative to tt
        startbin = convert(Int64,5/bin);
        endbin = convert(Int64,20/bin);
        for jj = 1:Nedges
            # for every neuron eidx[jj] that is synaptically connected to active[ii]
            # find all times in whole trial when neuron jj spiked
            jjspikeidx = find(s_all[eidx[jj],:].!=0);
            span = collect(tt+startbin:tt+endbin); # the span of time valid for being recruited
            recruited = intersect(jjspikeidx,span);
            if !isempty(recruited)
                # we know that neuron jj spiked at least once in the desired span after tt (when neuron ii fired)
                # hence, we count it as recruited, and we add that connection to the recruitment weight matrix.
                Recruitment[eidx[jj],active[ii]] = W[eidx[jj],active[ii]];
            end
        end
    end
    return Recruitment;
end

function compute_functional_delay(W,s_all,tt,bin)
    # check to see that recruitment graph is about 1/5th as dense as the functional graph
    Functional = zeros(size(W));
    Nunits = size(W)[1];
    active = find(s_all[:,tt].!=0);
    for ii = 1:length(active)
        startbin = convert(Int64,5/bin);
        endbin = convert(Int64,20/bin);
        for jj = 1:Nunits # for all units in the graph (not just the synaptically connected ones)
            jjspikeidx = find(s_all[jj,:].!=0); # find all time bins when unit jj spiked
            span = collect(tt+startbin:tt+endbin);
            # find the intersection of those bins with the recruitment span
            recruited = intersect(jjspikeidx,span);
            if !isempty(recruited)
                Functional[jj,active[ii]] = 1;
            end
        end
    end
    return Functional;
end


function shuffle_units(group_num,net_num);
    # in order to determine the source of the plane, begin by shuffling the neuron IDs relative to the spike raster
    # and re-running motif analysis. is it then on the same plane?

    io = load("D:/qing/$(group_num)/network/NetworkData$(net_num).jld");
    graph = io["network"]; # NxN matrix of topological weights
    ge = graph[:,1:Ne]./(10.0^-6);
    gi = graph[:,Ne+1:N]./(10.0^-5);
    graph = hcat(ge,gi);

    io = load("D:/qing/$(group_num)/spikes/SpikeData$(net_num).jld");
    spikes = io["spikes"];

    spike_set = [];
    active_inds = [];
    for ii = 1:length(spikes)
        # exclude units that never spiked outside the input stim period (first 30ms)
        post_stim  = (spikes[ii])[find(spikes[ii] .> stim_in)];
        if !isempty(post_stim)
            push!(spike_set, spikes[ii]);
            push!(active_inds,ii);
        end
    end

    # re-separate into excitatory and inhibitory populations
    e_edges = find(active_inds.<=Ne);
    # currently NOT considering units that are cross-type (i.e. e-to-i or i-to-e)
    W_active_e = graph[active_inds[e_edges],active_inds[e_edges]];
    i_edges = find(active_inds.>Ne);
    W_active_i = graph[active_inds[i_edges],active_inds[i_edges]];

    integ_frame = collect(5:30);
    nbins = run_total-maximum(integ_frame)-1;
    Nactive = length(spike_set);
    W_active = graph[active_inds,active_inds];

    Spikes_binned = zeros(Nactive,nbins);

    shuffleidx = randperm(Nactive);
    # shuffle the weight matrix
    W_shuffled = W_active[shuffleidx,shuffleidx];

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

    cycle_recruit_CC, middle_recruit_CC, fanin_recruit_CC, fanout_recruit_CC, all_recruit_CC, isempty_recruit = motifs_through_time(W_shuffled,Spikes_binned,nbins);
    fname = "/Users/maclean lab/Documents/qing/$(net_num)/nulls/shuffle_units_temp_fullres.mat";
    file = matopen(fname,"w");
    write(file,"fanout_recruit_CC",fanout_recruit_CC);
    write(file,"fanin_recruit_CC",fanin_recruit_CC);
    write(file,"middle_recruit_CC",middle_recruit_CC);
    write(file,"isempty_recruit",isempty_recruit);
    close(file);

    shuffleidx = randperm(length(e_edges));
    W_e_shuffled = W_active_e[shuffleidx,shuffleidx];

    graph = W_e_shuffled;
    subspikes = Spikes_binned[e_edges,:];
    fname = "/Users/maclean lab/Documents/qing/$(net_num)/nulls/shuffle_units_temp_fullres_E.mat";
    excit_inhib_analysis(graph,subspikes,nbins,fname);

    print("Done!");

end
