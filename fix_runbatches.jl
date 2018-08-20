# Loads in networks of interest, fixes some parameters and changes others.
# Re-runs the network and records resulting spikes and scores.

using MAT
include("C:/Users/maclean lab/Documents/qing/SNNactivity/jason_proj_2.jl")
include("C:/Users/maclean lab/Documents/qing/SNNactivity/equiv_demo.jl");
include("C:/Users/maclean lab/Documents/qing/SNNactivity/temporal_motifs.jl");
include("C:/Users/maclean lab/Documents/qing/SNNactivity/static_temp_analysis.jl");
#include("C:/Users/maclean lab/Documents/qing/SNNactivity/fix_runbatches.jl");

#= outer structure I envision would include calling everything needed to generate a 'batch'
    - creating the static topology, poisson input, and input units on which we
    will make modifications
    - the variations we are interested in are:
    - 1. fix all three
    - 2. fix topology and input units but regenerate poisson input train
    - 3. fix topology and poisson input but regenerate choice of input units
    - 4. fix topology while regenerating both input units and the poisson input train
    - (not necessarily in that order, but the function names will be self-explanatory)
    - I think the rest of the combinations are either meaningless or had been overdone
    - for each of these, we will run # times (for however many runs we want to examine of each condition)
    while keeping track of the score. the score will then include the branching parameter, of each unit
    and/or the average across the network. make sure you save everything somewhere reasonably organized.
=#
Nbatch = 500
run = 3;
group_num=1;
net_num=6974;
# make folders to save data

function run_batches(run,Nbatch,group_num,net_num)
    # repeat each condition some number of times, save output scores

    if run == 1
        #mkdir("C:/Users/maclean lab/Documents/qing/$(net_num)");
        mkdir("C:/Users/maclean lab/Documents/qing/$(net_num)/fix_all");
    end

    param = [.31,.22,.3,2]
    p_ie = param[1];
    p_ei = param[2];
    p_ii = param[3];
    R    = param[4];
    p_ee = .2;
    cluster = 50;
    p_ee_out = (p_ee*cluster)/(R+cluster-1);
    p_ee_in  = p_ee_out*R;
    p_ei_out =  (p_ei*cluster)/(R+cluster-1);
    p_ei_in  =  p_ei_out*R;
    t=0;
    delay=0;
    vm= -60*ones(N)+10*rand(N);
    w=zeros(N);
    gE=zeros(N);
    gI=zeros(N);

    #=generated_projected = randperm(N)[1:num_projected]
    wp = poisson_network(poisson_probability,num_poisson,num_projected,poisson_mean_strength,poisson_variance)*5*nS
    stim = poisson_stimulis_gen(num_poisson,poisson_spikerate,dt,stim_in,taup,poisson_max,stim_in) # give 10 ms of decay time before disregarding
    generated_stim = wp'*stim
    Network_holder = Network_Generation(Ne,Ni,p_ee_in,p_ee_out,cluster,p_ii,p_ie,p_ei_in,p_ei_out,log_mean,log_sigma,log_mean,log_sigma,log_mean,log_sigma,log_mean,log_sigma);
    we = Network_holder[1]*1*nS
    wi = Network_holder[2]*10*nS
    # find some way to translate back into extractable generated_net form
    generated_net = we+wi # there, fixed, that works=#

    # load in graph, input units, Poisson input train
    cd("D:/");
    io = load("D:/qing/$(group_num)/network/NetworkData$(net_num).jld");
    graph = io["network"]; # NxN matrix of topological weights
    io = load("D:/qing/$(group_num)/poisson_in_data/PoissonData$(net_num).jld");
    generated_stim = io["poisson_stim"]; # 500(units)x300 Poisson input stim
    generated_projected = io["input_neurons"]; # indices of input units
    cd("C:/");


    # fixing everything, we want to observe how the spikes and scores change
    batchscores = zeros(Nbatch,9);
    batchspikes = [];
    batchic = [];
    for jj = 1:Nbatch
        batchscores[jj,:],spikes,ic = fix_all(param,graph,generated_stim,generated_projected);
        push!(batchspikes,spikes);
        push!(batchic,ic);
    end
    # save all scores to some data file
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_all/Run$(run)_Spikes.jld";
    save(fname,"batchspikes",batchspikes);
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_all/Run$(run)_Scores.jld";
    save(fname,"batchscores",batchscores);
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_all/Run$(run)_IC.jld";
    save(fname,"batchic",batchic);

    #= fix topology
    if run == 1
        mkdir("C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo");
    end
    batchscores = zeros(Nbatch,9);
    batchspikes = [];
    batchstim = [];
    batchprojected = zeros(Nbatch,num_projected);
    for jj = 1:Nbatch
        batchscores[jj,:],spikes,batchprojected[jj,:],stim = fix_topology(param,graph);
        push!(batchspikes,spikes);
        push!(batchstim,stim);
    end
    # and save
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo/Run$(run)_Scores.jld";
    save(fname,"batchspikes",batchspikes);
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo/Run$(run)_Spikes.jld";
    save(fname,"batchscores",batchscores);
    =#

    # fix topology and input units, regenerate the poisson input trains
    if run == 1
        mkdir("C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo_proj")
    end
    batchscores = zeros(Nbatch,9);
    batchspikes = [];
    batchstim = [];
    batchic = [];
    for jj = 1:Nbatch
        batchscores[jj,:],spikes,stim,ic = fix_topology_projected(param,graph,generated_projected);
        push!(batchspikes,spikes);
        push!(batchstim,stim);
        push!(batchic,ic);
    end
    # and save
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo_proj/Run$(run)_Scores.jld";
    save(fname,"batchspikes",batchspikes);
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo_proj/Run$(run)_Spikes.jld";
    save(fname,"batchscores",batchscores);
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo_proj/Run$(run)_Stim.jld";
    save(fname,"batchstim",batchstim);
    # save the stim, the projected units are static and already in the original D: directory
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo_proj/Run$(run)_IC.jld";
    save(fname,"batchic",batchic);

    # fix topology and poisson input trains, change input units
    if run == 1
        mkdir("C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo_stim")
    end

    batchscores = zeros(Nbatch,9);
    batchspikes = [];
    batchic = [];
    batchprojected = zeros(Nbatch,num_projected);
    for jj = 1:Nbatch
        batchscores[jj,:],spikes,batchprojected[jj,:],ic = fix_topology_stim(param,graph,generated_stim);
        push!(batchspikes,spikes);
        push!(batchic,ic);
    end
    # and save
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo_stim/Run$(run)_Scores.jld";
    save(fname,"batchscores",batchscores);
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo_stim/Run$(run)_Spikes.jld";
    save(fname,"batchspikes",batchspikes);
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo_stim/Run$(run)_Projected.jld";
    save(fname,"batchprojected",batchprojected);
    # save the projected units, the stim is static and already in original D: directory
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_topo_stim/Run$(run)_IC.jld";
    save(fname,"batchic",batchic);

end

function fix_all_ic(param, generated_net, generated_stim, generated_projected, saved_ic)
    # score(param;trials=0,generated_net=0,plot=false,generated_stim=0,generated_projected=0,saved_ic=0)
    # this function reproduces the exact set of parameters, including initial conditions
    (score_vals, tSpike, projected, we+wi, gP_full, ic) = score(param;trials=0,generated_net=generated_net,plot=false,generated_stim=generated_stim,generated_projected=generated_projected,saved_ic)
    return score_vals, tSpike, ic
end

function fix_all(param,generated_net,generated_stim,generated_projected) # is this deterministic?
    # returns score=[excitatory rate, inhibitory rate, last spike time, network branching score]
    (score_vals, tSpike, projected, we+wi, gP_full, ic) = score(param;trials=0,generated_net=generated_net,plot=false,generated_stim=generated_stim,generated_projected=generated_projected,saved_ic=0)
    # note that plot=true will just give you the raster
    return score_vals, tSpike, ic
end

function fix_topology(param,generated_net)
    (score_vals, tSpike, projected, we+wi, gP_full, ic) = score(param;trials=0,generated_net=generated_net,plot=false,generated_stim=0,generated_projected=0,saved_ic=0)
    return score_vals, tSpike, projected, gP_full, ic;
end

function fix_topology_projected(param,generated_net,generated_projected)
    # as the projected units are static, save the Poisson stim each time (gP_full)
    (score_vals, tSpike, projected, we+wi, gP_full, ic) = score(param;trials=0,generated_net=generated_net,plot=false,generated_stim=0,generated_projected=generated_projected,saved_ic=0)
    return score_vals, tSpike, gP_full, ic;
end

function fix_topology_stim(param,generated_net,generated_stim)
    # as the Poisson stim is static, save the input units each time (projected)
    (score_vals, tSpike, projected, we+wi, gP_full, ic) = score(param;trials=0,generated_net=generated_net,plot=false,generated_stim=generated_stim,generated_projected=0,saved_ic=0)
    return score_vals, tSpike, projected, ic;
end

function analyze_batches(run,group_num,net_num,fixtype,bin)
    io = load("C:/Users/maclean lab/Documents/qing/$(net_num)/$(fixtype)/Run$(run)_Spikes.jld");
    SpikeData = io["batchspikes"];
    io = load("C:/Users/maclean lab/Documents/qing/$(net_num)/$(fixtype)/Run$(run)_Scores.jld");
    ScoreData = io["batchscores"];
    #print("In looking at run $(run) of network $(net_num):\n\n");

    inds = find(ScoreData[:,3] .> .95);
    #print("Out of $(length(SpikeData)) trials in this batch, $(length(inds)) made it to completion.\n\n");
    #print("Each of the successful trials in this batch had rates:\n\n")
    #print(ScoreData[inds,1]);
    # toggle this so we examine low rate networks that did not succeed
    # ofinterest = intersect(find(ScoreData[:,3].<.5),find(ScoreData[:,3].>.2),find(ScoreData[:,1].<5),find(ScoreData[:,1].>2));
    ofinterest = intersect(find(ScoreData[:,3].>.95),find(ScoreData[:,1].<5));

    # compare with original network's scores
    io = load("D:/qing/$(group_num)/scores/ScoreData$(net_num).jld");
    orig_scores = io["scores"];
    #print("\n\nThe original network's scores were:\n\n");
    #print(orig_scores);

    if !isempty(ofinterest)

        # run motif analysis on each satisfactory trial and save
        for trialidx = 1:length(ofinterest)

            # load in the topology
            io = load("D:/qing/$(group_num)/network/NetworkData$(net_num).jld");
            graph = io["network"]; # NxN matrix of topological weights
            # normalize according to source, since the graph returned from score() function and saved is we+wi
            ge = graph[:,1:Ne]./(10.0^-6);
            gi = graph[:,Ne+1:N]./(10.0^-5);
            graph = hcat(ge,gi);

            # take the spike data from the run where the score satisfied our criteria (completion and low rate)
            spikes = SpikeData[ofinterest[trialidx]];
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

            integ_frame = collect(5:20);
            nbins = convert(Int64,(run_total-maximum(integ_frame))/bin-1);
            Nactive = length(spike_set);
            W_active = graph[active_inds,active_inds];

            Spikes_binned = zeros(Nactive,nbins);

            for ii = 1:Nactive
                # find the indices for neuron ii for which we'll put a 1 in Spikes_binned
                if !isempty(spike_set[ii,])
                    idx = floor.(spike_set[ii]./bin);
                    idx = idx[find(idx.<=nbins)];
                    idx = idx[find(idx.>0)];
                    for jj = 1:length(idx)
                        Spikes_binned[ii,convert(Int64,idx[jj])] = 1;
                    end
                end
            end

            NetID = ofinterest[trialidx];

            if run == 1 && trialidx == 1
                #mkdir("C:/Users/maclean lab/Documents/qing/$(net_num)/motif_data");
                mkdir("C:/Users/maclean lab/Documents/qing/$(net_num)/motif_data/$(fixtype)/newCC/");
            end

#=
            cycle_recruit_CC, middle_recruit_CC, fanin_recruit_CC, fanout_recruit_CC, all_recruit_CC, isempty_recruit = motifs_through_time(W_active,Spikes_binned,nbins);
            fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/motif_data/$(fixtype)/Run$(run)_temp_fullres_$(NetID).mat";
            file = matopen(fname,"w");
            write(file,"fanout_recruit_CC",fanout_recruit_CC);
            write(file,"fanin_recruit_CC",fanin_recruit_CC);
            write(file,"middle_recruit_CC",middle_recruit_CC);
            write(file,"isempty_recruit",isempty_recruit);
            close(file);
            =#

            graph = W_active_e;
            subspikes = Spikes_binned[e_edges,:];
            fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/motif_data/$(fixtype)/newCC/Run$(run)_subgraph_e_$(NetID).mat";
            excit_inhib_analysis(graph,subspikes,nbins,fname,bin);

            #=graph = W_active_i;
            subspikes = Spikes_binned[i_edges,:];
            fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/motif_data/Run$(run)_temp_fullres_I_$(NetID).mat";
            excit_inhib_analysis(graph,subspikes,nbins,fname);\=#
            print("Done with that one!");

        end
    end

end

 #=
# use revised branching_param I wrote in jason_proj_2.jl
function branching_param(tSpike,W)
    # args are net.tSpike in form [N x num_spikes_per_neuron] array
    # returns the branching scores for every neuron and for the network on average
    # I need to get a better grasp of thse size and contents of we, wi
    # so far assuming w is available to me and is what it is (directed weight matrix of NxN values)
    units_bscore = zeros(N);
    # discretize spike times into dt Matrix
    dt = .1;
    nbins = run_total/dt;
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
    for ii = 1:N # for every neuron in the network
        if length(tSpike[ii,]) > 0
            # for all edges this neuron ii has in the network
            # (and first we have to find them)
            edges = find(W[ii,:].!=0);
            Nedges = length(edges);
            # also this is all matlab syntax that i'll have to fix
            # find all the time bins t in which neuron ii spiked,
            # examine the time bins t+1 for each of the neurons in edges
            cuebin = find(Spikes_binned[ii,:].==1);
            gobin = cuebin+1;
            # effectively you're doing the recruitment graph here
            Jcount = zeros(Nedges+1); # array to increment the count for how many times those number of neurons were recruited
            # we also have to account for the probability that zero were recruited (hence the +1, and we will index as though 0=1)
            for tt = 1:length(gobin) # for every time that neuron ii spiked:
                # out of the neurons that it's connected to (indexed in edges), how many of those had a spike (value of 1 in Spikes_binned) at time t+1?
                ct = length(find(Spikes_binned[edges,gobin[tt]].==1)); # if I'm doing julia right, this will give us the number of edges in this time which spiked (were recruited)
                Jcount[ct+1] = Jcount[ct+1] + 1; # increment the count for the occurence of this recruitment number
            end
            # calculate using equation 1 from Beggs & Plenz, 2003
            P = Jcount./length(cuebin); # probability of observing those many descendants
            k = collect(1:length(Jcount))-1
            units_bscore[ii] = sum(k.*P);
        else
            units_bscore[ii] = NaN;
        end
    end
    net_bscore = mean(filter(!isnan,units_bscore));
    return net_bscore, units_bscore;
end
=#
