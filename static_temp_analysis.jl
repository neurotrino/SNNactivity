#=
This script examines the current lowest rate network in "D:/qing/0/"
As of 180724, that network is 22000.
=#

include("/Users/maclean lab/Documents/qing/SNNactivity/jason_proj_2.jl");
include("/Users/maclean lab/Documents/qing/SNNactivity/equiv_demo.jl");
using MAT;

function main()
    # load the files we need
    # i.e. the lowest rate network we found in batch run 2
    cd("D:/");
    io = load("D:/qing/0/network/NetworkData22000.jld");
    graph = io["network"]; # NxN matrix of topological weights
    io = load("D:/qing/0/spikes/SpikeData22000.jld");
    spikes = io["spikes"]; # Nx: matrix of spike times (in ms)
    io = load("D:/qing/0/scores/ScoreData22000.jld");
    scores = io["scores"];
    # scores in array [e_rate,i_rate,frac_filled,e_synch_score,i_synch_score,
    # total_synch_score,e_branch_score,i_branch_score,total_branch_score]
    e_rate = scores[1];
    i_rate = scores[2];
    frac_time_filled = scores[3];
    e_synch = scores[4];
    i_synch = scores[5];
    total_synch = scores[6];
    e_branch = scores[7];
    i_branch = scores[8];
    total_branch = scores[9];

    step = 1;
    nbins = convert(Int64,floor.(run_total/step)-1);
    Spikes_binned = zeros(N,nbins);
    time = step*[1:1:nbins]-step+1;
    time = time[1];
    for ii = 1:N
        # find the indices for neuron ii for which we'll put a 1 in Spikes_binned
        if !isempty(spikes[ii,])
            indices = floor.(spikes[ii,]./step);
            indices = indices[find(indices.<=nbins)];
            indices = indices[find(indices.>0)];
            for jj = 1:length(indices)
                Spikes_binned[ii,convert(Int64,indices[jj])] = 1;
            end
        end
        # in the end we'll get a '1' in the right [neuron,timebin] array slot for each spike of each neuron
    end

    cycle_static_CC, middle_static_CC, fanin_static_CC, fanout_static_CC, all_static_CC = get_CC_static(Spikes_binned,graph,nbins);

    # save the static vars - use the step size in the file name
    fname = "/Users/maclean lab/Documents/qing/static_res$(step)ms_22000.mat";
    file = matopen(fname,"w");
    write(file,"fanout_static_CC",fanout_static_CC);
    write(file,"fanin_static_CC",fanin_static_CC);
    write(file,"middle_static_CC",middle_static_CC);
    close(file);

    cycle_recruit_CC, middle_recruit_CC, fanin_recruit_CC, fanout_recruit_CC, all_recruit_CC = motifs_through_time(graph,Spikes_binned,nbins);
    # update 20180725: now that we are trying greater dt's, this should work better
    # now that we have both types of motif timecourses (for the population), let's examine how the motifs change over time

    # save the temp vars (motifs calculated from recruitment graph)
    fname = "/Users/maclean lab/Documents/qing/temp_res$(step)ms_22000.mat";
    file = matopen(fname,"w");
    write(file,"fanout_recruit_CC",fanout_recruit_CC);
    write(file,"fanin_recruit_CC",fanin_recruit_CC);
    write(file,"middle_recruit_CC",middle_recruit_CC);
    close(file);

    # subsample twice, let's say
    # also we're using step size of 1 (full resolution)
    fname = "/Users/maclean lab/Documents/qing/temp_res1ms_22000_sub1.mat";
    subsample_analysis(graph,Spikes_binned,nbins,fname);
    fname = "/Users/maclean lab/Documents/qing/temp_res1ms_22000_sub2.mat";
    subsample_analysis(graph,Spikes_binned,nbins,fname);

    fname = "/Users/maclean lab/Documents/qing/temp_res1ms_22000_sub3.mat";
    subsample_analysis(graph,Spikes_binned,nbins,fname);
    fname = "/Users/maclean lab/Documents/qing/temp_res1ms_22000_sub4.mat";
    subsample_analysis(graph,Spikes_binned,nbins,fname);

    # wait, we should only subsample from the excitatory units
    fname = "/Users/maclean lab/Documents/qing/temp_res1ms_22000_sub5.mat";
    subsample_analysis(graph,Spikes_binned,nbins,fname);
    fname = "/Users/maclean lab/Documents/qing/temp_res1ms_22000_sub6.mat";
    subsample_analysis(graph,Spikes_binned,nbins,fname);

    # just all of the excitatory units at 1ms res
    fname = "/Users/maclean lab/Documents/qing/temp_res1ms_22000_excit.mat";
    excit_inhib_analysis(graph,Spikes_binned,nbins,fname,"E");
    # now all of the inhib units at 1ms res
    fname = "/Users/maclean lab/Documents/qing/temp_res1ms_22000_inhib.mat";
    excit_inhib_analysis(graph,Spikes_binned,nbins,fname,"I");

    fname = "/Users/maclean lab/Documents/qing/temp_res5ms_22000_excit.mat";
    excit_inhib_analysis(graph,Spikes_binned,nbins,fname,"E");

end

function subsample_analysis(Nsamp,graph,Spikes_binned,nbins,fname)
    # randomly sample some (Nsamp) of the units in our lowest-rate graph
    # leave this running, and then you can take off

    # randomly choose indices btwn 1 and Nsamp
    inds = randperm(size(graph)[1])[1:Nsamp];
    # select those rows and columns out of the graph to make a new W
    sort!(inds);
    W = graph[inds,inds];
    # select only those relevant rows out of Spikes_binned
    subspikes = Spikes_binned[inds,:];
    # call our motif clustering function
    cycle_recruit_CC, middle_recruit_CC, fanin_recruit_CC, fanout_recruit_CC, all_recruit_CC, isempty_recruit = motifs_through_time(W,subspikes,nbins);

    # save what we need to file
    file = matopen(fname,"w");
    write(file,"fanout_recruit_CC",fanout_recruit_CC);
    write(file,"fanin_recruit_CC",fanin_recruit_CC);
    write(file,"middle_recruit_CC",middle_recruit_CC);
    write(file,"all_recruit_CC",all_recruit_CC);
    write(file,"isempty_recruit",isempty_recruit);
    write(file,"inds",inds); # in case you want to keep track of which units of the full (active) net we're working with
    close(file);

end

function excit_inhib_analysis(graph,subspikes,nbins,fname,bin);
    # the graph and subspikes variables are already the correct subset of the original graph and its spikes
        # call our motif clustering function
        cycle_recruit_CC, middle_recruit_CC, fanin_recruit_CC, fanout_recruit_CC, all_recruit_CC, isempty_recruit = motifs_through_time(graph,subspikes,nbins,bin);

        # save what we need to file
        file = matopen(fname,"w");
        write(file,"fanout_recruit_CC",fanout_recruit_CC);
        write(file,"fanin_recruit_CC",fanin_recruit_CC);
        write(file,"middle_recruit_CC",middle_recruit_CC);
        write(file,"isempty_recruit",isempty_recruit);
        close(file);
end

#=
function recruit_time_test(graph, Spikes_binned)
    # determine when recruitment really occurs
    # 1. for some discrete 1ms timebin, determine which were the active units
    graph = W_active;
    start = 77;
    active = find(Spikes_binned[:,start].==1);
    # 2. for all the units that they're connected to, what's the count of active units?
    conn_idx = [];
    for nn = 1:length(active)
        push!(conn_idx,find(graph[:,active[nn]].!=0));
    end
    total_recruited = zeros(35,);
    for tt = 1:35
        for nn = 1:length(active)
            total_recruited[tt] += length(find(Spikes_binned[conn_idx[nn],tt].==1));
        end
    end
    # 3. when do we see a peak in the active units? - find the time of peak for this time bin
    recruit_peak = find(total_recruited.==maximum(total_recruited));
    # 4. sample another 1ms timebin as our starting point, and go from there.
    start = 777;
    active = find(Spikes_binned[:,start].==1);
    conn_idx = [];
    for nn = 1:length(active)
        push!(conn_idx,find(graph[:,active[nn]].!=0));
    end
    total_recruited = zeros(35,);
    for tt = 1:35
        for nn = 1:length(active)
            total_recruited[tt] += length(find(Spikes_binned[conn_idx[nn],tt].==1));
        end
    end
    recruit_peak = find(total_recruited.==maximum(total_recruited));

    # 5. maybe go through all 1ms time steps, and find the distribution of peak lags
    lag = 50;
    recruit_peak = zeros(lag,size(Spikes_binned)[2]-stim_in);
    for ii = stim_in+1:size(Spikes_binned)[2] # start after input stimulus injection stops
        start = ii;
        active = find(Spikes_binned[:,start].==1);
        conn_idx = [];
        for nn = 1:length(active)
            push!(conn_idx,find(graph[:,active[nn]].!=0));
        end
        total_recruited = zeros(lag,);
        for tt = 1:lag
            for nn = 1:length(active)
                # for all units active at any time t, at the time afterwards
                total_recruited[tt] += length(find(Spikes_binned[conn_idx[nn][:],tt].==1));
                # for active neuron nn, look at all the units it's connected to (in array conn_idx[nn])
                # and determine the
            end
        end
        recruit_peak[:,ii-stim_in] = total_recruited;
    end
    recruit_dist = zeros(size(recruit_peak)[1],);
    for ii = 1:size(recruit_peak)[1]
        recruit_dist[ii] = sum(recruit_peak[ii,:]);
    end
end
=#

function recruit_lag_test(graph, Spikes_binned, maxlag)
    # do this for the kinds of graphs we're interested in.
    # that is, examine if it differs between the two base networks
    # and between excitatory and inhibitory subpopulations
    lags = collect(1:maxlag);
    lagcount = zeros(maxlag,);
    for tt = stim_in+1:size(Spikes_binned)[2]-maxlag
        active = find(Spikes_binned[:,tt].==1);
        conn_idx = [];
        for nn = 1:length(active)
            push!(conn_idx,find(graph[:,active[nn]].!=0));
        end
        for uu = 1:maxlag
            for ii = 1:length(active)
                 for jj = 1:length(conn_idx[ii])
                     if (Spikes_binned[conn_idx[ii][jj],tt+lags[uu]]==1)
                         lagcount[uu] = lagcount[uu]+1;
                     end
                 end
             end
        end
    end
    Plots.plot(lagcount);
end

function toy(graph)
    adj_mat = graph[21:320,21:320];
    cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = clustered_motifs(adj_mat);
    fname = "/Users/maclean lab/Documents/qing/data/isabel/subset_motifs.mat";
    file = matopen(fname,"w");
    write(file,"adj_mat",adj_mat);
    write(file,"fanout_CC",fanout_CC);
    write(file,"fanin_CC",fanin_CC);
    write(file,"middleman_CC",middleman_CC);
    write(file,"cycle_CC",cycle_CC);
    write(file,"all_motifs_CC",all_motifs_CC);
    close(file);
end

function analyze_rates(group_num,net_num)
    # 1. load in network and run(s) of interest.
    # 2. calculate instantaneous spike rate of each unit and the population
    # 3. for all networks that succeeded, what did their rates look like across time?
    # 4. what was happening with the input pool immediately after input ceased?
    # 5. what is happening over time with the units that have recurrent connections?

    # load in the topology
    io = load("D:/qing/$(group_num)/network/NetworkData$(net_num).jld");
    graph = io["network"]; # NxN matrix of topological weights
    # normalize according to source, since the graph returned from score() function and saved is we+wi
    ge = graph[:,1:Ne]./(10.0^-6);
    gi = graph[:,Ne+1:N]./(10.0^-5);
    graph = hcat(ge,gi);

    # read in the spike sets we're interested in
    io = load("D:/qing/$(group_num)/spikes/SpikeData$(net_num).jld");
    spikes = io["spikes"];

    # read in the input data we're interested in
    io = load("D:/qing/$(group_num)/poisson_in_data/PoissonData$(net_num).jld");
    stim = io["poisson_stim"]; # 500(units)x300 Poisson input stim
    projected = io["input_neurons"]; # indices of input units

    rates = rate_est_active(spikes);
    e_rate = rates[1];
    i_rate = rates[2];

    t_res = 10; # resolution by which we calculate the instantaneous firing rate

    spike_set = spikes;

    #=
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
    =#
    Nactive = N;
    Spikes_binned = zeros(Nactive,run_total-1);

    for ii = 1:Nactive
        # find the indices for neuron ii for which we'll put a 1 in Spikes_binned
        if !isempty(spike_set[ii,])
            indices = floor.(spike_set[ii]);
            indices = indices[find(indices.<=run_total-1)];
            indices = indices[find(indices.>0)]; # get rid of the first zero
            for jj = 1:length(indices)
                Spikes_binned[ii,convert(Int64,indices[jj])] = 1;
            end
        end
    end

    # calculate 'instantaneous' firing rate
    nbins = convert(Int64,ceil.((run_total-1)/t_res));
    InstRate = zeros(Nactive, nbins);
    for ii = 1:nbins # for every time bin
        # count the number of spikes which occurred in that bin for each neuron
        ct = zeros(Nactive,);

    end

    Pop_InstRate = mean(Nactive,:);

end
