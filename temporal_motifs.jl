# New Recruitment Calculations
# begun on 20180727
# 1. load in low-rate network - the graph, the spikes, and the scores
# 2. compute a binary Spikes_binned matrix of N x run_total-1 dimensions. remove units that are never active
# 3. for every time point (1ms), except the last ~33ms, calculate the recruitment network,
# 4. and using that, calculate the population motifs.
# 5. save the data for the isomorphic motifs and overal CC to .mat files.

include("/Users/maclean lab/Documents/qing/SNNactivity/jason_proj_2.jl");
include("/Users/maclean lab/Documents/qing/SNNactivity/equiv_demo.jl");
include("/Users/maclean lab/Documents/qing/SNNactivity/static_temp_analysis.jl");
using MAT;

function main()
    cd("D:/");
    io = load("D:/qing/0/network/NetworkData22000.jld");
    graph = io["network"]; # NxN matrix of topological weights
    # normalize according to source, since the graph returned from score() function and saved is we+wi
    ge = graph[:,1:Ne]./(10.0^-6);
    gi = graph[:,Ne+1:N]./(10.0^-5);
    graph = hcat(ge,gi);

    io = load("D:/qing/0/spikes/SpikeData22000.jld");
    spikes = io["spikes"]; # Nx: matrix of spike times (in ms)
    io = load("D:/qing/0/scores/ScoreData22000.jld");
    scores = io["scores"];
    # scores in array [e_rate,i_rate,frac_filled,e_synch_score,i_synch_score,
    # total_synch_score,e_branch_score,i_branch_score,total_branch_score]
    # however, use new method to calculate rates (only on active units)
    cd("C:/");
    rates = rate_est_active(spikes);
    e_rate = rates[1];
    i_rate = rates[2];
    frac_time_filled = scores[3];
    e_synch = scores[4];
    i_synch = scores[5];
    total_synch = scores[6];
    e_branch = scores[7];
    i_branch = scores[8];
    total_branch = scores[9];

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
    nbins = run_total-maximum(integ_frame)-1;
    Nactive = length(spike_set);
    W_active = graph[active_inds,active_inds];

    Spikes_binned = zeros(Nactive,nbins);

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

    #=
    # determine what kind of lag for recruitment graphs makes the most sense
    maxlag = 50;
    recruit_lag_test(W_active, Spikes_binned, maxlag);
    recruit_lag_test(W_active_e, Spikes_binned[e_edges,:], maxlag);
    recruit_lag_test(W_active_i, Spikes_binned[i_edges,:], maxlag);
    =#

    cycle_recruit_CC, middle_recruit_CC, fanin_recruit_CC, fanout_recruit_CC, all_recruit_CC, isempty_recruit = motifs_through_time(W_active,Spikes_binned,nbins);
    fname = "/Users/maclean lab/Documents/qing/22000_motif_data/temp_fullres.mat";
    file = matopen(fname,"w");
    write(file,"fanout_recruit_CC",fanout_recruit_CC);
    write(file,"fanin_recruit_CC",fanin_recruit_CC);
    write(file,"middle_recruit_CC",middle_recruit_CC);
    write(file,"isempty_recruit",isempty_recruit);
    close(file);
    print("Done!");

    graph = W_active;
    Nsamp = 300;
    fname = "/Users/maclean lab/Documents/qing/22000_motif_data/temp_sub300_1.mat";
    subsample_analysis(Nsamp,graph,Spikes_binned,nbins,fname);
    fname = "/Users/maclean lab/Documents/qing/22000_motif_data/temp_sub300_2.mat";
    subsample_analysis(Nsamp,graph,Spikes_binned,nbins,fname);

    graph = W_active;
    Nsamp = 600;
    fname = "/Users/maclean lab/Documents/qing/22000_motif_data/temp_sub600_1.mat";
    subsample_analysis(Nsamp,graph,Spikes_binned,nbins,fname);
    fname = "/Users/maclean lab/Documents/qing/22000_motif_data/temp_sub600_2.mat";
    subsample_analysis(Nsamp,graph,Spikes_binned,nbins,fname);

    graph = W_active_e;
    subspikes = Spikes_binned[e_edges,:];
    fname = "/Users/maclean lab/Documents/qing/22000_motif_data/temp_fullres_E.mat";
    excit_inhib_analysis(graph,subspikes,nbins,fname);

    graph = W_active_i;
    subspikes = Spikes_binned[i_edges,:];
    fname = "/Users/maclean lab/Documents/qing/22000_motif_data/temp_fullres_I.mat";
    excit_inhib_analysis(graph,subspikes,nbins,fname);
    print("Done!");

end

function batch1net()
    # as of 20180727, the lowest rate network in batch 1 is 6974, with erate 4.36
    cd("D:/");
    io = load("D:/qing/1/network/NetworkData6974.jld");
    graph = io["network"]; # NxN matrix of topological weights
    # normalize according to source
    ge = graph[:,1:Ne]./(10.0^-6);
    gi = graph[:,Ne+1:N]./(10.0^-5);
    graph = hcat(ge,gi);

    io = load("D:/qing/1/spikes/SpikeData6974.jld");
    spikes = io["spikes"]; # Nx: matrix of spike times (in ms)
    io = load("D:/qing/1/scores/ScoreData6974.jld");
    scores = io["scores"];
    # scores in array [e_rate,i_rate,frac_filled,e_synch_score,i_synch_score,
    # total_synch_score,e_branch_score,i_branch_score,total_branch_score]
    # however, use new method to calculate rates (only on active units)
    cd("C:/");
    rates = rate_est_active(spikes);
    e_rate = rates[1];
    i_rate = rates[2];
    frac_time_filled = scores[3];
    e_synch = scores[4];
    i_synch = scores[5];
    total_synch = scores[6];
    e_branch = scores[7];
    i_branch = scores[8];
    total_branch = scores[9];

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

    #=
    # determine what kind of lag for recruitment graphs makes the most sense
    maxlag = 50;
    recruit_lag_test(W_active, Spikes_binned, maxlag);
    recruit_lag_test(W_active_e, Spikes_binned[e_edges,:], maxlag);
    recruit_lag_test(W_active_i, Spikes_binned[i_edges,:], maxlag);
    =#

    cycle_recruit_CC, middle_recruit_CC, fanin_recruit_CC, fanout_recruit_CC, all_recruit_CC, isempty_recruit = motifs_through_time(W_active,Spikes_binned,nbins);
    fname = "/Users/maclean lab/Documents/qing/6974_motif_data/temp_fullres.mat";
    file = matopen(fname,"w");
    write(file,"fanout_recruit_CC",fanout_recruit_CC);
    write(file,"fanin_recruit_CC",fanin_recruit_CC);
    write(file,"middle_recruit_CC",middle_recruit_CC);
    write(file,"isempty_recruit",isempty_recruit);
    close(file);
    print("Done!");


    graph = W_active;
    Nsamp = 300;
    fname = "/Users/maclean lab/Documents/qing/6974_motif_data/temp_sub300_1.mat";
    subsample_analysis(Nsamp,graph,Spikes_binned,nbins,fname);
    fname = "/Users/maclean lab/Documents/qing/6974_motif_data/temp_sub300_2.mat";
    subsample_analysis(Nsamp,graph,Spikes_binned,nbins,fname);

    graph = W_active;
    Nsamp = 600;
    fname = "/Users/maclean lab/Documents/qing/6974_motif_data/temp_sub600_1.mat";
    subsample_analysis(Nsamp,graph,Spikes_binned,nbins,fname);
    fname = "/Users/maclean lab/Documents/qing/6974_motif_data/temp_sub600_2.mat";
    subsample_analysis(Nsamp,graph,Spikes_binned,nbins,fname);

    graph = W_active_e;
    subspikes = Spikes_binned[e_edges,:];
    fname = "/Users/maclean lab/Documents/qing/6974_motif_data/temp_fullres_E.mat";
    excit_inhib_analysis(graph,subspikes,nbins,fname);

    graph = W_active_i;
    subspikes = Spikes_binned[i_edges,:];
    fname = "/Users/maclean lab/Documents/qing/6974_motif_data/temp_fullres_I.mat";
    excit_inhib_analysis(graph,subspikes,nbins,fname);
    print("Done!");

end
