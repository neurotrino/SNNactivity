# Motif Verification

using MAT
include("C:/Users/maclean lab/Documents/qing/SNNactivity/jason_proj_2.jl")
include("C:/Users/maclean lab/Documents/qing/SNNactivity/equiv_demo.jl");
include("C:/Users/maclean lab/Documents/qing/SNNactivity/temporal_motifs.jl");
include("C:/Users/maclean lab/Documents/qing/SNNactivity/static_temp_analysis.jl");
#include("C:/Users/maclean lab/Documents/qing/SNNactivity/fix_runbatches.jl");

# as of 20180808

function density_timecourse(group_num,net_num,bin)
    # 1. Decide to use 5ms timebin. Load everything and get set up.

    io = load("D:/qing/$(group_num)/network/NetworkData$(net_num).jld");
    graph = io["network"]; # NxN matrix of topological weights
    # normalize according to source
    ge = graph[:,1:Ne]./(10.0^-6);
    gi = graph[:,Ne+1:N]./(10.0^-5);
    graph = hcat(ge,gi);

    io = load("D:/qing/$(group_num)/spikes/SpikeData$(net_num).jld");
    spikes = io["spikes"]; # Nx: matrix of spike times (in ms)

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

    io = load("D:/qing/$(group_num)/scores/ScoreData$(net_num).jld");
    scores = io["scores"];
    # scores in array [e_rate,i_rate,frac_filled,e_synch_score,i_synch_score,
    # total_synch_score,e_branch_score,i_branch_score,total_branch_score]
    # however, use new method to calculate rates (only on active units)
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

    # 2. Examine density of recruitment graph across time.
    density_recruit = zeros(nbins-1,);
    nactiveunits = zeros(nbins-1,);
    Recruitment_e = [];
    subspikes = Spikes_binned[e_edges,:];
    Nrecruit = zeros(nbins-1,);
    for tt = 1:nbins-1
        recruit = compute_recruitment_delay(W_active_e,subspikes,tt,bin);
        Nrecruit_h = 0;
        Nrecruit_v = 0; # find the number of units that were active in this timestep
        # this number should be the same for the recruitment and functional graphs
        for ii = 1:length(e_edges)
            if !isempty(find(recruit[ii,:].!=0))
                Nrecruit_v = Nrecruit_v + 1;
            end
            if !isempty(find(recruit[:,ii].!=0))
                Nrecruit_h = Nrecruit_h + 1;
            end
        end
        if Nrecruit_h > Nrecruit_v
            Nrecruit[tt] = Nrecruit_h;
        else
            Nrecruit[tt] = Nrecruit_v;
        end
        x = length(find(recruit.>0));
        nactiveunits[tt] = x;
        density_recruit[tt] = x/(Nrecruit[tt]^2-Nrecruit[tt]);
        # normalized by the number of units in the recruitment graph
        push!(Recruitment_e,recruit);
    end

    #Plots.plot(density_recruit)

    # The density of the functional graph should be 5x that of recruitment
    density_functional = zeros(nbins-1,);
    Functional_e = [];
    Nfunctional = zeros(nbins-1,);
    for tt = 1:nbins-1
        functional = compute_functional_delay(W_active_e,subspikes,tt,bin);
        Nfunctional_h = 0;
        Nfunctional_v = 0;
        for ii = 1:Nactive_e
            if !isempty(find(functional[ii,:].!=0))
                Nfunctional_v = Nfunctional_v + 1;
            end
            if !isempty(find(functional[:,ii].!=0))
                Nfunctional_h = Nfunctional_h + 1;
            end
        end
        if Nfunctional_h > Nfunctional_v
            Nfunctional[tt] = Nfunctional_h;
        else
            Nfunctional[tt] = Nfunctional_v;
        end
        x = length(find(functional.>0));
        density_functional[tt] = x/(Nfunctional[tt]^2-Nfunctional[tt]);
        push!(Functional_e,functional);
    end


    # The density of the whole graph should be around .2
    Nactive_e = length(e_edges);
    density_e = length(find(W_active_e.!=0))/(Nactive_e^2-Nactive_e);
    # normalized by the number of units in the active excitatory graph

    # 3. Examine whether global firing fluctuates over time
    active = zeros(nbins-1,);
    globalfr = zeros(nbins-1,);
    for tt = 1:nbins-1
        active[tt] = length(find(subspikes[:,tt].!=0));
        globalfr[tt] = active[tt]/length(e_edges)*1000/bin;
    end

    #Plots.plot(globalfr)
    # Now that we have this, how do the different motifs change over time?

    vars = matread("C:/Users/maclean lab/Documents/qing/$(net_num)_motif_data/temp_fullres_E.mat");
    fanin = vars["fanin_recruit_CC"];
    fanout = vars["fanout_recruit_CC"];
    middle = vars["middle_recruit_CC"];
    #Plots.plot!(fanin*1000)
    #Plots.plot!(fanout*100000)
    #Plots.plot!(middle*10000)

    # 4. Examine whether there are more recurrent connections in recruitment networks than in the synaptic
    # (relative of course to the total number of units participating in any given timepoint)
    recur_syn = 0;
    recur = 0; # check equivalence of methods
    for ii = 1:length(e_edges)
        for jj = 1:length(e_edges)
            if W_active_e[ii,jj]!=0 && W_active_e[jj,ii]!=0 && ii!=jj
                recur_syn = recur_syn + 1;
            end
        end
        idx = find(W_active_e[ii,:].!=0);
        recur = recur + length(find(W_active_e[idx,ii].!=0));
    end
    recur_syn_p = recur_syn/((Nactive_e^2-Nactive_e)/2);
    # normalized by the number of possible recurrent connections in synaptic graph
    recur_syn_p2 = recur/length(find(W_active_e.!=0));
    # normalized by the number of actual connections in synaptic graph

    recur_recruit = zeros(length(Recruitment_e),);
    for tt = 1:length(Recruitment_e)
        for ii = 1:length(e_edges)
            idx = find(Recruitment_e[tt][ii,:].!=0);
            recur_recruit[tt] = recur_recruit[tt] + length(find(Recruitment_e[tt][idx,ii].!=0));
        end
    end
    recur_recruit_p = recur_recruit./((Nrecruit.^2-Nrecruit)/2);
    # normalized by the number of possible recurrent connections in recruitment graph (at every timept)
    Plots.histogram(recur_recruit_p)

    Plots.plot(recur_recruit_p);
    Plots.plot!(density_recruit);

    recur_recruit_p2 = recur_recruit./nactiveunits;
    # normalized by the number of actual connections in recruitment graph (at every timept)

    # how often are all connections in synaptic graph selected for recruitment graph?


#=
    print("Synaptic recurrence");
    print(recur_syn_p);
    print("Recruitment recurrence max and mean")
    print(maximum(recur_recruit_p));
    print(mean(recur_recruit_p));
    print("Mean inst global fr")
    print(mean(globalfr));
    print("Mean recruitment density")
    print(mean(density_recruit));


    # 5. Might as well calculate motifs in the new manner
    cycle_over_time,middleman_over_time,fanin_over_time,fanout_over_time,all_motifs_over_time,isempty_recruit = motifs_through_time(W_active_e,subspikes,nbins,bin);
    fname = "/Users/maclean lab/Documents/qing/$(net_num)/motif_data/new_subgraph_e.mat";
    file = matopen(fname,"w");
    write(file,"fanout_recruit_CC",fanout_over_time);
    write(file,"fanin_recruit_CC",fanin_over_time);
    write(file,"middle_recruit_CC",middleman_over_time);
    write(file,"cycle_recruit_CC",cycle_over_time);
    write(file,"isempty_recruit",isempty_recruit);
    close(file);
    print("Done!");
    =#

end


# function to calculate the density of various networks and plot a distribution
function density_dist(group_num,net_num,bin)
    # look at the 1000 runs we have in Run 1 of 1,6974
    io = load("D:/qing/$(group_num)/network/NetworkData$(net_num).jld");
    graph = io["network"]; # NxN matrix of topological weights
    # normalize according to source
    ge = graph[:,1:Ne]./(10.0^-6);
    gi = graph[:,Ne+1:N]./(10.0^-5);
    graph = hcat(ge,gi);

    io = load("C:/Users/maclean lab/Documents/qing/$(net_num)/fix_all/Run1_Scores.jld");
    scores = io["batchscores"];
    ofinterest = intersect(find(scores[:,1].<6.5),find(scores[:,1].>3),find(scores[:,3].>.2),find(scores[:,3].<.6));

    io = load("C:/Users/maclean lab/Documents/qing/$(net_num)/fix_all/Run1_Spikes.jld");
    Spikes = io["batchspikes"];

    Density_syn = zeros(length(ofinterest),);
    Density_rec = zeros(length(ofinterest),);
    Density_rec_recur = zeros(length(ofinterest),);
    Density_syn_recur = zeros(length(ofinterest),);

    for nn = 1:length(ofinterest) # go through every simulation that satisfied our scores
        # 1. calculate the density of the synaptic and recruitment graphs (mean recruitment across time)
        # 2. calculate the density of the recurrent connections in both graphs

        spikes = Spikes[ofinterest[nn]];
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

        # 2. Examine density of recruitment graph across time.
        density_recruit = zeros(nbins-1,);
        nactiveunits = zeros(nbins-1,);
        Recruitment_e = [];
        subspikes = Spikes_binned[e_edges,:];
        Nrecruit = zeros(nbins-1,);
        for tt = 1:nbins-1
            recruit = compute_recruitment_delay(W_active_e,subspikes,tt,bin);
            Nrecruit_h = 0;
            Nrecruit_v = 0; # find the number of units that were active in this timestep
            # this number should be the same for the recruitment and functional graphs
            for ii = 1:length(e_edges)
                if !isempty(find(recruit[ii,:].!=0))
                    Nrecruit_v = Nrecruit_v + 1;
                end
                if !isempty(find(recruit[:,ii].!=0))
                    Nrecruit_h = Nrecruit_h + 1;
                end
            end
            if Nrecruit_h > Nrecruit_v
                Nrecruit[tt] = Nrecruit_h;
            else
                Nrecruit[tt] = Nrecruit_v;
            end
            x = length(find(recruit.>0));
            nactiveunits[tt] = x;
            density_recruit[tt] = x/(Nrecruit[tt]^2-Nrecruit[tt]);
            # normalized by the number of units in the recruitment graph
            push!(Recruitment_e,recruit);
        end
        Density_rec[nn] = mean(density_recruit);

        # The density of the whole graph should be around .2
        Nactive_e = length(e_edges);
        density_e = length(find(W_active_e.!=0))/(Nactive_e^2-Nactive_e);
        # normalized by the number of units in the active excitatory graph
        Density_syn[nn] = density_e;

        # 4. Examine whether there are more recurrent connections in recruitment networks than in the synaptic
        # (relative of course to the total number of units participating in any given timepoint)
        recur_syn = 0;
        recur = 0; # check equivalence of methods
        for ii = 1:length(e_edges)
            for jj = 1:length(e_edges)
                if W_active_e[ii,jj]!=0 && W_active_e[jj,ii]!=0 && ii!=jj
                    recur_syn = recur_syn + 1;
                end
            end
            idx = find(W_active_e[ii,:].!=0);
            recur = recur + length(find(W_active_e[idx,ii].!=0));
        end
        recur_syn_p = recur_syn/((Nactive_e^2-Nactive_e)/2);
        # normalized by the number of possible recurrent connections in synaptic graph
        Density_syn_recur[nn] = recur_syn_p;

        recur_recruit = zeros(length(Recruitment_e),);
        for tt = 1:length(Recruitment_e)
            for ii = 1:length(e_edges)
                idx = find(Recruitment_e[tt][ii,:].!=0);
                recur_recruit[tt] = recur_recruit[tt] + length(find(Recruitment_e[tt][idx,ii].!=0));
            end
        end
        recur_recruit_p = recur_recruit./((Nrecruit.^2-Nrecruit)/2);
        # normalized by the number of possible recurrent connections in recruitment graph (at every timept)
        Density_rec_recur[nn] = mean(recur_recruit_p);
    end

    #= plot
    Plots.histogram(Density_rec_recur);
    Plots.histogram(Density_syn_recur);

    Plots.histogram(Density_rec);
    Plots.histogram(Density_syn);=#

    # and/or save to matlab
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/Run1_densities_trunc.mat";
    file = matopen(fname,"w");
    write(file,"Density_rec_recur",Density_rec_recur);
    write(file,"Density_syn_recur",Density_syn_recur);
    write(file,"Density_rec",Density_rec);
    write(file,"Density_syn",Density_syn);
    close(file);
    print("Done!");

end

function rate_motif_correspondence(group_num,net_num,bin)
    # This function examines
    # 1. the global rate and the global motif values over time
    # 2. the instantaneous firing rate of individual units and their motif values over time

    io = load("D:/qing/$(group_num)/network/NetworkData$(net_num).jld");
    graph = io["network"]; # NxN matrix of topological weights
    # normalize according to source
    ge = graph[:,1:Ne]./(10.0^-6);
    gi = graph[:,Ne+1:N]./(10.0^-5);
    graph = hcat(ge,gi);

    io = load("D:/qing/$(group_num)/spikes/SpikeData$(net_num).jld");
    spikes = io["spikes"]; # Nx: matrix of spike times (in ms)

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

    # remember we're still only looking at excitatory units
    subspikes = Spikes_binned[e_edges,:];

    GlobalRate = zeros(nbins,);
    indivbins = convert(Int64,(run_total-maximum(integ_frame))/10-1);
    IndivRate = zeros(length(e_edges),indivbins);
    for ii = 1:nbins
        GlobalRate[ii] = length(find(subspikes[:,ii].==1))/bin;
        # for each excitatory unit, find how many spikes it emitted in every 10ms bin and we get the instantaneous (more of less) firing rate
        # we will later calculate motifs for every 10ms for every neuron.
        end
    end

    for tt = 1:indivbins
        for jj = 1:length(e_edges)
            if !isempty(subspikes[jj,:])
                idx = floor.(subspikes[jj,])
                IndivRate[jj,tt] = ;
            end
        end
    end

    # calculate motif values to match with the global rate
    cycle_recruit_CC, middle_recruit_CC, fanin_recruit_CC, fanout_recruit_CC, all_recruit_CC = motifs_through_time(W_active_e,subspikes,nbins,bin);
    cycle_recruit_CC_nan, middle_recruit_CC_nan, fanin_recruit_CC_nan, fanout_recruit_CC_nan, all_recruit_CC_nan = motifs_through_time_nanfilt(W_active_e,subspikes,nbins,bin);

    # calculate all four motif values to match with these instantaneous rate bins
    indiv_cycle_CC = zeros(length(e_edges),indivbins);
    indiv_middle_CC = zeros(length(e_edges),indivbins);
    indiv_fanin_CC = zeros(length(e_edges),indivbins);
    indiv_fanout_CC = zeros(length(e_edges),indivbins);
    indiv_all_CC = zeros(length(e_edges),indivbins);

    # save everything
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/rate_motif_corr.mat";
    file = matopen(fname,"w");
    write(file,"cycle_CC",cycle_recruit_CC);
    write(file,"middle_CC",middle_recruit_CC);
    write(file,"fanin_CC",fanin_recruit_CC);
    write(file,"fanout_CC",fanout_recruit_CC);
    write(file,"all_CC",all_recruit_CC);
    write(file,"GlobalRate",GlobalRate);
    # save the motif values for if we are nan'ing (rather than zero'ing) the undefined denom units
    write(file,"cycle_CC_nan",cycle_recruit_CC_nan);
    write(file,"middle_CC_nan",middle_recruit_CC_nan);
    write(file,"fanin_CC_nan",fanin_recruit_CC_nan);
    write(file,"fanout_CC_nan",fanout_recruit_CC_nan);
    write(file,"all_CC_nan",all_recruit_CC_nan);
    #=
    write(file,"IndivRate",IndivRate);
    write(file,"indiv_cycle_CC",indiv_cycle_CC);
    write(file,"indiv_middle_CC",indiv_middle_CC);
    write(file,"indiv_fanin_CC",indiv_fanin_CC);
    write(file,"indiv_fanout_CC",indiv_fanout_CC);
    write(file,"indiv_all_CC",indiv_all_CC);
    =#
    close(file);

end

function rate_comparison(group_num,net_num,bin)
    # Compare with rates from networks before we altered the poisson input
    # Load in the old networks, let's try the fix_all from 6974
    io = load("D:/qing/$(group_num)/network/NetworkData$(net_num).jld");
    graph = io["network"]; # NxN matrix of topological weights

    io = load("C:/Users/maclean lab/Documents/qing/$(net_num)/fix_all/Run1_Spikes.jld");
    Spikes = io["batchspikes"];

    io = load("C:/Users/maclean lab/Documents/qing/$(net_num)/fix_all/Run1_Scores.jld");
    Scores = io["batchscores"];

    # random sample to match the size of our Run2 sample.
    ridx = rand(1:length(Spikes),100,1);
    Scores = Scores[ridx];
    Spikes = Spikes[ridx];

    ofinterest = find(Scores[:,3].>.95);

    e_rate_old = zeros(length(ofinterest),);
    for ii = 1:length(ofinterest)
        rates = rate_est_active(Spikes[ofinterest[ii]]);
        e_rate_old[ii] = rates[1];
    end

    # Use that input info and run new fix-all's, saving the values
    io = load("D:/qing/$(group_num)/poisson_in_data/PoissonData$(net_num).jld");
    generated_stim = io["poisson_stim"]; # 500(units)x300 Poisson input stim
    generated_projected = io["input_neurons"]; # indices of input units
    param = [.31,.22,.3,2];
    for ii = 1:size(Spikes)[1]
        # generate as many new runs as to be equivalent to the number of old runs.
        # filter them after.
        batchscores = zeros(size(Spikes)[1],9);
        batchspikes = [];
        for jj = 1:size(Spikes)[1]
            batchscores[jj,:],spikes = fix_all(param,graph,generated_stim,generated_projected);
            push!(batchspikes,spikes);
        end
    end
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_all/Run2_Spikes.jld";
    save(fname,"batchspikes",batchspikes);
    fname = "C:/Users/maclean lab/Documents/qing/$(net_num)/fix_all/Run2_Scores.jld";
    save(fname,"batchscores",batchscores);

    io = load("C:/Users/maclean lab/Documents/qing/$(net_num)/fix_all/Run2_Scores.jld");
    Scores = io["batchscores"];
    ofinterest = find(Scores[:,3].>.95);

    e_rate_new = zeros(length(ofinterest),);
    for ii = 1:length(ofinterest)
        rates = rate_est_active(Spikes[ofinterest[ii]]);
        e_rate_new[ii] = rates[1];
    end

    # save the e_rates of the two kinds of networks
    fname = "C:/Users/maclean lab/Documents/qing/oldnewrates.jld";
    file = matopen(fname,"w");
    write(file,"e_rate_new",e_rate_new);
    write(file,"e_rate_old",e_rate_old);
    close(file);

end
