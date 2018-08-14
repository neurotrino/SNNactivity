#=
This script will examine real datasets, recorded by Joe Dechery*, for those
metrics which have been examined in SNNs. Namely, we will be scoring for things
such as synchrony, criticality (via branching), and the geometric mean frequency
of different triangle motifs across time.

*This data has appeared in Dechery & MacLean, 2018.

Partial correlation matrices were computed by Subhodh Kotekal, and spike times
were computed by Carolina Yu using the OASIS inference method (Friedrich et al.,
2017).
=#

# the file with functions that Kyle has compiled
include("/Users/maclean lab/Documents/qing/jason_proj_2.jl");

# module for .mat file i/o
using MAT

#= May 25th 2017, 199 units
file = matopen("/Users/maclean lab/Documents/qing/data/weights/invivo_dataset_20170525.mat");
set1 = read(file, "matrix");
W1 = set1["pcorr"];
close(file);
=#

# March 16th 2017, 292 units
# read file for partial correlation weight matrix, what will be taken as 'ground
# truth'
file = matopen("/Users/maclean lab/Documents/qing/data/20170316/invivo_dataset_20170316.mat");
set = read(file, "matrix");
W = set["pcorr"];
close(file);
vars = matread("/Users/maclean lab/Documents/qing/data/20170316/spikes_for_each_movie.mat");
# instead what you need to do is to 'clean' the data - use the interval btwn each movie,
# and even then not all that - use only 1.5 seconds afterwards.
# read in the stimulus types
stim = matread("/Users/maclean lab/Documents/qing/data/20170316/316_stimtype.mat");
stimtype = stim["stim_type"];
# and what we have is an array for each of the ten movies.
# timesteps with value 0.0 indicate gray screen

function analyze_invivo_set(W,vars,stimtype)
    #=
    This function takes in one session's in vivo recordings.
    W denotes the partial correlation matrix.
    vars denotes the variables contained in the inferred spikes .mat file,
    including each movie presentation trial and the associated spikes.
    =#
    # remove pcorr self-connections
    # negative values in the pcorr matrix indicate anticorrelation
    # for this reason, we threshold our weight matrix at 0
    for ii = 1:size(W)[1]
        for jj = 1:size(W)[2]
            if ii==jj
                W[ii,jj] = 0;
            end
            if W[ii,jj] < 0
                W[ii,jj] = 0;
            end
        end
    end
    Nunits = size(W)[1];
    # combine movies in a dictionary and iterate through them all when calculating recruitment
    # remove fluomats for sake of looping
    delete!(vars, "fluomats");
    RecruitNet = Dict();
    MotifFreqs = Dict();
    NetScores = Dict();
    PopMotif = Dict();
    m_idx = 0;
    # iterate through each movie
    for (k,v) in vars
        m_idx = m_idx + 1;
        # based on the values in stimtype[m_idx], extract v where no stim occurred
        nostim_idx = find(stimtype[m_idx].==0);
        # however, we don't want any residual effects from the stimulus either
        # for this purpose, we will remove the first ~1.5 seconds after stim offset
        # the frame resolution is 30ms
        nostim_idx_mat = [];
        lastii = 1;
        for ii = 1:length(nostim_idx)-1
            # we want to separate v (the s_all's) into segments
            # this is because calculation of the recruitment network rests on subsequent timeframe continuity
            if nostim_idx[ii+1] > (nostim_idx[ii] + 1) # not consecutive, create new row
                push!(nostim_idx_mat,nostim_idx[lastii:ii]');
                #nostim_idx_mat[rowidx,:] = nostim_idx[length(nostim_idx_mat)+1:ii];
                lastii = ii+1;
            end
        end
        for ii = 2:length(nostim_idx_mat)
            # remove the first half of each grey period except for the first
            nostim_idx_mat[ii] = nostim_idx_mat[ii][convert(Int64,ceil.(length(nostim_idx_mat[ii])/2)):length(nostim_idx_mat[ii])];
        end
        # dimensions of block, neuron, timeframe
        # v_nostim = zeros(size(nostim_idx_mat)[1],size(W)[1],);
        v_nostim = [];
        # fill out the blocks based on indices in nostim_idx_mat
        for ii = 1:size(nostim_idx_mat)[1]
            temp = [];
            for jj = 1:size(W)[1]
                push!(temp,v[jj,nostim_idx_mat[ii]]);
            end
            push!(v_nostim,temp);
        end
        Recruitment = [];
        for ii = 1:size(v_nostim)[1]
            recruit = compute_recruitment(W,v_nostim[ii]); # where W is the pcorr matrix
            push!(Recruitment,recruit);
        end
        RecruitNet["$(k)"] = Recruitment;

        #=
        for ii = 1:size(v_nostim)[1]
            # for each block, let's see the motif cycling of the population and
            # that of the top 5% of most active units
            # can call these functions as before (w some slight modifications)
            #invivo_scores(Recruitment[ii],v_nostim[ii]);
            #invivo_motifs_over_time(Recruitment[ii],"$(k)");
            vis_highrate_motifcycling(Recruitment[ii],v_nostim[ii],"$(k)");
        end
        =#

        # compute the population motif CCs at each time point in each block
        Nblock = size(v_nostim)[1];
        PM_arr = []; # as the number of timepts in each block is inconsistent
        # Dict PopMotif contains each movie by name
        # PM_arr is in dimensions of [block, motif (one of four), timept]
        for ii = 1:Nblock
            T = size(v_nostim[ii][1])[2];
            pm_at_time = zeros(4,T);
            pm_at_time[1,:],pm_at_time[2,:],pm_at_time[3,:],pm_at_time[4,:] = population_motifcycling(Recruitment[ii],v_nostim[ii]);
            push!(PM_arr,pm_at_time);
        end
        # save every block of this movie
        PopMotif["$(k)"] = PM_arr;

        #=
        M_arr = zeros(4,size(Recruitment)[3]);
        S_arr = zeros(3,);
        # now score each of these movie's networks
        S_arr[1:3] = invivo_scores(W,v);
        # now look at motifs over time for each movie presentation trial
        M_arr[1,:],M_arr[2,:],M_arr[3,:],M_arr[4,:] = invivo_motifs_over_time(Recruitment,"$(k)");
        # save to appropriate file
        fname = "/Users/maclean lab/Documents/qing/data/20170316/motifs_$(k).jld";
        save(fname,"M_arr",M_arr);
        MotifFreqs["$(k)"] = M_arr;
        NetScores["$(k)"] = S_arr;
        =#

    end
#=
    # loop through every movie and gather all the blocks together and make a scatterplot of fan-in, middleman, cycle
    Vega.scatterplot()
    for (k,v) in PopMotif # for every movie
        for ii = 1:size(v)[1] # for every block
            scatterplot!(v[ii][3,:],v[ii][2,:],v[ii][1,:],marker=".")
        end
    end
    scatterplot!(title = "Motif Cycling",
    xlabel = "fan-in",
    ylabel = "middleman",
    zlabe; = "cycle")
=#
    # can also aggregate across all movie presentations, so we get overall scores for the whole collection of neurons
    #save("/Users/maclean lab/Documents/qing/data/20170316/motifs_allmovies.jld","MotifFreqs",MotifFreqs);
    #save("/Users/maclean lab/Documents/qing/data/20170316/scores_allmovies.jld","NetScores",NetScores);

    file = matopen("/Users/maclean lab/Documents/qing/PopMotif20170316.mat", "w")
    write(file, "PopMotif", PopMotif)
    close(file) # save to matlab, so that I can visualize it there

    return PopMotif;
end

function compute_recruitment(W,s_all)
    #= This function returns the recruitment network, which contains the Wij
    value of each real connection that was functionally active in a sequential
    way.
    Since we are discontinuous through time, we will have BLOCKS. Each block
    will have its own recruitment network.
    =#
    Recruitment = zeros(size(W)[1],size(W)[2],size(s_all)[2]); # dimensions of synaptic weight matrix through time
    Nunits = size(W)[1];
    for ii = 1:Nunits
        eidx = find(W[ii,:].!=0);
        edge_vals = W[ii,eidx];
        Nedges = length(eidx);
        # ^ this'll give us the synaptic connections
        # go through every time step in s_all1 and find when ii spiked at t and any
        # of the units in eidx spiked at t+1.
        # find when neuron ii spiked
        stimes = find(s_all[ii].!=0);
        if length(stimes) > 0
            for tt = 1:length(stimes) # for every time that neuron ii spiked
                for jj = 1:Nedges # for every neuron jj that is synaptically connected to ii
                    if stimes[tt]+1 <= size(s_all[1])[2] # to ensure we're not out of bounds
                        if s_all[eidx[jj]][stimes[tt]+1]==1 # if neuron jj spiked at time t+1
                            Recruitment[ii,jj,stimes[tt]] = W[ii,eidx[jj]]; # add that synaptic weight to the recruitment network matrix
                            # otherwise recruitment Wij remains at zero
                        end
                    end
                end
            end
        end
    end
    return Recruitment;
end

# calculate in-vivo data's scores - synchrony, branching, etc, across time.

#= Can think about it like this: for this network, and maybe every unit within
this network, for every movie it watches and every timepoint within that movie,
what are the score measures, what are the motifs that occur and with what
frequency?
=#

# Now we look at motifs in the recruitment network over time.

#= Notes: function clustering_coef(adj_mat) returns all_motifs_CC.
function clustered_motifs(adj_mat) returns cycle_cc, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC
You may implement it by calling it for each timepoint for each movie and for all neurons.
=#

function invivo_motifs_over_time(Recruitment,movie;showplot=true)
    #= This function calls clustered_motifs(adj_mat) to return the clustering
    coefficients of each motif for each unit in a network.
    This function works for one movie presentation trial. To examine multiple
    trials, recording days and networks, iterate over this function.
    What we hope to capture, then, is the vector of CCs for each unit in the
    network at every time step.
    The adj_mat argument will be the recruitment network at that point in time.
    =#
    T = size(Recruitment)[3];
    N = size(Recruitment)[1];
    cycle_CC = zeros(N,T);
    middleman_CC = zeros(N,T);
    fanin_CC = zeros(N,T);
    fanout_CC = zeros(N,T);
    all_motifs_CC = zeros(N,T);
    for tt = 1:T
        # using the recruitment network at each timepoint, find the CC for each triangle motif
        # at certain timepoints, it is possible that NO spikes occurred
        adj_mat = Recruitment[:,:,tt];
        # the motif CC of each neuron at time tt
        cycle_CC[:,tt], middleman_CC[:,tt], fanin_CC[:,tt], fanout_CC[:,tt], all_motifs_CC[:,tt] = clustered_motifs(adj_mat);
        # if we get a bunch of NaN's, it's because that neuron is not part of recruitment graph at time tt
        # change the NaN's to 0's in jason_proj_2.jl script
        # but this is okay, bc we are ignoring the NaNs when we are calculating the mean population CC at each time point
    end
    # quantify how prominent each motif was at each time step
    mu_cycle = zeros(T,);
    mu_middleman = zeros(T,);
    mu_fanin = zeros(T,);
    mu_fanout = zeros(T,);
    for tt = 1:T
        # calculate mean population CC for each motif at each time point
        mu_cycle[tt] = mean(filter(!isnan,cycle_CC[:,tt]));
        mu_middleman[tt] = mean(filter(!isnan,middleman_CC[:,tt]));
        mu_fanin[tt] = mean(filter(!isnan,fanin_CC[:,tt]));
        mu_fanout[tt] = mean(filter(!isnan,fanout_CC[:,tt]));
    end
    # alternatively, we can be more deterministic and say that a neuron IS part
    # of the motif for which its CC is highest at that time point
    cycle_ct = zeros(T,);
    middleman_ct = zeros(T,);
    fanin_ct = zeros(T,);
    fanout_ct = zeros(T,);
    for tt = 1:T
        for nn = 1:N
            CCvect = [cycle_CC[nn,tt],middleman_CC[nn,tt],fanin_CC[nn,tt],fanout_CC[nn,tt]];
            max = indmax(CCvect);
            # increment the count by one unit for the appropriate motif (greatest CC for that motif)
            if max == 1
                cycle_ct[tt] = cycle_ct[tt]+1;
            elseif max == 2
                middleman_ct[tt] = middleman_ct[tt]+1;
            elseif max == 3
                fanin_ct[tt] = fanin_ct[tt]+1;
            elseif max == 4
                fanout_ct[tt] = fanout_ct[tt]+1;
            end
        end
    end
    # in terms of frequency of occurrence over the whole population of neurons
    cycle_f = cycle_ct./N;
    middleman_f = middleman_ct./N;
    fanin_f = fanin_ct./N;
    fanout_f = fanout_ct./N;
    # now plot
    ts = 1:1:T;
    if(showplot==true)
        Plots.plot(ts,cycle_f,color="blue")
        plot!(ts,middleman_f,color="red")
        plot!(ts,fanin_f,color="orange")
        plot!(ts,fanout_f,color="green")
        plot!(title = "Motif frequency over time for movie $(movie)",
        ylabel = "frequency",
        xlabel = "time",
        label=["cycle","middleman","fan-in","fan-out"])
        Plots.savefig("/Users/maclean lab/Documents/qing/data/20170316/figs/$(movie)")
        # visualize motif cycling
        vis_motif_cycling(mu_cycle,mu_middleman,mu_fanin,mu_fanout);
    end
    #return cycle_f,middleman_f,fanin_f,fanout_f;
    return cycle_CC,middleman_CC,fanin_CC,fanout_CC;
end

function vis_motif_cycling(mu_cycle,mu_middleman,mu_fanin,mu_fanout);
    #=
    Visualize in 3D the interplay between middleman, cycle, and fan-in motifs
    =#
    Plots.plot(mu_fanin,mu_middleman,mu_cycle)
    plot!(title = "Motif cycling",
    xlabel = "fan-in",
    ylabel = "middleman",
    zlabel = "cycle")
    # Plots.savefig("")
end

function vis_highrate_motifcycling(recruit,v,movie)
    #=
    This function takes the neurons with the highest 5% firing rates and plots
    the interplay between middleman, cycle, and fan-in motifs for some block in
    some movie
    =#
    # get the firing rates of each of the units
    # later we'll do instantaneous firing rates
    ct = zeros(size(recruit)[1]);
    for ii = 1:size(recruit)[1]
        ct[ii] = length(find(v[ii].!=0));
    end
    top = find(ct.>=(0.5*maximum(ct)));
    # get the motifs for every time point
    cycle_CC,middleman_CC,fanin_CC,fanout_CC = invivo_motifs_over_time(recruit,movie,showplot=false);
    # visualize
    for ii = 1:length(top)
        Plots.plot(fanin_CC[top[ii],:],middleman_CC[top[ii],:],cycle_CC[top[ii],:])
        plot!(title = "Motif cycling for neuron $(top[ii])",
        xlabel = "fan-in",
        ylabel = "middleman",
        zlabel = "cycle")
    end
end

function population_motifcycling(recruit,v)
    # this function returns the mean population motif CCs for every time point in a block of a movie
    ct = zeros(size(recruit)[1]);
    for ii = 1:size(recruit)[1]
        ct[ii] = length(find(v[ii].!=0));
    end
    # get the motifs for every time point
    cycle_CC,middleman_CC,fanin_CC,fanout_CC = invivo_motifs_over_time(recruit,movie,showplot=false);
    T = size(cycle_CC)[2];
    mu_cycle = zeros(T,);
    mu_middleman = zeros(T,);
    mu_fanin = zeros(T,);
    mu_fanout = zeros(T,);
    for tt = 1:T
        # calculate mean population CC for each motif at each time point
        mu_cycle[tt] = mean(filter(!isnan,cycle_CC[:,tt]));
        mu_middleman[tt] = mean(filter(!isnan,middleman_CC[:,tt]));
        mu_fanin[tt] = mean(filter(!isnan,fanin_CC[:,tt]));
        mu_fanout[tt] = mean(filter(!isnan,fanout_CC[:,tt]));
    end
    return mu_cycle, mu_middleman, mu_fanin, mu_fanout;
end

function highrate_motifcycling(recruit,v)
    # considerations of whether we need to be bound to the same neurons throughout
end
#=
function clustered_motifs(adj_mat)

	if(isempty(adj_mat))
		return NaN,NaN,NaN,NaN,NaN
	end

	dim = size(adj_mat)[1]

	temp = (adj_mat).^(1/3)
	temp_prime = (temp').^(1/3)
	temp_squared = (temp^2).^(1/3)

	d_in = temp_prime*ones(dim)
	d_out = temp*ones(dim)
	d_bi = diag(temp_squared)
    # go back to this - in calculating pcorr we can have bidirectional edges and
    # we want to calculate the degree of that for each unit in our network at
    # this point in time (a vector of length N). does this account for it?
    # (I think it does)
	d_tot = d_in+d_out

	denom = (d_in.*d_out-d_bi)

	cycle_CC  = diag(temp_squared*temp)./denom
	middleman_CC = diag(temp*temp_prime*temp)./denom
	fanin_CC  = diag(temp_prime*temp_squared)./(d_in.*(d_in-1))
	fanout_CC  = diag(temp_squared*temp_prime)./(d_out.*(d_out-1))
	all_motifs_CC = diag(((temp+temp_prime)^3))./(2*(d_tot.*(d_tot-1)-2*d_bi))

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
=#

function invivo_scores(W,spikemat)
    #=
    This function takes in the weight matrix (abs value, no self-connections)
    and the binary spike matrix for a movie presentation trial.
    It returns a vector of scores for the network's activity for that movie.
    score components are:
    e_rate
    i_rate
    frac_filled
    e_synch_score
    i_synch_score
    total_synch_score
    e_branch_score
    i_branch_score
    total_branch_score
    =#
    # convert the binary spikemat into tSpike format
    tSpike = [];
    for ii = 1:size(spikemat)[1]
        timeidx = find(spikemat[ii,:].==1);
        push!(tSpike,timeidx');
    end
    score = zeros(3);
    score[1],score[2] = invivo_rate_est(tSpike);
    score[3],neuron_branches = invivo_branching_param(spikemat,tSpike,W) #branch_e, branch_i, branch_total
    return score;
end

function invivo_rate_est(spike_set)
    num_spikes = 0;
    Nunits = size(spike_set)[1];
    max_vals = zeros(Nunits);
    stim_in = 0;
    for i = 400:499
		post_stim  = (spike_set[i])[find(spike_set[i] .> stim_in)]; # only count spikes after stimulis
		if(isempty(post_stim))
			max_vals[i] = 0
		else
			max_vals[i] = maximum(post_stim);
		end
		num_spikes += length(post_stim);
	end
	max_vals = maximum(max_vals)-stim_in; # ignore stimulis input period
	spikes_per = (num_spikes/Nunits);
	rate_e = 1000*spikes_per/max_vals; #divide by 1000 bc ms conversion
    return rate_e; # for in-vivo we only have excitatory units
end

function invivo_branching_param(Spikes_binned,tSpike,W)
    Nunits = size(W)[1];
    units_bscore = zeros(Nunits);
    for ii = 1:Nunits # for every neuron in the network
        if length(tSpike[ii,]) > 0 # presuming that this neuron fired at all
            # for all edges this neuron ii has in the network
            # (and first we have to find them)
            edges = find(W[ii,:].!=0);
            Nedges = length(edges);
            # find all the time bins t in which neuron ii spiked,
            # examine the time bins t+1 for each of the neurons in edges
            cuebin = find(Spikes_binned[ii,:].==1);
            gobin = cuebin+1;
            Jcount = zeros(Nedges+1); # array to increment the count for how many times those number of neurons were recruited
            # we also have to account for the probability that zero were recruited (hence the +1, and we will index as though 0=1)
            for tt = 1:length(gobin)-1 # for every time that neuron ii spiked:
                # out of the neurons that it's connected to (indexed in edges), how many of those had a spike (value of 1 in Spikes_binned) at time t+1?
                ct = length(find(Spikes_binned[edges,gobin[tt]].==1)); # this will give us the number of edges in this time which spiked (were recruited)
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
    #net_bscore_e = mean(filter(!isnan,units_bscore[1:Ne]));
    #net_bscore_i = mean(filter(!isnan,units_bscore[Ne+1:N]));
    return net_bscore, units_bscore;
end

function invivo_synchrony(spikemat,W)

end
