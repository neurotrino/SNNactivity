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
include("/Users/maclean lab/Documents/qing/equiv_demo.jl");
include("/Users/maclean lab/Documents/qing/static_temp_analysis.jl");

# module for .mat file i/o
using MAT

function main_revis()
    # revision as of 20180726
    # we wish to see whether all_motifs_CC is relatively constant over time
    file = matopen("/Users/maclean lab/Documents/qing/data/20170316/invivo_dataset_20170316.mat");
    set = read(file,"matrix");
    W = set["pcorr"];
    close(file);
    # clean up W
    W = clean_pcorr(W);
    Nunits = size(W)[1];

    vars = matread("/Users/maclean lab/Documents/qing/data/20170316/spikes_for_each_movie.mat");
    stim = matread("/Users/maclean lab/Documents/qing/data/20170316/316_stimtype.mat");
    stimtype = stim["stim_type"];
    fluomats = vars["fluomats"];
    delete!(vars,"fluomats");

    # extract the greyscreen fluorescence traces and spikes
    spikemat = getgreyspikes(vars,fluomats,stimtype);
    fluormat = getgreyfluor(vars,fluomats,stimtype);
    # format is [mat][movie][block][neuron][timestep]

    # calculate the recruitment graph at every timestep for each movie and grey block
    # and from that, calculate the motif values for each time step
    AllCC = Dict();
    MiddleCC = Dict();
    FaninCC = Dict();
    FanoutCC = Dict();
    for ii = 1:length(spikemat) # for every movie
        AllCC["mov$(ii)"] = [];
        MiddleCC["mov$(ii)"] = [];
        FaninCC["mov$(ii)"] = [];
        FanoutCC["mov$(ii)"] = [];
        for jj = 1:length(spikemat[ii]) # for every grey block
            # spikemat[ii][jj] is in format N x time
            # which is essentially Spikes_binned
            # for every time, calculate the motifs of all the N, and their average
            nbins = length(spikemat[ii][jj][1]);
            cycle_recruit_CC, middle_recruit_CC, fanin_recruit_CC, fanout_recruit_CC, all_recruit_CC = motifs_through_time2(W,spikemat[ii][jj],nbins);
            push!(AllCC["mov$(ii)"],all_recruit_CC);
            push!(MiddleCC["mov$(ii)"],middle_recruit_CC);
            push!(FaninCC["mov$(ii)"],fanin_recruit_CC);
            push!(FanoutCC["mov$(ii)"],fanout_recruit_CC);
        end
    end

    # save what we want to matlab
    file = matopen("/Users/maclean lab/Documents/qing/temp_invivo_20170316.mat", "w")
    write(file,"AllCC",AllCC);
    write(file,"FanoutCC",FanoutCC);
    write(file,"FaninCC",FaninCC);
    write(file,"MiddleCC",MiddleCC);
    close(file)

    # attempt the above but with grouping 3 frames together
    # that is, alter spikemat
    triframe_spikemat = [];
    for ii = 1:length(spikemat) # for every movie
        temp = [];
        for jj = 1:length(spikemat[ii]) # for every grey block
            for nn = 1:length(spikemat[ii][jj]) # for every unit
                tsteps = length(spikemat[ii][jj][nn]);
                tristeps = convert(Int64,ceil.(tsteps/3));
                spikes_temp = zeros(Nunits,tristeps);
                # examine every three time steps
                # if any of them contain a 1, put a 1 in that space of spikes_temp
                for tt = 1:tsteps
                    if spikemat[ii][jj][nn][tt] == 1
                        spikes_temp[nn,convert(Int64,ceil.(tt/3))] = 1;
                    end
                end
            end
            push!(temp,spikes_temp);
        end
        push!(triframe_spikemat,temp);
    end

    AllCC = Dict();
    MiddleCC = Dict();
    FaninCC = Dict();
    FanoutCC = Dict();
    for ii = 1:length(triframe_spikemat) # for every movie
        AllCC["mov$(ii)"] = [];
        MiddleCC["mov$(ii)"] = [];
        FaninCC["mov$(ii)"] = [];
        FanoutCC["mov$(ii)"] = [];
        for jj = 1:length(triframe_spikemat[ii]) # for every grey block
            # spikemat[ii][jj] is in format N x time
            # which is essentially Spikes_binned
            # for every time, calculate the motifs of all the N, and their average
            nbins = size(triframe_spikemat[ii][jj])[2];
            cycle_recruit_CC, middle_recruit_CC, fanin_recruit_CC, fanout_recruit_CC, all_recruit_CC = motifs_through_time2(W,triframe_spikemat[ii][jj],nbins);
            push!(AllCC["mov$(ii)"],all_recruit_CC);
            push!(MiddleCC["mov$(ii)"],middle_recruit_CC);
            push!(FaninCC["mov$(ii)"],fanin_recruit_CC);
            push!(FanoutCC["mov$(ii)"],fanout_recruit_CC);
        end
    end

    # save what we want to matlab
    file = matopen("/Users/maclean lab/Documents/qing/temp_invivo_triframe_20170316.mat", "w")
    write(file,"AllCC",AllCC);
    write(file,"FanoutCC",FanoutCC);
    write(file,"FaninCC",FaninCC);
    write(file,"MiddleCC",MiddleCC);
    close(file)

end

function motifs_through_time2(W,Spikes_binned,nbins)
    cycle_over_time = zeros(size(Spikes_binned[1])[2]-1);
	middleman_over_time = zeros(size(Spikes_binned[1])[2]-1);
	fanin_over_time = zeros(size(Spikes_binned[1])[2]-1);
	fanout_over_time = zeros(size(Spikes_binned[1])[2]-1);
	all_motifs_over_time = zeros(size(Spikes_binned[1])[2]-1);
    for tt = 1:convert(Int64,nbins-1)
        recruit = compute_recruitment3(W,Spikes_binned,tt)
        cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = clustered_motifs(recruit);
		cycle_over_time[tt] = mean(cycle_CC);
		middleman_over_time[tt] = mean(middleman_CC);
		fanin_over_time[tt] = mean(fanin_CC);
		fanout_over_time[tt] = mean(fanout_CC);
		all_motifs_over_time[tt] = mean(all_motifs_CC);
    end
    return cycle_over_time,middleman_over_time,fanin_over_time,fanout_over_time,all_motifs_over_time;
end

function main()
    # March 16th 2017, 292 units
    # read file for partial correlation weight matrix, what will be taken as 'ground
    # truth'
    file = matopen("/Users/maclean lab/Documents/qing/data/20170316/invivo_dataset_20170316.mat");
    set = read(file, "matrix");
    W = set["pcorr"];
    close(file);
    # clean up W
    W = clean_pcorr(W);
    Nunits = size(W)[1];
    cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = clustered_motifs(W);
    # find the units in the top decile of middleman, fan-in, and fan-out motifs.
    middle, fanin, fanout, topidx = strongest_motifs(middleman_CC,fanin_CC,fanout_CC,Nunits);
    isoidx = intersect(topidx[1,:],topidx[2,:],topidx[3,:]);
    # contains the unit indices that were in all three motifs' top decile

    vars = matread("/Users/maclean lab/Documents/qing/data/20170316/spikes_for_each_movie.mat");
    stim = matread("/Users/maclean lab/Documents/qing/data/20170316/316_stimtype.mat");
    stimtype = stim["stim_type"];
    fluomats = vars["fluomats"];
    delete!(vars,"fluomats");
    # timesteps in stimtype with value 0.0 indicate gray screen

    # extract the greyscreen fluorescence traces and spikes
    spikemat = getgreyspikes(vars,fluomats,stimtype);
    fluormat = getgreyfluor(vars,fluomats,stimtype);
    # format is [mat][movie][block][neuron][timestep]

    isomotifs_CC = [];
    push!(isomotifs_CC,middleman_CC);
    push!(isomotifs_CC,fanin_CC);
    push!(isomotifs_CC,fanout_CC);

    # save everything we need to work with in matlab
    file = matopen("/Users/maclean lab/Documents/qing/topfluo20170316.mat", "w")
    write(file,"isomotifs_CC",isomotifs_CC);
    close(file)
end

function clean_pcorr(W)
    # this function cleans up the pcorr matrix, i.e. it removes all negative values and self-connections.
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
    return W;
end

function strongest_motifs(middle,fanin,fanout,Nunits)
    CC = zeros(3,Nunits);
    CC[1,:] = middle;
    CC[2,:] = fanin;
    CC[3,:] = fanout;
    tenp = convert(Int64,floor.(.1*Nunits));
    decile = zeros(3,tenp);
    topidx = Array{Int64}(zeros(3,tenp));
    for ii = 1:3
        ordered = sort(CC[ii,:], rev=true);
        topidx[ii,:] = find(CC[ii,:].>=ordered[tenp]);
        decile[ii,:] = CC[ii,topidx[ii,:]];
    end
    return decile[1,:],decile[2,:],decile[3,:],topidx;
end

function getgreyspikes(vars,fluomats,stimtype)
    movies = [];
    for (k,v) in vars
        push!(movies,k)
    end
    spikemat = [];
    for ii = 1:length(fluomats) # ii indexes the movie
        nostim_idx = find(stimtype[ii].==0);
        nostim_idx_mat = [];
        lastjj = 1;
        for jj = 1:length(nostim_idx)-1 # segment into blocks
            # we want to separate s_all into segments
            if nostim_idx[jj+1] > (nostim_idx[jj] + 1) # not consecutive, create new row
                push!(nostim_idx_mat,nostim_idx[lastjj:jj]');
                #nostim_idx_mat[rowidx,:] = nostim_idx[length(nostim_idx_mat)+1:ii];
                lastjj = jj+1;
            end
        end
        for jj = 2:length(nostim_idx_mat)
            # remove the first half of each grey period except for the first
            nostim_idx_mat[jj] = nostim_idx_mat[jj][convert(Int64,ceil.(length(nostim_idx_mat[jj])/2)):length(nostim_idx_mat[jj])];
        end
        # dimensions of block, neuron, timeframe
        # v_nostim = zeros(size(nostim_idx_mat)[1],size(W)[1],);
        v_nostim = [];
        # fill out the blocks based on indices in nostim_idx_mat
        for jj = 1:size(nostim_idx_mat)[1] # jj indexes the blocks
            temp = [];
            for kk = 1:size(W)[1] # kk indexes the neurons
                push!(temp,vars[movies[ii]][kk,nostim_idx_mat[jj]]);
            end
            push!(v_nostim,temp);
        end
        push!(spikemat,v_nostim);
    end
    return spikemat;
    # what we have now is spikemat[movie][block][neuron][time] filled with 0's (no spike) and 1's (spike).
end

function getgreyfluor(vars,fluomats,stimtype)
    fluormat = [];
    for ii = 1:length(fluomats)
        nostim_idx = find(stimtype[ii].==0);
        nostim_idx_mat = [];
        lastjj = 1;
        for jj = 1:length(nostim_idx)-1
            if nostim_idx[jj+1] > (nostim_idx[jj] + 1)
                push!(nostim_idx_mat,nostim_idx[lastjj:jj]');
                lastjj = jj+1;
            end
        end
        for jj = 2:length(nostim_idx_mat)
            nostim_idx_mat[jj] = nostim_idx_mat[jj][convert(Int64,ceil.(length(nostim_idx_mat[jj])/2)):length(nostim_idx_mat[jj])];
        end
        v_nostim = [];
        for jj = 1:size(nostim_idx_mat)[1]
            temp = [];
            for kk = 1:size(W)[1]
                push!(temp,fluomats[ii][kk,nostim_idx_mat[jj]]);
            end
            push!(v_nostim,temp);
        end
        push!(fluormat,v_nostim);
    end
    return fluormat;
end
