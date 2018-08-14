#include("/Users/maclean lab/Downloads/jason_proj_2.jl")


function in_deg_stats(run_number,run_number_2)
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    net_path = string(path_string,"/network/")
    net = readdir(net_path)
    cd(net_path)
    net = load(net[2])
    net = net["network"]
    in_degs = zeros(N)
    #figure()
    #PyPlot.plot(in_degs)
    #plt[:hist](in_degs[1:Ne],nbins,color="blue",alpha=0.6,normed=true)
    #plt[:hist](in_degs[Ne+1:N],nbins,color="red",alpha=0.6,normed=true)

    n_samples = 10000

    cyc_e_mean = 0.0
    cyc_i_mean = 0.0
    cyc_e_std = 0.0
    cyc_i_std = 0.0

    mid_e_mean = 0.0
    mid_i_mean = 0.0
    mid_e_std = 0.0
    mid_i_std = 0.0

    fanin_e_mean = 0.0
    fanin_i_mean = 0.0
    fanin_e_std = 0.0
    fanin_i_std = 0.0

    fanout_e_mean = 0.0
    fanout_i_mean = 0.0
    fanout_e_std = 0.0
    fanout_i_std = 0.0

    cc_e_mean = 0.0
    cc_i_mean = 0.0
    cc_e_std = 0.0
    cc_i_std = 0.0

    tic()
    #Series(mean(), Variance())
    for i = 1:n_samples
        print(i,"\t",cyc_e_mean,"\n")
        projected_bootstrap = sort(randperm(N)[1:num_projected])
        cyc, mid, fanin, fanout, cc = clustered_motifs(net[projected_bootstrap,projected_bootstrap])

        cyc_e_mean += cyc_e_mean + (mean(cyc[projected_bootstrap .<= Ne])-cyc_e_mean)/i
        cyc_i_mean += cyc_i_mean + (mean(cyc[projected_bootstrap .> Ne])-cyc_i_mean)/i

        mid_e_mean += mid_e_mean + (mean(mid[projected_bootstrap .<= Ne])-mid_e_mean)/i
        mid_i_mean += mid_i_mean + (mean(mid[projected_bootstrap .> Ne])-mid_i_mean)/i

        fanin_e_mean += fanin_e_mean + (mean(fanin[projected_bootstrap .<= Ne])-fanin_e_mean)/i
        fanin_i_mean += fanin_i_mean + (mean(fanin[projected_bootstrap .> Ne])-fanin_i_mean)/i

        fanout_e_mean += fanout_e_mean + (mean(fanout[projected_bootstrap .<= Ne])-fanout_e_mean)/i
        fanout_i_mean += fanout_i_mean + (mean(fanout[projected_bootstrap .> Ne])-fanout_i_mean)/i

        cc_e_mean += cc_e_mean + (mean(cc[projected_bootstrap .<= Ne])-cc_e_mean)/i
        cc_i_mean += cc_i_mean + (mean(cc[projected_bootstrap .> Ne])-cc_i_mean)/i

    end
    toc()

    std_e = std(in_degs[1:Ne])
    std_i = std(in_degs[Ne+1:N])
    mean_e = mean(in_degs[1:Ne])
    mean_i = mean(in_degs[Ne+1:N])
    cycle_CC, middleman_CC, fanin_CC, fanout_CC, all_motifs_CC = clustered_motifs(net)

    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number_2)
    analysis_path = string(path_string,"/analysis/")
    projected_path = string(path_string,"/projected/")
    score_path = string(path_string,"/scores/")
    list = readdir(analysis_path)
    list2 = readdir(projected_path)
    list3 = readdir(score_path)

    cd(analysis_path)

    z_score_in_deg_e = []
    z_score_in_deg_i = []

    z_score_in_deg_e_total = []
    z_score_in_deg_i_total = []

    z_score_cycle_e = []
    z_score_cycle_i = []
    z_score_mid_e = []
    z_score_mid_i = []
    z_score_fanin_e = []
    z_score_fanin_i = []
    z_score_fanout_e = []
    z_score_fanout_i = []
    z_score_cc_e = []
    z_score_cc_i = []

    ignite_scores = []

    for i = 1:length(list)
        print(i,"\n")
        cd(analysis_path)
        data = load(list[i])
        cd(projected_path)
        projected = load(list2[i])["projected"]
        e_projected = sort(projected[projected.<=Ne])
        i_projected = sort(projected[projected.>Ne])

        motif = data["cycle_CC"]
        push!(z_score_cycle_e,(mean(motif[1:length(e_projected)])-mean_e_cycle)/std_e_cycle)
        push!(z_score_cycle_i,(mean(motif[1:length(i_projected)])-mean_i_cycle)/std_i_cycle)

        motif = data["middleman_CC"]
        push!(z_score_mid_e,(mean(motif[1:length(e_projected)])-mean_e_mid)/std_e_mid)
        push!(z_score_mid_i,(mean(motif[1:length(i_projected)])-mean_i_mid)/std_i_mid)

        motif = data["fanin_CC"]
        push!(z_score_fanin_e,(mean(motif[1:length(e_projected)])-mean_e_fanin)/std_e_fanin)
        push!(z_score_fanin_i,(mean(motif[1:length(i_projected)])-mean_i_fanin)/std_i_fanin)

        motif = data["fanout_CC"]
        push!(z_score_fanout_e,(mean(motif[1:length(e_projected)])-mean_e_fanout)/std_e_fanout)
        push!(z_score_fanout_i,(mean(motif[1:length(i_projected)])-mean_i_fanout)/std_i_fanout)

        motif = data["all_motifs_CC"]
        push!(z_score_cc_e,(mean(motif[1:length(e_projected)])-mean_e_all)/std_e_all)
        push!(z_score_cc_i,(mean(motif[1:length(i_projected)])-mean_i_all)/std_i_all)

        e_net = net[e_projected,e_projected]
        in_deg_e = zeros(size(e_net)[1])
        for j = 1:size(e_net)[1]
            in_deg_e[j] = length(find(e_net[:,j]))
        end
        in_deg_e = in_deg_e/size(e_net)[1]
        push!(z_score_in_deg_e,(mean(in_deg_e)-mean_e)/std_e)

        i_net = net[i_projected,i_projected]
        in_deg_i = zeros(size(i_net)[1])
        for j = 1:size(i_net)[1]
            in_deg_i[j] = length(find(i_net[:,j]))
        end
        in_deg_i = in_deg_i/size(i_net)[1]
        push!(z_score_in_deg_i,(mean(in_deg_i)-mean_i)/std_i)

        i_net = net[:,i_projected]
        in_deg_i = zeros(length(i_projected))
        for j = 1:length(i_projected)
            in_deg_i[j] = length(find(i_net[:,j]))
        end
        in_deg_i = in_deg_i/N
        push!(z_score_in_deg_i_total,(mean(in_deg_i)-mean_i)/std_i)

        e_net = net[:,e_projected]
        in_deg_e = zeros(length(e_projected))
        for j = 1:length(e_projected)
            in_deg_e[j] = length(find(e_net[:,j]))
        end
        in_deg_e = in_deg_e/N
        push!(z_score_in_deg_e_total,(mean(in_deg_e)-mean_e)/std_e)

        cd(score_path)
        scores = load(list3[i])["score_vals"]
        push!(ignite_scores,scores[3])


    end

    figure()
    PyPlot.scatter(ignite_scores,z_score_in_deg_e,s=.5)
    title("z scores for out degree excitatory neurons (projected sub pop only) vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_in_deg_i,s=.5)
    title("z scores for out degree inhibitory neurons (projected sub pop only) vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_in_deg_e_total,s=.5)
    title("z scores for out degree excitatory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_in_deg_i_total,s=.5)
    title("z scores for out degree inhibitory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_cycle_e,s=.5)
    title("z scores for cycle_CC excitatory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_cycle_i,s=.5)
    title("z scores for cycle_CC inhibitory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_cc_e,s=.5)
    title("z scores for CC excitatory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_cc_i,s=.5)
    title("z scores for CC inhibitory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_mid_e,s=.5)
    title("z scores for middleman_CC excitatory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_mid_e,s=.5)
    title("z scores for middleman_CC inhibitory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_fanin_e,s=.5)
    title("z scores for fanin_CC excitatory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_fanin_i,s=.5)
    title("z scores for fanin_CC inhibitory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_fanout_e,s=.5)
    title("z scores for fanout_CC excitatory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    figure()
    PyPlot.scatter(ignite_scores,z_score_fanout_i,s=.5)
    title("z scores for fanout_CC inhibitory neurons vs ignite scores")
    xlabel("ignite scores")
    ylabel("z score")

    ###### histograms ########
    nbins = 30
    figure()
    plt[:hist](z_score_in_deg_e[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_in_deg_e[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores out degree excitatory proj sub pop only")

    figure()
    plt[:hist](z_score_in_deg_i[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_in_deg_i[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores out degree inhibitory proj sub pop only")

    figure()
    plt[:hist](z_score_in_deg_e_total[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_in_deg_e_total[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores out degree excitatory")

    figure()
    plt[:hist](z_score_in_deg_i_total[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_in_deg_i_total[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores out degree inhibitory")

    figure()
    plt[:hist](z_score_cc_e[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_cc_e[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores CC excitatory")

    figure()
    plt[:hist](z_score_cc_i[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_cc_i[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores CC inhibitory")

    figure()
    plt[:hist](z_score_mid_e[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_mid_e[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores middleman_CC excitatory")

    figure()
    plt[:hist](z_score_mid_i[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_mid_i[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores middleman_CC inhibitory")

    figure()
    plt[:hist](z_score_fanin_e[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_fanin_e[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores fanin_CC excitatory")

    figure()
    plt[:hist](z_score_fanin_i[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_fanin_i[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores fanin_CC inhibitory")

    figure()
    plt[:hist](z_score_fanout_e[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_fanout_e[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores fanout_CC excitatory")

    figure()
    plt[:hist](z_score_fanout_i[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_fanout_i[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores fanout_CC inhibitory")

    figure()
    plt[:hist](z_score_cycle_e[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_cycle_e[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores cycle_CC excitatory")

    figure()
    plt[:hist](z_score_cycle_i[ignite_scores .< .95],nbins, color = "red", alpha = .6, normed = true)
    plt[:hist](z_score_cycle_i[ignite_scores .> .95],nbins, color = "blue", alpha = .6, normed = true)
    title("Histogram z scores cycle_CC inhibitory")


    samp1 = convert(Array{Float64,1}, z_score_cc_e[ignite_scores .< .95])
    samp2 = convert(Array{Float64,1}, z_score_cc_e[ignite_scores .> .95])
    p_vals = ApproximateTwoSampleKSTest(samp1, samp2)
    print(p_vals)

    #=
    figure()
    plt[:hist](cycle_CC[Ne+1:N],nbins,color="red",alpha=0.6,normed=true)
    plt[:hist](cycle_CC[1:Ne],nbins,color="blue",alpha=0.6,normed=true)
    figure()
    plt[:hist](middleman_CC[Ne+1:N],nbins,color="red",alpha=0.6,normed=true)
    plt[:hist](cycle_CC[1:Ne],nbins,color="blue",alpha=0.6,normed=true)
    figure()
    plt[:hist](fanin_CC[Ne+1:N],nbins,color="red",alpha=0.6,normed=true)
    plt[:hist](cycle_CC[1:Ne],nbins,color="blue",alpha=0.6,normed=true)
    figure()
    plt[:hist](fanout_CC[Ne+1:N],nbins,color="red",alpha=0.6,normed=true)
    plt[:hist](cycle_CC[1:Ne],nbins,color="blue",alpha=0.6,normed=true)
    figure()
    plt[:hist](all_motifs_CC[Ne+1:N],nbins,color="red",alpha=0.6,normed=true)
    plt[:hist](cycle_CC[1:Ne],nbins,color="blue",alpha=0.6,normed=true)
    =#

end

function clean_dir(path,run_number)
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
        if(data[1]>5)
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


function find_lowest_rate_network(run_number,path) # revision as of 20180801
    low_score = Inf
    path_string = string(path,run_number)
    score_path = string(path_string,"/scores/")
    list = readdir(score_path)
    spikes_path = string(path_string,"/spikes/")
    list2 = readdir(spikes_path)
    print("done")
    cd(score_path)
    ind = 0
    count = 0;
    for i = 1:length(list)
        data = load(list[i])["scores"]
        if(data[3]>.95) # criteria.
            if(data[1]<4.5)
                count = count+1;
                low_score = data[1]
                ind = i
                #print("\n",data,"\n")
            end
        end
    end
    print("count: ",count,"\n");
    #rasterplot_from_file(string(spikes_path,list2[ind]))
    print(list2[ind]);
    #spiking_vis(list2[ind],run_number)
end

function find_highest_rate_network(run_number)
    low_score = -Inf
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    score_path = string(path_string,"/scores/")
    list = readdir(score_path)
    spikes_path = string(path_string,"/spikes/")
    list2 = readdir(spikes_path)
    print(list2)
    cd(score_path)
    ind = 0
    for i = 1:length(list)
        data = load(list[i])["score_vals"]
        if(data[3]>.95)
            if(data[1]>low_score)
                low_score = data[1]
                ind = i
                print("\n",data[1],"\n")
            end
        end
    end
    #rasterplot_from_file(list2[ind])
    list2[ind]
    spiking_vis(string(spikes_path,list2[ind]),run_number)
end

function find_network_in_range(run_number)
    low = .45
    high = .55
    low_score = Inf
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    score_path = string(path_string,"/scores/")
    list = readdir(score_path)
    spikes_path = string(path_string,"/spikes/")
    list2 = readdir(spikes_path)
    #print(list2)
    cd(score_path)
    ind = 0
    for i = 1:length(list)
        #print("\n",i/length(list),"\n")
        data = load(list[i])["score_vals"]
        if((data[3]<high) & (data[3]>low))
            spiking_vis(list2[i],run_number)
            print("\n",data[1],"\n")
        end
    end
    #rasterplot_from_file(list2[ind])
    #list2[ind]
    #spiking_vis(list2[ind],run_number)
end

function spiking_vis(file,run_number)
    path_string = string("/Users/maclean lab/Documents/qing/",run_number)
    score_path = string(path_string,"/scores/")
    spikes_path = string(path_string,"/spikes/")
    cd(spikes_path)
    spikes = load(file)["spikes"]
    ISI_hist(spikes)
    rasterplot_from_file(file)
end

function score_dist(run_number)
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    score_path = string(path_string,"/scores/")
    list = readdir(score_path)

    cd(score_path)
    time_scores = []
    rate_scores_finished_i = []
    rate_scores_finished_e = []
    rate_scores_notfinished_i = []
    rate_scores_notfinished_e = []
    for i = 1:length(list)
        #print(i/length(list))
        #print("\n")
        data = load(list[i])["score_vals"]
        #rate_scores[i] = data[1]
        #time_scores[i] = data[2]
        if(data[3]<.35)
            if(data[1]<1000)
                push!(rate_scores_notfinished_e,data[1])
            end
            if(data[2]<1000)
                push!(rate_scores_notfinished_i,data[2])
            end
            push!(time_scores,data[3])
        else
            if(data[1]<1000)
                push!(rate_scores_finished_e,data[1])
            end
            if(data[2]<1000)
                push!(rate_scores_finished_i,data[2])
            end
        end
        #if((rate_scores[i]>1) & (rate_scores[i]<8))
        #    rasterplot_from_file(list[i])
        #end
        #if((time_scores[i]>.14) & (time_scores[i]<.3))
        #    rasterplot_from_file(list[i])
        #end
    end
    nbins = 50
    nbins2 = 20
    plt[:hist](rate_scores_finished_e,nbins,color="blue",alpha=0.6,normed=true)
    plt[:hist](rate_scores_notfinished_e,nbins,color="red",alpha=0.6,normed=true)
    figure()
    plt[:hist](rate_scores_finished_i,nbins,color="blue",alpha=0.6,normed=true)
    plt[:hist](rate_scores_notfinished_i,nbins,color="red",alpha=0.6,normed=true)
    figure()
    plt[:hist](time_scores,bins=100,normed=true)
    return 0
end

function rasterplot_from_file(file)
    Spike = load(file)["spikes"]
    t=[]
    neuron=[]
    for n=1:length(Spike)
        append!(t,Spike[n])
        append!(neuron,n*ones(length(Spike[n])))
    end
    t=convert(Array{Float64},t)
    neuron=convert(Array{Float64},neuron)
    idx=find(t.>0)
    t=t[idx]
    neuron=neuron[idx]
    figure()
    PyPlot.scatter(t,neuron,s=.2)
    return 0
end

function psth_from_file(file,bin_width,time_total)
    spikes = load(file)["spikes"]
    rate_vec_e,rate_vec_i = psth(spikes,bin_width,time_total)
    return rate_vec_e,rate_vec_i
end



function cluster_set(run_number)
    bins = 2
    time_total = 1000
    time_start = 0:bins:(time_total-bins)
    time_ends  = time_start + bins
    time_vec = (time_start+time_ends)/2
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    score_path = string(path_string,"/scores/")
    analysis_path = string(path_string,"/analysis/")
    list = readdir(score_path)
    list2 = readdir(analysis_path)
    cd(score_path)

    cluster_vec_unfinished = zeros(length(time_vec))
    cluster_vec_finished   = zeros(length(time_vec))
    cluster_vec_unfinished_mean = zeros(length(time_vec))
    cluster_vec_finished_mean   = zeros(length(time_vec))
    first_quartile_finished_clustering = zeros(length(time_vec))
    third_quartile_finished_clustering = zeros(length(time_vec))
    first_quartile_unfinished_clustering = zeros(length(time_vec))
    third_quartile_unfinished_clustering = zeros(length(time_vec))

    NaN_count_finished = zeros(length(time_vec))
    NaN_count_unfinished = zeros(length(time_vec))
    for i = 1:length(list)
        print(i/length(list))
        print("\n")
        data = load(list[i])["score_vals"]
        if((data[3]<.95) && (data[3]>0)) # this structure only works for early trial sets
            psth_data = load(string(analysis_path,list2[i]))
            cluster= psth_data["cluster_vals"]
            cluster_vec_unfinished= [cluster_vec_finished cluster]
        elseif(data[3]>.95)
            psth_data = load(string(analysis_path,list2[i]))
            cluster= psth_data["cluster_vals"]
            cluster_vec_finished  = [cluster_vec_finished cluster]
        end
    end

    for i = 1:length(time_start)

        temp_vec_clustering = cluster_vec_finished[i,:]
        temp_vec_clustering = temp_vec_clustering[.~isnan.(temp_vec_clustering)] # remove nan
        temp_vec_clustering = temp_vec_clustering[.~isinf.(temp_vec_clustering)] # remove inf

        temp_vec_clustering2 = cluster_vec_unfinished[i,:]
        temp_vec_clustering2 = temp_vec_clustering2[.~isnan.(temp_vec_clustering2)] # remove nan
        temp_vec_clustering2 = temp_vec_clustering2[.~isinf.(temp_vec_clustering2)] # remove inf

        first_quartile_finished_clustering[i] = nquantile(temp_vec_clustering,10)[2]
        third_quartile_finished_clustering[i] = nquantile(temp_vec_clustering,10)[10]
        first_quartile_unfinished_clustering[i] = nquantile(temp_vec_clustering2,10)[2]
        third_quartile_unfinished_clustering[i] = nquantile(temp_vec_clustering2,10)[10]

        cluster_vec_unfinished_mean[i] = mean(temp_vec_clustering2)
        cluster_vec_finished_mean[i] = mean(temp_vec_clustering)

        NaN_count_finished[i] = 1 - length(temp_vec_clustering)/length(list)
        NaN_count_unfinished[i] = 1 - length(temp_vec_clustering2)/length(list)

    end
    figure()
    PyPlot.plot((time_start+time_ends)/2,NaN_count_finished,color="blue")
    PyPlot.plot((time_start+time_ends)/2,NaN_count_unfinished,color="red")

    figure()
    PyPlot.fill_between((time_start+time_ends)/2,first_quartile_finished_clustering,third_quartile_finished_clustering,color="blue",alpha=.5)
    PyPlot.plot((time_start+time_ends)/2,cluster_vec_finished_mean,color="blue")
    PyPlot.fill_between((time_start+time_ends)/2,first_quartile_unfinished_clustering,third_quartile_unfinished_clustering,color="red",alpha=.5)
    PyPlot.plot((time_start+time_ends)/2,cluster_vec_unfinished_mean,color="red")
    return 0
end

function psth_set(run_number)
    bins = 2
    time_total = 1000
    time_start = 0:bins:(time_total-bins)
    time_ends  = time_start + bins
    time_vec = (time_start+time_ends)/2
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    score_path = string(path_string,"/scores/")
    analysis_path = string(path_string,"/analysis/")
    list = readdir(score_path)
    list2 = readdir(analysis_path)
    cd(score_path)

    rate_vec_unfinished_e = zeros(length(time_vec))
    rate_vec_finished_e = zeros(length(time_vec))

    rate_vec_unfinished_i = zeros(length(time_vec))
    rate_vec_finished_i = zeros(length(time_vec))

    rate_vec_unfinished_ratio = zeros(length(time_vec))
    rate_vec_finished_ratio = zeros(length(time_vec))

    rate_vec_unfinished_ratio_mean = zeros(length(time_vec))
    rate_vec_finished_ratio_mean = zeros(length(time_vec))

    rate_vec_unfinished = zeros(length(time_vec))
    rate_vec_finished = zeros(length(time_vec))

    first_quartile_finished_e = zeros(length(time_start))
    third_quartile_finished_e = zeros(length(time_start))

    first_quartile_unfinished = zeros(length(time_vec))
    third_quartile_unfinished = zeros(length(time_vec))

    first_quartile_finished = zeros(length(time_vec))
    third_quartile_finished = zeros(length(time_vec))

    first_quartile_finished_i = zeros(length(time_start))
    third_quartile_finished_i = zeros(length(time_start))

    first_quartile_finished_ratio = zeros(length(time_start))
    third_quartile_finished_ratio = zeros(length(time_start))

    first_quartile_unfinished_e = zeros(length(time_start))
    third_quartile_unfinished_e = zeros(length(time_start))

    first_quartile_unfinished_i = zeros(length(time_start))
    third_quartile_unfinished_i = zeros(length(time_start))

    first_quartile_unfinished_ratio = zeros(length(time_start))
    third_quartile_unfinished_ratio = zeros(length(time_start))

    NaN_count_finished = zeros(length(time_vec))
    NaN_count_unfinished = zeros(length(time_vec))
    for i = 1:length(list)
        print(i/length(list))
        print("\n")
        data = load(list[i])["score_vals"]
        if((data[3]<.95) && (data[3]>0)) # this structure only works for early trial sets
            #data_e,data_i = psth_from_file(list[i],bins,time_total)
            psth_data = load(string(analysis_path,list2[i]))
            data_e = psth_data["PSTH_e"]
            data_i = psth_data["PSTH_i"]

            rate_vec_unfinished_e = [rate_vec_unfinished_e data_e]
            rate_vec_unfinished_i = [rate_vec_unfinished_i data_i]


            ratio = data_i./data_e
            rate_vec_unfinished_ratio = [rate_vec_unfinished_ratio ratio]
        elseif(data[3]>.95)
            psth_data = load(string(analysis_path,list2[i]))
            data_e = psth_data["PSTH_e"]
            data_i = psth_data["PSTH_i"]

            rate_vec_finished_e   = [rate_vec_finished_e data_e]
            rate_vec_finished_i   = [rate_vec_finished_i data_i]

            ratio = data_i./data_e
            rate_vec_finished_ratio = [rate_vec_finished_ratio ratio]
        end
    end

    for i = 1:length(time_start)
        first_quartile_unfinished_e[i] = nquantile(rate_vec_unfinished_e[i,:],10)[2]
        third_quartile_unfinished_e[i] = nquantile(rate_vec_unfinished_e[i,:],10)[10]
        first_quartile_unfinished_i[i] = nquantile(rate_vec_unfinished_i[i,:],10)[2]
        third_quartile_unfinished_i[i] = nquantile(rate_vec_unfinished_i[i,:],10)[10]

        first_quartile_finished_e[i] = nquantile(rate_vec_finished_e[i,:],10)[2]
        third_quartile_finished_e[i] = nquantile(rate_vec_finished_e[i,:],10)[10]
        first_quartile_finished_i[i] = nquantile(rate_vec_finished_i[i,:],10)[2]
        third_quartile_finished_i[i] = nquantile(rate_vec_finished_i[i,:],10)[10]



        temp_vec = rate_vec_finished_ratio[i,:]
        temp_vec = temp_vec[.~isnan.(temp_vec)] # remove nan
        temp_vec = temp_vec[.~isinf.(temp_vec)] # remove inf

        temp_vec2 = rate_vec_unfinished_ratio[i,:]
        temp_vec2 = temp_vec2[.~isnan.(temp_vec2)]
        temp_vec2 = temp_vec2[.~isinf.(temp_vec2)]



        rate_vec_unfinished_ratio_mean[i] = mean(temp_vec2)
        rate_vec_finished_ratio_mean[i] = mean(temp_vec)

        first_quartile_finished_ratio[i] = nquantile(temp_vec,10)[2]
        third_quartile_finished_ratio[i] = nquantile(temp_vec,10)[10]
        first_quartile_unfinished_ratio[i] = nquantile(temp_vec2,10)[2]
        third_quartile_unfinished_ratio[i] = nquantile(temp_vec2,10)[10]


        NaN_count_finished[i] = 1 - length(temp_vec)/length(list)
        NaN_count_unfinished[i] = 1 - length(temp_vec2)/length(list)

    end

    rate_vec_unfinished_e = mean(rate_vec_unfinished_e,2)
    rate_vec_unfinished_i = mean(rate_vec_unfinished_i,2)

    rate_vec_finished_e   = mean(rate_vec_finished_e,2)
    rate_vec_finished_i   = mean(rate_vec_finished_i,2)


    figure()
    PyPlot.fill_between((time_start+time_ends)/2,first_quartile_finished_e,third_quartile_finished_e,color="blue",alpha=.5)
    PyPlot.fill_between((time_start+time_ends)/2,first_quartile_unfinished_e,third_quartile_unfinished_e,color="red",alpha=.5)
    PyPlot.plot((time_start+time_ends)/2,rate_vec_finished_e,color="blue")
    PyPlot.plot((time_start+time_ends)/2,rate_vec_unfinished_e,color="red")

    figure()
    PyPlot.fill_between((time_start+time_ends)/2,first_quartile_finished_i,third_quartile_finished_i,color="blue",alpha=.5)
    PyPlot.fill_between((time_start+time_ends)/2,first_quartile_unfinished_i,third_quartile_unfinished_i,color="red",alpha=.5)
    PyPlot.plot((time_start+time_ends)/2,rate_vec_finished_i,color="blue")
    PyPlot.plot((time_start+time_ends)/2,rate_vec_unfinished_i,color="red")

    figure()
    PyPlot.fill_between((time_start+time_ends)/2,first_quartile_finished_ratio,third_quartile_finished_ratio,color="blue",alpha=.5)
    PyPlot.plot((time_start+time_ends)/2,rate_vec_finished_ratio_mean,color="blue")
    PyPlot.fill_between((time_start+time_ends)/2,first_quartile_unfinished_ratio,third_quartile_unfinished_ratio,color="red",alpha=.5)
    PyPlot.plot((time_start+time_ends)/2,rate_vec_unfinished_ratio_mean,color="red")

    figure()
    PyPlot.plot((time_start+time_ends)/2,NaN_count_finished,color="blue")
    PyPlot.plot((time_start+time_ends)/2,NaN_count_unfinished,color="red")

    return 0
end

function ratio_time_corr(run_number)
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    score_path = string(path_string,"/scores/")
    list = readdir(score_path)

    cd(score_path)
    time_scores = []
    rate_scores = []
    for i = 1:length(list)
        print(i/length(list))
        print("\n")
        data = load(list[i])["score_vals"]
        if(data[3]<.95)
            push!(rate_scores,data[2]/data[1])
            push!(time_scores,data[3])
        end
    end
    figure()
    PyPlot.scatter(time_scores,rate_scores,s=.5)
    return 0
end

function cluster_vis(run_number;one_plot=false)
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    score_path = string(path_string,"/scores/")
    list = readdir(score_path)

    cd(score_path)
    time_scores_unfinished = Array{Float64,1}()
    cycle_scores_unfinished = Array{Float64,1}()
    in_scores_unfinished = Array{Float64,1}()
    out_scores_unfinished = Array{Float64,1}()
    middle_scores_unfinished = Array{Float64,1}()
    all_unfinished_mean = Array{Float64,1}()
    cycle_scores_unfinished_median = Array{Float64,1}()
    in_scores_unfinished_median = Array{Float64,1}()
    out_scores_unfinished_median = Array{Float64,1}()
    middle_scores_unfinished_median = Array{Float64,1}()
    all_unfinished_median = Array{Float64,1}()

    time_scores_finished = Array{Float64,1}()
    cycle_scores_finished = Array{Float64,1}()
    in_scores_finished = Array{Float64,1}()
    out_scores_finished = Array{Float64,1}()
    middle_scores_finished = Array{Float64,1}()
    all_finished_mean = Array{Float64,1}()
    cycle_scores_finished_median = Array{Float64,1}()
    in_scores_finished_median = Array{Float64,1}()
    out_scores_finished_median = Array{Float64,1}()
    middle_scores_finished_median = Array{Float64,1}()
    all_finished_median = Array{Float64,1}()

    for i = 1:length(list)
        print(i/length(list))
        print("\n")
        data = load(list[i])["score_vals"]
        if(data[3]<.95)
            push!(cycle_scores_unfinished,data[4])
            push!(in_scores_unfinished,data[7])
            push!(out_scores_unfinished,data[6])
            push!(middle_scores_unfinished,data[5])
            push!(time_scores_unfinished,data[3])

            push!(cycle_scores_unfinished_median,data[8])
            push!(in_scores_unfinished_median,data[11])
            push!(out_scores_unfinished_median,data[10])
            push!(middle_scores_unfinished_median,data[9])

            push!(all_unfinished_mean,data[12])
            push!(all_unfinished_median,data[13])

        else
            push!(cycle_scores_finished,data[4])
            push!(in_scores_finished,data[7])
            push!(out_scores_finished,data[6])
            push!(middle_scores_finished,data[5])
            push!(time_scores_finished,data[3])

            push!(cycle_scores_finished_median,data[8])
            push!(in_scores_finished_median,data[11])
            push!(out_scores_finished_median,data[10])
            push!(middle_scores_finished_median,data[9])

            push!(all_finished_mean,data[12])
            push!(all_finished_median,data[13])

        end
    end
    if(one_plot)
        figure()
        PyPlot.scatter(cycle_scores_unfinished,time_scores_unfinished,s=.5,color="blue")
        PyPlot.scatter(out_scores_unfinished,time_scores_unfinished,s=.5,color="red")
        PyPlot.scatter(middle_scores_unfinished,time_scores_unfinished,s=.5,color="green")
        PyPlot.scatter(in_scores_unfinished,time_scores_unfinished,s=.5,color="black")
    else
        figure()
        PyPlot.scatter(cycle_scores_unfinished,time_scores_unfinished,s=.5,color="red")
        PyPlot.scatter(cycle_scores_finished,time_scores_finished,s=.5,color="blue")
        figure()
        PyPlot.scatter(out_scores_unfinished,time_scores_unfinished,s=.5,color="red")
        PyPlot.scatter(out_scores_finished,time_scores_finished,s=.5,color="blue")
        figure()
        PyPlot.scatter(middle_scores_unfinished,time_scores_unfinished,s=.5,color="red")
        PyPlot.scatter(middle_scores_finished,time_scores_finished,s=.5,color="blue")
        figure()
        PyPlot.scatter(in_scores_unfinished,time_scores_unfinished,s=.5,color="red")
        PyPlot.scatter(in_scores_finished,time_scores_finished,s=.5,color="blue")
    end
    figure()
    PyPlot.scatter3D(middle_scores_finished,in_scores_finished,cycle_scores_finished,s=.5,color="blue")
    PyPlot.scatter3D(middle_scores_unfinished,in_scores_unfinished,cycle_scores_unfinished,s=.5,color="red")

    nbins = 20
    figure() #middle
    binned = minimum([middle_scores_unfinished; middle_scores_finished]):.00001:maximum([middle_scores_unfinished; middle_scores_finished])
    dist1 = plt[:hist](middle_scores_finished,bins=binned,color="blue",alpha=0.6,normed=true)
    dist2 = plt[:hist](middle_scores_unfinished,bins=binned,color="red",alpha=0.6,normed=true)
    p_vals = ApproximateTwoSampleKSTest(middle_scores_unfinished, middle_scores_finished)
    print(p_vals)

    figure() #in
    binned = minimum([in_scores_unfinished; in_scores_finished]):.00001:maximum([in_scores_unfinished; in_scores_finished])
    dist1 = plt[:hist](in_scores_finished,bins=binned,color="blue",alpha=0.6,normed=true)
    dist2 = plt[:hist](in_scores_unfinished,bins=binned,color="red",alpha=0.6,normed=true)
    p_vals = ApproximateTwoSampleKSTest(in_scores_unfinished, in_scores_finished)
    print(p_vals)

    figure() #out
    binned = minimum([out_scores_unfinished; out_scores_finished]):.00001:maximum([out_scores_unfinished; out_scores_finished])
    dist1 = plt[:hist](out_scores_finished,bins=binned,color="blue",alpha=0.6,normed=true)
    dist2 = plt[:hist](out_scores_unfinished,bins=binned,color="red",alpha=0.6,normed=true)
    p_vals = ApproximateTwoSampleKSTest(out_scores_unfinished, out_scores_finished)
    print(p_vals)

    figure() #cycle
    binned = minimum([cycle_scores_unfinished; cycle_scores_finished]):.00001:maximum([cycle_scores_unfinished; cycle_scores_finished])
    dist1 = plt[:hist](cycle_scores_finished,bins=binned,color="blue",alpha=0.6,normed=true)
    dist2 = plt[:hist](cycle_scores_unfinished,bins=binned,color="red",alpha=0.6,normed=true)
    p_vals = ApproximateTwoSampleKSTest(cycle_scores_unfinished, cycle_scores_finished)
    print(p_vals)



    return 0
end

function rate_time_corr(run_number)
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    score_path = string(path_string,"/scores/")
    list = readdir(score_path)

    cd(score_path)
    time_scores = []
    rate_scores_i = []
    rate_scores_e = []
    for i = 1:length(list)
        print(i/length(list))
        print("\n")
        data = load(list[i])["score_vals"]
        if(data[3]<.95)
            if((data[1]<1000) & (data[2]<1000) )
                push!(rate_scores_e,data[1])
                push!(rate_scores_i,data[2])
                push!(time_scores,data[3])
            end
        end
    end
    figure()
    PyPlot.scatter(time_scores,rate_scores_i,color="red",s=.5)
    PyPlot.scatter(time_scores,rate_scores_e,color="blue",s=.5)
    return 0
end

function raster_grab(run_number;time_range=[.9 1], rate_range=[0 5])
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    score_path = string(path_string,"/scores/")
    spikes_path = string(path_string,"/spikes/")
    list = readdir(score_path)
    list2= readdir(spikes_path)
    cd(score_path)
    for i = 1:length(list)
        print(i/length(list))
        print("\n")
        data = load(list[i])["score_vals"]
        if((data[3]>time_range[1]) && (data[3]<time_range[2]) && (data[1]>rate_range[1]) && (data[1]<rate_range[2]))
            cd(spikes_path)
            rasterplot_from_file(list2[i])
            cd(score_path)
        end
    end
    return 0
end

function fraction_inhib_corr(run_number)
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    score_path = string(path_string,"/scores/")
    list = readdir(score_path)

    cd(score_path)
    time_scores = []
    frac_scores = []
    rate_scores_e = []
    rate_scores_i = []
    for i = 1:length(list)
        print(i/length(list))
        print("\n")
        data = load(list[i])["score_vals"]
        projected = load(list[i])["projected_neurons"]
        frac = length(find(projected.>4000))/length(projected)
        if(data[3]<.95)
            push!(rate_scores_e,data[1])
            push!(rate_scores_i,data[2])
            push!(frac_scores,frac)
            push!(time_scores,data[3])
        end
    end
    figure()
    PyPlot.scatter(time_scores,frac_scores,s=.5)
    figure()
    PyPlot.scatter(rate_scores_e,frac_scores,s=.5,color="blue")
    PyPlot.scatter(rate_scores_i,frac_scores,s=.5,color="red")
    return 0
end

function graph_energy(run_number)
    path_string = string("/Users/kyle/desktop/Jason_Data/",run_number)
    score_path = string(path_string,"/scores/")
    list = readdir(score_path)
    time_scores_unfinished = Array{Float64,1}()
    time_scores_finished = Array{Float64,1}()
    energy_score_unfinsihed = Array{Float64,1}()
    energy_score_finished = Array{Float64,1}()
    cd(score_path)
    for i = 1:length(list)
        print(i/length(list))
        print("\n")
        data = load(list[i])["score_vals"]
        if(data[3]<.95)

            push!(time_scores_unfinished,data[3])
            push!(energy_score_unfinsihed,data[14])

        else

            push!(time_scores_finished,data[3])
            push!(energy_score_finished,data[14])

        end
    end
    print("\n\n",energy_score_finished,"\n\n")
    print("\n\n",energy_score_unfinsihed,"\n\n")

end

function fraction_recruited_corr(run_number)
end
