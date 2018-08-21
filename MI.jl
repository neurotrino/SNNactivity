using ProgressMeter

function main(graph,spikes)
	dt_vals = [5,10,15,20,25]
	rate = length.(spikes)/10

	cycle_wd,middleman_wd,fanin_wd,fanout_wd,all_wd = clustered_motifs(graph)
	cycle_wd_uw,middleman_wd_uw,fanin_wd_uw,fanout_wd_uw,all_wd_uw = unweighted_clustered_motifs(graph)
	
	figure()
	PyPlot.scatter(mean(graph,1),rate,s=.1)
	ylabel("rate")
	xlabel("mean pre")
	title("rate v mean pre synaptic strength, wiring diagram")

	figure()
	PyPlot.scatter(mean(graph,2),rate,s=.1)
	ylabel("rate")
	xlabel("mean post")
	title("rate v mean post synaptic strength, wiring diagram")

	figure()
	PyPlot.scatter(all_wd,rate,s=.1)
	ylabel("rate")
	xlabel("all clusters")
	title("rate v all clusters, weighted wiring diagram")

	figure()
	PyPlot.scatter(fanout_wd,rate,s=.1)
	ylabel("rate")
	xlabel("fanout clusters")
	title("rate v fanout clusters, weighted wiring diagram")

	figure()
	PyPlot.scatter(fanin_wd,rate,s=.1)
	ylabel("rate")
	xlabel("fanin clusters")
	title("rate v fanin clusters, weighted wiring diagram")

	figure()
	PyPlot.scatter(middleman_wd,rate,s=.1)
	ylabel("rate")
	xlabel("middleman clusters")
	title("rate v middleman clusters, weighted wiring diagram")

	figure()
	PyPlot.scatter(middleman_wd,rate,s=.1)
	ylabel("rate")
	xlabel("cylce clusters")
	title("rate v cycle clusters, weighted wiring diagram")

	figure()
	PyPlot.scatter(all_wd_uw,rate,s=.1)
	ylabel("rate")
	xlabel("all clusters")
	title("rate v all clusters, unweighted wiring diagram")

	figure()
	PyPlot.scatter(fanout_wd_uw,rate,s=.1)
	ylabel("rate")
	xlabel("fanout clusters")
	title("rate v fanout clusters, unweighted wiring diagram")

	figure()
	PyPlot.scatter(fanin_wd_uw,rate,s=.1)
	ylabel("rate")
	xlabel("fanin clusters")
	title("rate v fanin clusters, unweighted wiring diagram")

	figure()
	PyPlot.scatter(middleman_wd_uw,rate,s=.1)
	ylabel("rate")
	xlabel("middleman clusters")
	title("rate v middleman clusters, unweighted wiring diagram")

	figure()
	PyPlot.scatter(middleman_wd_uw,rate,s=.1)
	ylabel("rate")
	xlabel("cylce clusters")
	title("rate v cycle clusters, unweighted wiring diagram")

	for i = 1:length(dt_vals)
		MI_graph = generate_mi_graph(spikes,dt_vals[i])
		MI_graph[MI_graph .< 0] = 0
		cycle_MI,middleman_MI,fanin_MI,fanout_MI,all_MI = clustered_motifs(MI_graph)
		cycle_MI_uw,middleman_MI_uw,fanin_MI_uw,fanout_MI_uw,all_MI_uw = unweighted_clustered_motifs(MI_graph)
	
		figure()
		PyPlot.scatter(mean(MI_graph,1),rate,s=.1)
		ylabel("rate")
		xlabel("mean pre")
		title(string("rate v mean pre synaptic strength MI"," ",string(dt_vals[i])," ms"))

		figure()
		PyPlot.scatter(mean(MI_graph,2),rate,s=.1)
		ylabel("rate")
		xlabel("mean post")
		title(string("rate v mean post synaptic strength MI"," ",string(dt_vals[i])," ms"))

		figure()
		PyPlot.scatter(all_MI,rate,s=.1)
		ylabel("rate")
		xlabel("all clusters")
		title(string("rate v all clusters, weighted MI"," ",string(dt_vals[i])," ms"))

		figure()
		PyPlot.scatter(fanout_MI,rate,s=.1)
		ylabel("rate")
		xlabel("fanout clusters")
		title(string("rate v fanout clusters, weighted MI"," ",string(dt_vals[i])," ms"))

		figure()
		PyPlot.scatter(fanin_MI,rate,s=.1)
		ylabel("rate")
		xlabel("fanin clusters")
		title(string("rate v fanin clusters, weighted MI"," ",string(dt_vals[i])," ms"))

		figure()
		PyPlot.scatter(middleman_MI,rate,s=.1)
		ylabel("rate")
		xlabel("middleman clusters")
		title(string("rate v middleman clusters, weighted MI"," ",string(dt_vals[i])," ms"))

		figure()
		PyPlot.scatter(middleman_MI,rate,s=.1)
		ylabel("rate")
		xlabel("cylce clusters")
		title(string("rate v cycle clusters, weighted MI"," ",string(dt_vals[i])," ms"))

		figure()
		PyPlot.scatter(all_MI_uw,rate,s=.1)
		ylabel("rate")
		xlabel("all clusters")
		title(string("rate v all clusters, unweighted MI"," ",string(dt_vals[i])," ms"))

		figure()
		PyPlot.scatter(fanout_MI_uw,rate,s=.1)
		ylabel("rate")
		xlabel("fanout clusters")
		title(string("rate v fanout clusters, unweighted MI"," ",string(dt_vals[i])," ms"))

		figure()
		PyPlot.scatter(fanin_MI_uw,rate,s=.1)
		ylabel("rate")
		xlabel("fanin clusters")
		title(string("rate v fanin clusters, unweighted MI"," ",string(dt_vals[i])," ms"))

		figure()
		PyPlot.scatter(middleman_MI_uw,rate,s=.1)
		ylabel("rate")
		xlabel("middleman clusters")
		title(string("rate v middleman clusters, unweighted MI"," ",string(dt_vals[i])," ms"))

		figure()
		PyPlot.scatter(middleman_MI_uw,rate,s=.1)
		ylabel("rate")
		xlabel("cylce clusters")
		title(string("rate v cycle clusters, unweighted MI"," ",string(dt_vals[i])," ms"))
	end
end

function binary_raster_gen(spike_set,discretize_dt)
	bins = 0:discretize_dt:run_total
	raster = zeros(length(spike_set),length(bins)-1)
	for i = 1:length(spike_set)
		discrete_train = unique(round.(spike_set[i]/discretize_dt))
		discrete_train = discrete_train[discrete_train.>0]

		for j = 1:length(discrete_train)
			raster[i,Int(discrete_train[j])] = 1
		end
	end
	return raster
end

function generate_mi_graph(spikes,dt)
	raster = binary_raster_gen(spikes,dt)
	MI_graph = confMI_mat(raster)
	signed_graph = signed_MI(MI_graph,raster)
	pos_graph = pos(signed_graph)
	reexpress_graph = reexpress_param(pos_graph)
	background_graph = background(reexpress_graph)
	residual_graph = residual(background_graph,MI_graph)
	normed_MI_graph = normed_residual(residual_graph)
	return normed_MI_graph
end

function signed_MI(graph,raster)
	# takes MI graph returns signed_MI MI graph
	neurons = size(graph)[1]
	signed_graph = copy(graph)
	@showprogress 1 "Computing signed_MI Graph" for post = 1:neurons
		for pre = (post+1):neurons
			corr_mat = cor(hcat(raster[pre,:],raster[post,:]))
			factor = sign(corr_mat[1,2])
			if ~isnan(factor)
				signed_graph[post,pre] *= factor
				signed_graph[pre,post] *= factor
			end
		end
	end
	return signed_graph
end

function pos(graph)
	#takes signed_MI MI graph, returns positive version
	pos_graph = copy(graph)
	pos_graph[find(x->x<0,graph)] = 0
	return pos_graph 
end

function reexpress_param(graph)
	#takes pos graph and returns redist graph
	steps = 100
	data = graph[:]
	data = data[find(x->x>0,data)]
	upper_exp=10
	lower_exp=0.00000000000001
	upper_skew = skewness(data.^upper_exp)
	lower_skew = skewness(data.^lower_exp)
	@showprogress 1 "Computing Reexpress Parameter" for i = 1:steps
		new_exp = (upper_exp + lower_exp)/2
		new_skew = skewness(data.^new_exp)
	    if new_skew<0
	        lower_exp=new_exp;
	        lower_skew=new_skew;
	    else
	        upper_exp=new_exp;
	        upper_skew=new_skew;
	    end
	end
	if abs(upper_skew)<abs(lower_skew)
		reexpress = upper_exp
	else
		reexpress = lower_exp
	end
	reexpress_graph = copy(graph)
	reexpress_graph = reexpress_graph.^reexpress
	return reexpress_graph
end

function background(graph)
	#takes reexpress graph returns background graph
	background = copy(graph)
	neurons = size(graph)[1]
	@showprogress 1 "Computing Background Graph" for pre = 1:neurons
		for post = 1:neurons
			if (post != pre)
				background[post,pre] = mean(graph[post,1:neurons .!= pre])*mean(graph[1:neurons .!= post,pre])
			end
		end
	end
	return background
end

function residual(background_graph,graph)
	#takes background graph and MI graph
	#returns residual graph
	residual = copy(graph)
	neurons = size(graph)[1]
	b,m = linreg(background_graph[:],graph[:])
	residual = graph - (m*background_graph + b)
	return residual
end

function normed_residual(graph)
	#tales residual graph returns
	#normalized residual graph
	norm_residual = copy(graph)
	neurons = size(graph)[1]
	@showprogress 1 "Normalizing Residual Graph" for pre = 1:neurons
		for post = 1:neurons
			if (post != pre)
				norm_residual[post,pre] = std(graph[post,1:neurons .!= pre])*std(graph[1:neurons .!= post,pre])
			end
		end
	end
	cutoff = median(norm_residual)
	norm_residual = 1./(sqrt.(max.(norm_residual,ones(neurons,neurons)*cutoff)))
	return norm_residual.*graph
end

function MI(train_1,train_2,lag,alpha)
	MI = 0
	states = [0,1]
	for i = 1:length(states)
		i_inds = find(x->x==states[i],train_1)
		if (length(i_inds) > 0)
			p_i = length(i_inds)/length(train_1)
			for j = 1:length(states)
				j_inds = find(x->x==states[j],train_2) - lag
				if length(j_inds) > 0
					j_inds = j_inds[j_inds>0]
					p_j = length(j_inds)/(length(train_2)-lag)
					p_i_and_j = length(intersect(i_inds,j_inds))/(length(train_1)-lag)
					if (alpha > 0)
						MI = MI + alpha + (1-alpha) * p_i_and_j * log2(p_i_and_j/(p_i*p_j))
					elseif (p_i_and_j > 0)
						MI += p_i_and_j * log2(p_i_and_j/(p_i*p_j))
					end
				end
			end
		end
	end 
	return MI
end

function confMI(train_1,train_2,lag,alpha)
	MI = 0
	states = [0,1]
	for i = 1:length(states)
		i_inds = find(x->x==states[i],train_1)
		p_i = length(i_inds)/length(train_1)
		if (length(i_inds) > 0)
			for j = 1:length(states)
				j_inds = find(x->x==states[j],train_2)
				j_inds_lagged = j_inds - lag
				if (length(j_inds) > 0)
					j_inds_lagged = j_inds_lagged[j_inds_lagged.>0]
					j_inds = union(j_inds,j_inds_lagged)
					p_j = length(j_inds)/(length(train_2)-lag)
					p_i_and_j = length(intersect(i_inds,j_inds))/(length(train_1)-lag)
					if (alpha > 0)
						MI = MI + alpha + (1-alpha) * p_i_and_j * log2(p_i_and_j/(p_i*p_j))
					elseif (p_i_and_j > 0)
						MI += p_i_and_j * log2(p_i_and_j/(p_i*p_j))
					end
				end
			end
		end
	end 
	return MI
end

function confMI_mat(raster)
	lag = 1
	alpha = 0
	neurons = size(raster)[1]
	mat = zeros(neurons,neurons)
	@showprogress 1 "Computing Confluent MI"  for post = 1:neurons
		for pre = 1:neurons
			if post != pre
				mat[post,pre] = confMI(raster[pre,:],raster[post,:],lag,alpha)
			end
		end
	end
	return mat
end

function consecMI_mat(raster)
	lag = 1
	alpha = 0
	neurons = size(raster)[1]
	mat = zeros(neurons,neurons)
	@showprogress 1 "Computing Consectuve MI" for post = 1:neurons
		for pre = 1:neurons
			if post != pre
				mat[post,pre] = MI(raster[pre,:],raster[post,:],lag,alpha)
			end
		end
	end
	return mat
end

function simulMI_mat(raster)
	lag = 0
	alpha = 0
	neurons = size(raster)[1]
	mat = zeros(neurons,neurons)
	@showprogress 1 "Computing Simultaneous MI" for post = 1:neurons
		for pre = 1:neurons
			if post != pre
				mat[post,pre] = MI(raster[pre,:],raster[post,:],lag,alpha)
			end
		end
	end
	return mat
end

function unweighted_clustered_motifs(adj_mat) # revision as of 20180728

	if(isempty(adj_mat)) # with the way calculating recruitment, unlikely to be the case.
		return NaN,NaN,NaN,NaN,NaN
	elseif(isempty(find(adj_mat.!=0)))
		return NaN,NaN,NaN,NaN,NaN
	end

	W = adj_mat'; # Wij convention in Fagiolo is the transpose of ours

	dim = size(adj_mat)[1];

	if(!isempty(find(W.!=0)))
		W[W .!= 0] = 1;
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

function clustered_motifs(adj_mat) # revision as of 20180728

	if(isempty(adj_mat)) # with the way calculating recruitment, unlikely to be the case.
		return NaN,NaN,NaN,NaN,NaN
	elseif(isempty(find(adj_mat.!=0)))
		return NaN,NaN,NaN,NaN,NaN
	end

	W = adj_mat'; # Wij convention in Fagiolo is the transpose of ours

	dim = size(adj_mat)[1];

	if(!isempty(find(W.!=0)))
		W = (W).^(1/3);
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