using ProgressMeter

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

function main(spikes,dt)
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