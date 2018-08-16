#what to do about 0. alpha + (1-alpha)*p or should i filter NaN
#check post pre conventions

function signed(graph,raster)
	# takes MI graph returns signed MI graph
	N = size(graph)[1]
	signed_graph = copy(graph)
	for i = 1:N
		for j = i:N
			corr_mat = cor(hcat(raster[i,:],raster[j,:]))
			signed_graph[j,i] *= sign(corr_mat[1,2])
			signed_graph[i,j] *= sign(corr_mat[2,1])
		end
	end
	return signed_graph
end

function pos(graph)
	#takes signed MI graph, returns positive version
	pos_graph = copy(graph)
	pos_graph[find(x->x<0,graph)] = 0
	return pos_graph 
end

function reexpress_param(graph)
	#takes pos graph and returns redist graph
	steps = 100
	data = graph[:]
	data = data[find(x->x<=0,data)]
	upper_exp=10
	lower_exp=0.00000000000001
	upper_skew = skewness(data.^upper_exp)
	lower_skew = skewness(data.^lower_exp)
	for i = 1:steps
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
	print(reexpress,"\n\n")
	reexpress_graph = copy(graph)
	reexpress_graph = reexpress_graph.^reexpress_graph
	return reexpress_graph
end

function background(graph)
	#takes reexpress graph returns background graph
	background = copy(graph)
	N = size(graph)[1]
	for i = 1:N
		for j = 1:N
			background[j,i] = mean(graph[j,1:N .!= i])*mean(graph[1:N .!= j,i])
		end
	end
	return background
end

function residual(background_graph,graph)
	#takes background graph and MI graph
	#returns residual graph
	residual = copy(graph)
	N = size(graph)[1]
	data_y = graph[:]
	data_x = hcat(background_graph[:],ones(N*N))
	coefs = data_x\data_y
	m = coefs[1]
	b = coefs[2]
	residual = graph - (m*graph + b)
	return residual
end

function normed_residual(graph)
	#tales residual graph returns
	#normalized residual graph
	norm_residual = copy(graph)
	N = size(graph)[1]
	for i = 1:N
		for j = 1:N
			norm_residual[j,i] = std(graph[j,1:N .!= i])*std(graph[1:N .!= j,i])
		end
	end
	cutoff = median(norm_residual)
	norm_residual = 1./(sqrt.(max.(norm_residual,ones(N,N)*cutoff)))
	return norm_residual.*graph
end

function MI(train_1,train_2,lag,alpha)
	MI = 0
	states = [0,1]
	for i = 1:length(states)
		i_inds = find(x->x==states[i],train_1)
		p_i = length(i_inds)/length(train_1)
		for j = 1:length(states)
			j_inds = find(x->x==states[j],train_2[1+lag:end]) - lag
			p_j = length(j_inds)/(length(train_2)-lag)
			p_i_and_j = length(intersect(i_inds,j_inds))/(length(train_1)-lag)
			if (alpha > 0)
				MI = MI + alpha + (1-alpha) * p_i_and_j * log2(p_i_and_j/(p_i*p_j))
			elseif (p_i_and_j > 0)
				MI += p_i_and_j * log2(p_i_and_j/(p_i*p_j))
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
		for j = 1:length(states)
			j_inds = find(x->x==states[j],train_2)
			j_inds_lagged = j_inds - lag
			if j_inds_lagged[1] == 0
				j_inds_lagged = j_inds_lagged[2:end]
			end
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
	return MI
end

function confMI_mat(raster)
	lag = 1
	alpha = 0
	N = size(raster)[1]
	mat = zeros(N,N)
	for post = 1:N
		for pre = 1:N
			if post != pre
				N[post,pre] = confMI(raster[pre,:],raster[post,:],lag,alpha)
			end
		end
	end
end

function consecMI_mat(raster)
	lag = 1
	alpha = 0
	N = size(raster)[1]
	mat = zeros(N,N)
	for post = 1:N
		for pre = 1:N
			if post != pre
				N[post,pre] = MI(raster[pre,:],raster[post,:],lag,alpha)
			end
		end
	end
end

function simulMI_mat(raster)
	lag = 0
	alpha = 0
	N = size(raster)[1]
	mat = zeros(N,N)
	for post = 1:N
		for pre = 1:N
			if post != pre
				N[post,pre] = MI(raster[pre,:],raster[post,:],lag,alpha)
			end
		end
	end
end