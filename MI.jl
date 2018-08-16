#what to do about 0. alpha + (1-alpha)*p or should i filter NaN
#check post pre conventions

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