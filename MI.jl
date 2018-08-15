#what to do about 0. alpha + (1-alpha)*p or should i filter NaN
#check post pre conventions
#could do this with a vector of states [0,1] and not declare
#everything explicitely but why bother for now

#where do we land on doing consecutive vs simultaneous vs confluent
#can code up confluent if we want it

function MI(train_1,train_2,lag,alpha)
	p_i_1 = sum(train_1)/length(train_1)
	p_i_0 = 1 - p_i_1

	p_j_plus_lag_1 = sum(train_2[1+lag:end])/(length(train_1)-lag)
	p_j_plus_lag_0 = 1 - p_j_plus_lag_1

	p_i_1_and_j_plus_lag_1 = length(intersect(find(x->x==1,train_1),find(x->x==1,train_1)-lag))/(length(train_1)-lag)
	p_i_0_and_j_plus_lag_1 = length(intersect(find(x->x==0,train_1),find(x->x==1,train_1)-lag))/(length(train_1)-lag)
	p_i_1_and_j_plus_lag_0 = length(intersect(find(x->x==1,train_1),find(x->x==0,train_1)-lag))/(length(train_1)-lag)
	p_i_0_and_j_plus_lag_0 = length(intersect(find(x->x==0,train_1),find(x->x==0,train_1)-lag))/(length(train_1)-lag)

	p_i_1 = alpha + (1-alpha)*p_i_1
	p_i_0 = alpha + (1-alpha)*p_i_0
	p_j_plus_lag_1 = alpha + (1-alpha)*p_j_plus_lag_1
	p_j_plus_lag_0 = alpha + (1-alpha)*p_j_plus_lag_0
	p_i_1_and_j_plus_lag_1 = alpha + (1-alpha)*p_i_1_and_j_plus_lag_1
	p_i_0_and_j_plus_lag_1 = alpha + (1-alpha)*p_i_0_and_j_plus_lag_1
	p_i_1_and_j_plus_lag_0 = alpha + (1-alpha)*p_i_1_and_j_plus_lag_0
	p_i_0_and_j_plus_lag_0 = alpha + (1-alpha)*p_i_0_and_j_plus_lag_0


	MI = 0
	MI += p_i_1_and_j_plus_lag_1*log2(p_i_1_and_j_plus_lag_1/(p_i_1*p_j_plus_lag_1))
	MI += p_i_0_and_j_plus_lag_1*log2(p_i_0_and_j_plus_lag_1/(p_i_0*p_j_plus_lag_1))
	MI += p_i_1_and_j_plus_lag_0*log2(p_i_1_and_j_plus_lag_0/(p_i_1*p_j_plus_lag_0))
	MI += p_i_0_and_j_plus_lag_0*log2(p_i_0_and_j_plus_lag_0/(p_i_0*p_j_plus_lag_0))
end

function consecMI_mat(raster)
	lag = 1
	alpha = .00001
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
	alpha = .00001
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