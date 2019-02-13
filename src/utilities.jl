# include("ndgrid.jl")

function load_data(;path::String = "../data/gt_objective.csv", slice::Int64=30)::Array{Float64,1}
	
	data = squeeze(readdlm(path), 2)
	
	if slice < size(data, 1)
		data = data[1:slice:end] # downsampling
	end

	return data
end

function createtoydata(n1::Int64, n2::Int64)
    # Create toy data
    x1, y1 = meshgrid(linspace(-5.0, 5.0, n1), linspace(1.0, 6.0, n1))
    D1 = [x1[:] y1[:]]

    x2, y2 = meshgrid(linspace(-3.5,-1.5, n2), linspace(2.5, 4.5, n2))
    D2 = [x2[:] y2[:]]

    x3, y3 = meshgrid(linspace( 1.5, 3.5, n2), linspace(2.5, 4.5, n2))
    D3 = [x3[:] y3[:]]


    return [D1; D2; D3]

end

function meshgrid(x, y)
    return [i for j in y, i in x]
end

function unique_triplets(n::Int64)
    @assert n > 2
    
    triplets = zeros(Int64, n*binomial(n-1, 2), 3)
    counter = 0


    for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k
            counter += 1
            triplets[counter,:] = [i,j,k]
        end
    end

    return triplets
end

function label(data::Array{Float64,1}; probability_success::Array{Float64,3}=ones(size(data,1),size(data,1),size(data,1))) 
    label(reshape(data, size(data,1), 1), probability_success=probability_success)
end

function label(data::Array{Float64,2}; probability_success::Array{Float64,3}=ones(size(data,1),size(data,1),size(data,1)))::Array{Int64,2}
	# probability represents the probability of swapping the order of a
	# random triplet

	# We prealocate the possible total amount of triplets. Before returning,
	# we clip the array 'triplets' to the amount of nonzero elements.
	n = size(data,1)
	triplets = zeros(Int64, n*binomial(n-1, 2), 3)
	counter = 0

	D = distances(data, size(data,1))::Array{Float64,2}

	for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k

			@inbounds mistake = probability_success[i,j,k] .<= 1 - rand()

			if D[i,j] < D[i,k]
				counter +=1
            	if !mistake
            		@inbounds triplets[counter,:] = [i, j, k]
            	else
            		@inbounds triplets[counter,:] = [i, k, j]
            	end
            elseif D[i,j] > D[i,k]
				counter += 1
				if !mistake
					@inbounds triplets[counter,:] = [i, k, j]
				else
					@inbounds triplets[counter,:] = [i, j, k]
				end
			end
	    end
	end

    return triplets[1:counter,:]
end

function label_distances(D::Array{Int64,2}; probability_success::Array{Float64,3}=ones(size(D,1),size(D,1),size(D,1)))::Array{Int64,2}
    n = size(D,1)
    triplets = zeros(Int64, n*binomial(n-1, 2), 3)
    counter = 0

    for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k

            @inbounds mistake = probability_success[i,j,k] .<= 1 - rand()

            if D[i,j] < D[i,k]
                counter +=1
                if !mistake
                    @inbounds triplets[counter,:] = [i, j, k]
                else
                    @inbounds triplets[counter,:] = [i, k, j]
                end
            elseif D[i,j] > D[i,k]
                counter += 1
                if !mistake
                    @inbounds triplets[counter,:] = [i, k, j]
                else
                    @inbounds triplets[counter,:] = [i, j, k]
                end
            end
        end
    end

    return triplets[1:counter,:]
end

function relabel!(data::Array{Float64,2}, 
	              S::Array{Int64,2}, 
	              probability_success::Array{Float64,3})

	n = size(data,1)
	
	distances = pairwise(Cityblock(), data') # Compares columns

	for t = 1:size(S,1)
		i,j,k = S[t,:]

		mistake = probability_success[i,j,k] .<= 1 - rand()

		if distances[i,j] < distances[i,k] && !mistake
			if !mistake
				S[t,:] = [i, j, k]
			else
				S[t,:] = [i, k, j]
			end
		elseif distances[i,j] > distances[i,k] && !mistake
			if !mistake
				S[t,:] = [i, k, j]
			else
				S[t,:] = [i, j, k]
			end
		end
	end

end

function generate_T(n::Int64)
	@assert n > 3

	T = Array{Array{Int64,1},1}(0)

	for i = 1:n, j = 1:n, k = 1:n
		if i != j && i != k && j != k && j < k
			push!(T, [i, j, k])
		end
	end

	return T[shuffle(1:end)]
end

function T_to_S!(data::Array{Float64,2}, 
				 T::Array{Array{Int64,1},1},
				 S::Array{Array{Int64,1},1},
				 probability_success::Array{Float64, 3})

	n = size(data, 1)

	distances = pairwise(Cityblock(), data') # Compares columns

	for i = 1:2*ceil(n*log(n))

		i, j, k = pop!(T)
		# mistake = rand() .<= (1 - probability_success[i,j,k])

		if distances[i,j] < distances[i,k] #&& !mistake
			# Correctly labeled
            S = push!(S, [i, j, k])
        elseif distances[i,j] > distances[i,k] #&& mistake
			# Mistake
			S = push!(S, [i, j, k])
		end
	end
end

function S_to_triplets(S::Array{Array{Int64,1},1})
	@assert !isempty(S)

	no_triplets = size(S, 1)
	triplets = zeros(Int64, no_triplets, 3)
	
	for t = 1:no_triplets
		triplets[t,:] = S[t]
	end

	return triplets
end

# A = A[[1, 3:end], :]

function subset(triplets::Array{Int64,2}, fraction::Real)
        @assert !isempty(triplets) # Check for no triplets

        n = maximum(triplets)
        amount = floor(Int64, fraction * n * binomial(n-1, 2))

        triplets = triplets[shuffle(1:end),:]

        return triplets[1:amount, :]
end

function subset!(triplets::Array{Int64,2}, number_of_triplets::Real)
		@assert !isempty(triplets) # Check for no triplets
		@assert number_of_triplets > 2.0

        S = triplets[1:floor(Int64, number_of_triplets), :]
		triplets = triplets[ceil(Int64, number_of_triplets):end, :]

		return S
end

function distances(X::Array{Float64,1}, no_objects::Int64)::Array{Float64,2}
    return distances(reshape(X, size(X,1), 1), no_objects)
end

function distances(X::Array{Float64,2}, no_objects::Int64)::Array{Float64,2}
    D = zeros(Float64, no_objects, no_objects)

    for j = 1:no_objects, i = j:no_objects
        @inbounds D[i,j] = norm(X[i,:] - X[j,:])
    end

    return D + D'
end

function distances(X::Array{Int64,1}; no_objects::Int64=length(X))::Array{Int64,2}
    D = zeros(Int64, no_objects, no_objects)

    for j = 1:no_objects, i = j:no_objects
        @inbounds D[i,j] = abs(X[i] - X[j])
    end

    return D + D'
end

function triplet_violations(X::Array{Float64,2},
                            triplets::Array{Int64,2};
                            no_triplets = size(triplets,1)::Int64,
                            no_objects = maximum(triplets)::Int64,
                            no_dims = size(X,2)::Int64)
    "This function is more exact in floating point operations than 
    triplet_violations, but it is a bit slower and uses double the memory"

    D = zeros(Float64, no_objects, no_objects)

    for j = 1:no_objects, i = j:no_objects
        @inbounds D[i,j] = norm(X[i,:] - X[j,:])
    end

    D =  D + D'

    no_viol::Int64 = 0
    violations = zeros(Bool, no_triplets, )

    for t = 1:no_triplets
        violations[t] = D[triplets[t,1], triplets[t,2]] > D[triplets[t,1], triplets[t,3]]
    end
    
    no_viol = reduce(+, violations)

    return violations, no_viol/no_triplets
end

function correct_triplets(X::Array{Float64,2}, triplets::Array{Int64,2})
	
	violations, no_violations = triplet_violations(X, triplets)

	S = deepcopy(triplets)

	for t = 1:size(S,1)
		if violations[t]
			S[t,:] = [S[t,1], S[t,3], S[t,2]]
		end
	end

	return S
end

function flip_triplets!(triplets::Array{Int64,2}, fraction::Float64)
	amount = floor(Int64, fraction * size(triplets,1))

    flip_triplets!(triplets, amount)
end

function flip_triplets!(triplets::Array{Int64,2}, amount::Int64)
	for t = 1:amount
		i,j,k = triplets[t,:]
		triplets[t,:] = [i, k, j]
	end
end

function scale(data::Array{Int64,1}, X::Array{Float64,1})
    # We solve the scaling problem by min || aX - data - b||^2,
    # where (a,b) are the scale and offset parameters
    @assert size(data) == size(X)

    a, b = [X -ones(size(X))]\data

    return a*X - b

end

function scale(data::Array{Float64,1}, X::Array{Float64,1}; MSE::Bool=false)
    # We solve the scaling problem by min || aX - data - b||^2,
    # where (a,b) are the scale and offset parameters
    @assert size(data) == size(X)

    a, b = [X -ones(size(X))]\data

    if MSE
        return a*X - b, norm(a*X - b - data)^2
    else
        return a*X - b
    end
end

function scale(data::Array{Float64,1}, X::Array{Float64,1}, t1::Int64, t2::Int64)
    # We solve the scaling problem by min || aX - data - b||^2,
    # where (a,b) are the scale and offset parameters
    @assert size(data) == size(X)

    a, b = [X[t1:t2] -ones(size(X[t1:t2]))]\data[t1:t2]

    return a,b
end

"""
Find the best linear transformation (rotation, reflection, and bias)
to match Y to X.
"""

function procrustes(X::Array{Float64,2}, Y::Array{Float64,2}; reflect=false)

    @assert size(X) == size(Y) "X and Y must be the same size"

    nrows, ncols   = size(X)

    Xmean = mean(X, 1)
    Ymean = mean(Y, 1)
    
    # Center the columns (mean zero for the columns)
    V = eye(nrows) - 1/nrows * ones(nrows, nrows)
    X0 = V * X
    Y0 = V * Y
    
    # Checking that not all points are the same
    checkX = all([X[i,:] ≈ X[1,:] for i in 1:nrows])
    checkY = all([Y[i,:] ≈ Y[1,:] for i in 1:nrows])

    if !checkX && !checkY
        # We compute the Frobenius norm of the column-wise
        # centered arrays, so that we can scale to unit norm,
        # so that the Frobenius norm of the arrays is equal 
        # to 1.0
        Xnorm = vecnorm(X0)
        Ynorm = vecnorm(Y0)

        X0 = X0 / Xnorm
        Y0 = Y0 / Ynorm

        # We find the best rotation matrix R for Y
        U, Σ, V = svd(X0' * Y0)
        R = V * U'
        
        reflected = det(R) < 0
        if reflect != reflected
            # ... then either force a reflection, or undo one.
            V[:,end] = -V[:,end]
            Σ[end,end] = -Σ[end,end]
            R = V * U'
        end
        
        # The minimized unstandardized distance Σ(X0,b*Y0*R) is
        # ||X0||^2 + b^2*||Y0||^2 - 2*b*trace(R*X0'*Y0)
        traceR = sum(Σ) # == trace(sqrtm(A'*A)) when reflect is 'best'
        
        
        # The optimum scaling of Y in MSE sense
        b = traceR * Xnorm / Ynorm
        
        # The standardized distance between X and b*Y*R+c.
        d = 1 - traceR.^2

        Z = Xnorm * traceR * Y0 * R + repmat(Xmean, nrows, 1)
        
    # The degenerate cases: X all the same, and Y all the same.
    elseif checkX
        d = 0
        Z = repmat(Xmean, nrows, 1)
    else # !checkX & checkY
        d = 1
        Z = repmat(Xmean, nrows, 1)
    end

    return Z, d, Xnorm * traceR * R, Xmean, reflected
end
