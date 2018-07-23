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
    x1, y1 = meshgrid(linspace(-5.0, 5.0, n1), linspace(1.0 ,6.0, n1))
    D1 = [x1[:] y1[:]]

    x2, y2 = meshgrid(linspace(-3.5,-1.5, n2), linspace(2.5 ,4.5, n2))
    D2 = [x2[:] y2[:]]

    x3, y3 = meshgrid(linspace( 1.5, 3.5, n2), linspace(2.5 ,4.5, n2))
    D3 = [x3[:] y3[:]]


    return [D1; D2; D3]

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

        no_triplets = size(triplets, 1)
        amount = convert(Int64, floor(fraction * no_triplets))

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

function scale(data::Array{Float64,1}, X::Array{Float64,1})
    # We solve the scaling problem by min || aX - data - b||^2,
    # where (a,b) are the scale and offset parameters
    @assert size(data) == size(X)

    a, b = [X -ones(size(X))]\data

    return a*X - b, norm(a*X - b - data)^2
end

function scale!(data::Array{Float64,1}, X::Array{Float64,1})
    # We solve the scaling problem by min || aX - data - b||^2,
    # where (a,b) are the scale and offset parameters
    @assert size(data) == size(X)

    n::Int64 = size(X,1)
    a, b = [X -ones(size(X))]\Y

    X = a*X - b
end

function procrustes(X::Array{Float64,2}, Y::Array{Float64,2}; doReflection=true)
    #PROCRUSTES Procrustes Analysis
    #   D = PROCRUSTES(X, Y) determines a linear transformation (translation,
    #   reflection, orthogonal rotation, and scaling) of the points in the
    #   matrix Y to best conform them to the points in the matrix X.  The
    #   "goodness-of-fit" criterion is the sum of squared errors.  PROCRUSTES
    #   returns the minimized value of this dissimilarity measure in D.  D is
    #   standardized by a measure of the scale of X, given by
    #
    #      sum(sum((X - repmat(mean(X,1), size(X,1), 1)).^2, 1))
    #
    #   i.e., the sum of squared elements of a centered version of X.  However,
    #   if X comprises repetitions of the same point, the sum of squared errors
    #   is not standardized.
    #
    #   X and Y are assumed to have the same number of points (rows), and
    #   PROCRUSTES matches the i'th point in Y to the i'th point in X.  Points
    #   in Y can have smaller dimension (number of columns) than those in X.
    #   In this case, PROCRUSTES adds columns of zeros to Y as necessary.
    #
    #   [D, Z] = PROCRUSTES(X, Y) also returns the transformed Y values.
    #
    #   Examples:
    #
    #      # Create some random points in two dimensions
    #      n = 10;
    #      X = normrnd(0, 1, [n 2]);
    #
    #      # Those same points, rotated, scaled, translated, plus some noise
    #      S = [0.5 -sqrt(3)/2; sqrt(3)/2 0.5]; # rotate 60 degrees
    #      Y = normrnd(0.5*X*S + 2, 0.05, n, 2);
    #
    #      # Conform Y to X, plot original X and Y, and transformed Y
    #      [d, Z, tr] = procrustes(X,Y);
    #      plot(X(:,1),X(:,2),'rx', Y(:,1),Y(:,2),'b.', Z(:,1),Z(:,2),'bx');
    #
    #   See also FACTORAN, CMDSCALE.

    #   References:
    #     [1] Seber, G.A.F., Multivariate Observations, Wiley, New York, 1984.
    #     [2] Gower, J.C. and Dijskterhuis, G.B., Procrustes Problems, Oxford
    #         Statistical Science Series, Vol 30. Oxford University Press, 2004.
    #     [3] Bulfinch, T., The Age of Fable; or, Stories of Gods and Heroes,
    #         Sanborn, Carter, and Bazin, Boston, 1855.

    n, m   = size(X)
    ny, my = size(Y)

    if ny != n
        error("procrustes: Input size mismatch")
    elseif my > m
        error("procrustes: Too many columns")
    end

    # Center at the origin.
    muX = mean(X,1)
    muY = mean(Y,1)
    X0 = X - repmat(muX, n, 1)
    Y0 = Y - repmat(muY, n, 1)

    ssqX = sum(X0.^2,1)
    ssqY = sum(Y0.^2,1)
    constX = all(ssqX .<= abs.(eps(eltype(X))*n*muX).^2)
    constY = all(ssqY .<= abs.(eps(eltype(X))*n*muY).^2)
    ssqX = sum(ssqX)
    ssqY = sum(ssqY)

    if !constX && !constY
        # The "centered" Frobenius norm.
        normX = sqrt(ssqX) # == sqrt(trace(X0*X0'))
        normY = sqrt(ssqY) # == sqrt(trace(Y0*Y0'))

        # Scale to equal (unit) norm.
        X0 = X0 / normX
        Y0 = Y0 / normY

        # Make sure they're in the same dimension space.
        if my < m
            Y0 = [Y0 zeros(Float64, n, m-my)]
        end

        # The optimum rotation matrix of Y.
        A = X0' * Y0
        L, D, M = svd(A)
        T = M * L'
        
        haveReflection = det(T) < 0
        if doReflection != haveReflection
            # ... then either force a reflection, or undo one.
            M[:,end] = -M[:,end]
            D[end,end] = -D[end,end]
            T = M * L'
        end
        
        # The minimized unstandardized distance D(X0,b*Y0*T) is
        # ||X0||^2 + b^2*||Y0||^2 - 2*b*trace(T*X0'*Y0)
        traceTA = sum(D) # == trace(sqrtm(A'*A)) when doReflection is 'best'
        
        
        # The optimum scaling of Y.
        b = traceTA * normX / normY
        
        # The standardized distance between X and b*Y*T+c.
        d = 1 - traceTA.^2

        Z = normX*traceTA * Y0 * T + repmat(muX, n, 1)
        
    # The degenerate cases: X all the same, and Y all the same.
    elseif constX
        d = 0
        Z = repmat(muX, n, 1)
    else # !constX & constY
        d = 1
        Z = repmat(muX, n, 1)
    end

    return d, Z
end