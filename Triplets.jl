module Triplets

using ToeplitzMatrices
using DataFrames
using Distances
using Convex, SCS
using PyCall; @pyimport cy_tste

# Set seed
random_seed = 4
srand(random_seed)

	function scale(data::Array{Float64,2}, X::Array{Float64,2})
		# Finding optimal scaling values
		a = Variable(1) # scaling parameter
		b = Variable(1) # offset parameter

		problem = minimize(sumsquares(a * X - data - b))
		solve!(problem, SCSSolver(verbose=false))

		X_scaled = a.value * X - b.value

		return X_scaled, problem.optval
	end

	function generate_triplets(data::Array{Float64,2}, probability_success::Array{Float64,3})
		# probability represents the probability of swapping the order of a
		# random triplet


		# We prealocate the possible total amount of triplets. Before returning,
		# we clip the array 'triplets' to the amount of nonzero elements.
		n = size(data)[1]
		triplets = zeros(Int64, 3, convert(Int64, ceil(n*(n-1)*(n-2)/2)))
		counter = 0

		distances = pairwise(Cityblock(), data'); # Compares columns

		for i = 1:n, j = 1:n, k = 1:n
            if i != j && i != k && j != k
				random_number = rand()

				if distances[i,j] < distances[i,k] && random_number .> (1 - probability_success[i,j,k])
					counter +=1
                    triplets[:,counter] = [i; j; k]
                elseif distances[i,j] > distances[i,k] && random_number .<= (1 - probability_success[i,j,k])
					counter += 1
					triplets[:,counter] = [i; j; k]
				end
		    end
		end

        return triplets[:,1:counter]
	end

	# function scaling(data, X)

	# 	min_value = data[260]
	# 	# max_value = maximum(data)
	# 	max_value = data[100] # Hard-coded value for rescaling

	# 	if X[90] > X[100] # flip X if needed
	# 	    X = -X
	# 	end

	# 	# Clip: When there are violation triplets that go beyond -100 or +100
	# 	X_copy = copy(X)
	# 	if maximum(X) > 85 || minimum(X) < -70
	# 	  X_copy = copy(X)
	# 	  X[X .< -70] = 0
	# 	  X[X .> 85] = 0
	# 	end

	# 	# To scale X into [a,b] use: x_scaled = (x-min(x))*(b-a)/(max(x)-min(x)) + a
	# 	return @data((X_copy-X[260])*(max_value-min_value)/(X[100]-X[260]) + min_value)

	# end

	function calculate_embedding(triplets::Array{Int64,2}, fraction::Real; d::Int64=1, α::Int64=2, λ::Float64=0.0, project::Bool=true, save_each_iteration::Bool=false)

		if !isempty(triplets) # Check for no triplets

			total = size(triplets)[2]
			amount = convert(Int64, floor(fraction*total))

			subset_triplets = triplets[:,rand(1:size(triplets)[2], amount)]

			@time X, triplet_violations, iterations = cy_tste.tste(PyReverseDims(subset_triplets), d, λ, α,
								   							# save_each_iteration=true,
								   							# seed=random_seed,
								   							use_log=true,
								   							project=project,
								   							window_size=5)
			X = X[2:end,:] # Why is the embedding 268x1 instead of 267x1? Need to check		
		else
			X = 0
			triplet_violations = Inf
		end

		return X, triplet_violations, iterations
		# return X, X_initial

	end

	function initial_condition(triplets, coeffs, σ=0.0001)
		X0 = σ*randn(maximum(triplets),1)

		column = squeeze(hcat(coeffs, zeros(1, length(X0) - length(coeffs))), 1)
		A = TriangularToeplitz(column, :L)

		return full(A)*X0
	end

	function medianfilter(data, window_size::Int64)
		filtered = copy!(zeros(size(data)), data)

		pre_window = floor(Int, window_size/2)

		for i = 1:pre_window
			filtered[i] = data[i]
			filtered[end+1-i] = data[end+1-i]
		end

		for i = pre_window+1:length(data) - pre_window
			filtered[i] = median(data[i-pre_window:i+pre_window])
		end

		return filtered
	end

end # module
