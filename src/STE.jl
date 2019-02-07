@def check_ste_params begin
    if !haskey(params, :σ)
        throw(ArgumentError("params has no key :σ"))
    elseif params[:σ] <= 0
        throw(ArgumentError("Parameter :σ must be > 1"))
    end
end

struct STE <: TripletEmbedding
	@add_embedding_fields

	function STE(
		triplets::Array{Int64,2},
		dimensions::Int64,
		params::Dict{Symbol,Real},
		X::Embedding)

		no_triplets::Int64 = size(triplets,1)
		no_items::Int64 = maximum(triplets)

		@check_embedding_conditions
		@check_ste_params

        new(triplets, dimensions, params, X, no_triplets, no_items)
    end

    """
    Constructor that initializes with a random (normal)
    embedding.
    """
    function STE(
        triplets::Array{Int64,2},
        dimensions::Int64,
        params::Dict{Symbol,Real})

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)

        if dimensions > 0
            X = Embedding(0.0001*randn(maximum(triplets), dimensions))
        else
            throw(ArgumentError("Dimensions must be >= 1"))
        end

        @check_embedding_conditions
        @check_ste_params

        new(triplets, dimensions, params, X, no_triplets, no_items)
    end

    """
    Constructor that takes an initial condition.
    """
    function STE(
        triplets::Array{Int64,2},
        params::Dict{Symbol,Real},
        X::Embedding)

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)
        dimensions = size(X,2)

        @check_embedding_conditions
        @check_ste_params

        new(triplets, dimensions, params, X, no_triplets, no_items)
    end
end

function σ(te::STE)
    return te.params[:σ]
end

function gradient(te::STE)

	P::Float64 = 0.0
	C::Float64 = 0.0
    constant = 1/σ(te)^2

	sum_X = zeros(Float64, no_items(te), )
	K = zeros(Float64, no_items(te), no_items(te))

	# Compute normal density kernel for each point
	# i,j range over points; k ranges over dimensions
	for k in 1:dimensions(te), i in 1:no_items(te)
		@inbounds sum_X[i] += X(te)[i,k] * X(te)[i,k]
	end

	for j in 1:no_items(te), i in 1:no_items(te)
		@inbounds K[i,j] = sum_X[i] + sum_X[j]
		for k in 1:dimensions(te)
			@inbounds K[i,j] += -2 * X(te)[i,k] * X(te)[j,k]
		end
		@inbounds K[i,j] = exp( -constant * K[i,j] / 2)
	end

	nthreads::Int64 = Threads.nthreads()
	work_ranges = partition_work(no_triplets(te), nthreads)

	# Define costs and gradient vectors for each thread
	Cs = zeros(Float64, nthreads, )
	∇Cs = [zeros(Float64, no_items(te), dimensions(te)) for _ = 1:nthreads]

	Threads.@threads for tid in 1:nthreads
		Cs[tid] = gradient_kernel(te, K, ∇Cs[tid], constant, work_ranges[tid])
	end

	C += sum(Cs)
    ∇C = ∇Cs[1]

	for i in 2:length(∇Cs)
		∇C .+= ∇Cs[i]
	end

    for i in 1:dimensions(te), n in 1:no_items(te)
		@inbounds ∇C[n,i] = - ∇C[n, i]
	end

	return C, ∇C
end

function gradient_kernel(te::STE,
                        K::Array{Float64,2},
                        ∇C::Array{Float64,2},
                        constant::Float64,
                        triplets_range::UnitRange{Int64})

    C::Float64 = 0.0
    
    for t in triplets_range
        @inbounds triplets_A = triplets(te)[t, 1]
        @inbounds triplets_B = triplets(te)[t, 2]
        @inbounds triplets_C = triplets(te)[t, 3]

		# Compute log probability for each triplet
        # This is exactly p_{ijk}, which is the equation in the lower-left of page 3 of the t-STE paper.
        @inbounds P = K[triplets_A, triplets_B] / (K[triplets_A, triplets_B] + K[triplets_A, triplets_C])
        C += -log(P)

        for i in 1:dimensions(te)
            # Calculate the gradient of *this triplet* on its points.
            @inbounds A_to_B = (1 - P) * (X(te)[triplets_A, i] - X(te)[triplets_B, i])
            @inbounds A_to_C = (1 - P) * (X(te)[triplets_A, i] - X(te)[triplets_C, i])

            @inbounds ∇C[triplets_A, i] += - constant * (A_to_B - A_to_C)
            @inbounds ∇C[triplets_B, i] +=   constant *  A_to_B
            @inbounds ∇C[triplets_C, i] += - constant *  A_to_C
        end
    end
    return C
end