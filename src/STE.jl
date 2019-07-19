@def check_ste_params begin
    if !haskey(params, :σ)
        throw(ArgumentError("params has no key :σ"))
    elseif params[:σ] <= 0
        throw(ArgumentError("Parameter :σ must be > 1"))
    end
end

struct STE <: TripletEmbedding
    @add_embedding_fields
    constant::Float64

    function STE(
        triplets::Array{Int64,2},
        dimensions::Int64,
        params::Dict{Symbol,Real},
        X::Embedding)

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)
        constant = 1/params[:σ]^2

        @check_embedding_conditions
        @check_ste_params

        new(triplets, dimensions, params, X, no_triplets, no_items, constant)
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
        constant = 1/params[:σ]^2

        if dimensions > 0
            X = Embedding(0.0001*randn(maximum(triplets), dimensions))
        else
            throw(ArgumentError("Dimensions must be >= 1"))
        end

        @check_embedding_conditions
        @check_ste_params

        new(triplets, dimensions, params, X, no_triplets, no_items, constant)
    end

    """
    Constructor that takes an initial condition as Embedding.
    """
    function STE(
        triplets::Array{Int64,2},
        params::Dict{Symbol,Real},
        X::Embedding)

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)
        dimensions = size(X.X,2)
        constant = 1/params[:σ]^2

        @check_embedding_conditions
        @check_ste_params

        new(triplets, dimensions, params, X, no_triplets, no_items, constant)
    end

    """
    Constructor that takes an initial condition as an Array.
    
    Vectors are not allowed as initial conditions, please use an Array{Float,2} 
    of size = (n,1).
    """
    function STE(
        triplets::Array{Int64,2},
        params::Dict{Symbol,Real},
        X::Matrix{Float64})

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)
        dimensions::Int64 = size(X,2)
        constant = 1/params[:σ]^2
        X = Embedding(X)

        @check_embedding_conditions
        @check_ste_params

        new(triplets, dimensions, params, X, no_triplets, no_items, constant)

    end

end

function σ(te::STE)
    return te.params[:σ]
end

function kernel(te::STE)
    K = pairwise(SqEuclidean(), X(te), dims=1)

    for j in 1:no_items(te), i in 1:no_items(te)
        @inbounds K[i,j] = exp( -te.constant * K[i,j] / 2)
    end
    return K
end

function gradient(te::STE)

    P::Float64 = 0.0
    C::Float64 = 0.0

    K = kernel(te)

    nthreads::Int64 = Threads.nthreads()
    work_ranges = partition_work(no_triplets(te), nthreads)

    # Define costs and gradient vectors for each thread
    Cs = zeros(Float64, nthreads, )
    ∇Cs = [zeros(Float64, no_items(te), dimensions(te)) for _ = 1:nthreads]

    Threads.@threads for tid in 1:nthreads
        Cs[tid] = gradient_kernel(te, K, ∇Cs[tid], work_ranges[tid])
    end

    C += sum(Cs)
    ∇C = ∇Cs[1]

    for i in 2:length(∇Cs)
        ∇C .+= ∇Cs[i]
    end

    return C, -∇C
end

function gradient_kernel(te::STE,
                        K::Array{Float64,2},
                        ∇C::Array{Float64,2},
                        triplets_range::UnitRange{Int64})

    C::Float64 = 0.0
    
    for t in triplets_range
        @inbounds i = triplets(te)[t,1]
        @inbounds j = triplets(te)[t,2]
        @inbounds k = triplets(te)[t,3]

        # Compute log probability for each triplet
        # This is exactly p_{ijk}, which is the equation in the lower-left of page 3 of the STE paper.
        @inbounds P = K[i,j] / (K[i,j] + K[i,k])
        C += -log(P)

        for d in 1:dimensions(te)
            # Calculate the gradient of *this triplet* on its points.
            @inbounds dx_j = (1 - P) * (X(te)[i,d] - X(te)[j,d])
            @inbounds dx_k = (1 - P) * (X(te)[i,d] - X(te)[k,d])

            @inbounds ∇C[i, d] += - te.constant * (dx_j - dx_k)
            @inbounds ∇C[j, d] +=   te.constant *  dx_j
            @inbounds ∇C[k, d] += - te.constant *  dx_k
        end
    end
    return C
end