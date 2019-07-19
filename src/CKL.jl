@def check_ckl_params begin
    if !haskey(params, :μ)
        throw(ArgumentError("params has no key :μ"))
    elseif params[:μ] <= 0
        throw(ArgumentError("Parameter :μ must be > 1"))
    end
end

struct CKL <: TripletEmbedding
    @add_embedding_fields

    function CKL(
        triplets::Array{Int64,2},
        dimensions::Int64,
        params::Dict{Symbol,Real},
        X::Embedding)

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)

        @check_embedding_conditions
        @check_ckl_params

        new(triplets, dimensions, params, X, no_triplets, no_items)
    end

    """
    Constructor that initializes with a random (normal)
    embedding.
    """
    function CKL(
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
        @check_ckl_params

        new(triplets, dimensions, params, X, no_triplets, no_items)
    end

    """
    Constructor that takes an initial condition as an Embedding.
    """
    function CKL(
        triplets::Array{Int64,2},
        params::Dict{Symbol,Real},
        X::Embedding)

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)
        dimensions = size(X,2)

        @check_embedding_conditions
        @check_ckl_params

        new(triplets, dimensions, params, X, no_triplets, no_items)
    end

    """
    Constructor that takes an initial condition as an Matrix.
    """
    function CKL(
        triplets::Array{Int64,2},
        params::Dict{Symbol,Real},
        X::Matrix{Float64})

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)
        dimensions = size(X,2)
        X = Embedding(X)

        @check_embedding_conditions
        @check_ckl_params

        new(triplets, dimensions, params, X, no_triplets, no_items)
    end
end

function kernel(te::CKL, D::Array{Float64,2})
    
    nom = zeros(no_triplets(te))
    den = zeros(no_triplets(te))

    for t in 1:no_triplets(te)
        @inbounds i = triplets(te)[t,1]
        @inbounds j = triplets(te)[t,2]
        @inbounds k = triplets(te)[t,3]        

        @inbounds nom[t] = max(te.params[:μ] + D[i,k], eps())
        @inbounds den[t] = max(te.params[:μ] + nom[t] + D[i,j], eps())
    end
    
    return Dict(:nom => nom, :den => den)
end

function gradient(te::CKL)

    P::Float64 = 0.0
    C::Float64 = 0.0

    D = distances(te)
    K = kernel(te, D)

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

function gradient_kernel(te::CKL,
                        K::Dict{Symbol, Array{Float64,1}},
                        ∇C::Array{Float64,2},
                        triplets_range::UnitRange{Int64})

    C::Float64 = 0.0
    
    for t in triplets_range
        @inbounds i = triplets(te)[t,1]
        @inbounds j = triplets(te)[t,2]
        @inbounds k = triplets(te)[t,3]

        # Compute log probability for each triplet
        C += -(log(K[:nom][t]) - log(K[:den][t]))

        for d in 1:dimensions(te)
            # Calculate the gradient of *this triplet* on its points.
            @inbounds dx_j = X(te)[i,d] - X(te)[j,d]
            @inbounds dx_k = X(te)[i,d] - X(te)[k,d]

            @inbounds ∇C[i, d] +=   2 / K[:nom][t] * dx_k - 2 / K[:den][t] * (dx_j + dx_k)
            @inbounds ∇C[j, d] +=                           2 / K[:den][t] *  dx_j
            @inbounds ∇C[k, d] += - 2 / K[:nom][t] * dx_k + 2 / K[:den][t] *  dx_k
        end
    end
    return C
end