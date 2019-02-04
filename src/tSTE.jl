@def check_tste_params begin
    if !haskey(params, :α)
        throw(ArgumentError("params has no key :α"))
    elseif params[:α] <= 1
        throw(ArgumentError("Parameter α must be > 1"))
    end

    if !haskey(params, :λ)
        throw(ArgumentError("params has no key :λ"))
    elseif params[:λ] < 0
        throw(ArgumentError("Regularizer λ must be >= 0"))
    end
end

struct tSTE <: TripletEmbedding
    @add_embedding_fields

    function tSTE(
        triplets::Array{Int64,2},
        dimensions::Int64,
        params::Dict{Symbol,Real},
        X::Embedding)
        
        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)

        @check_embedding_conditions
        @check_tste_params
        
        new(triplets, dimensions, params, X, no_triplets, no_items)
    end

    """
    Constructor that initializes with a random (normal)
    embedding.
    """
    function tSTE(
        triplets::Array{Int64,2},
        dimensions::Int64,
        params::Dict{Symbol,Real})

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)

        if dimensions > 0
            X = Embedding(randn(maximum(triplets), dimensions))
        else
            throw(ArgumentError("Dimensions must be >= 1"))
        end

        @check_embedding_conditions
        @check_tste_params

        new(triplets, dimensions, params, X, no_triplets, no_items)
    end

    """
    Constructor that takes an initial condition.
    """
    function tSTE(
        triplets::Array{Int64,2},
        params::Dict{Symbol,Real},
        X::Embedding)

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)
        dimensions = size(X,2)

        @check_embedding_conditions
        @check_tste_params

        new(triplets, dimensions, params, X, no_triplets, no_items)
    end
end

function α(te::tSTE)
    return te.params[:α]
end

function λ(te::tSTE)
    return te.params[:λ]
end

function gradient(te::tSTE)

    P::Float64 = 0.0
    C::Float64 = 0.0 + λ(te) * sum(X(te).^2) # Initialize cost including l2 regularization cost

    sum_X = zeros(Float64, no_items(te), )
    K = zeros(Float64, no_items(te), no_items(te))
    Q = zeros(Float64, no_items(te), no_items(te))

    A_to_B::Float64 = 0.0
    A_to_C::Float64 = 0.0
    constant::Float64 = (α(te) + 1) / α(te)

    triplets_A::Int64 = 0
    triplets_B::Int64 = 0
    triplets_C::Int64 = 0

    # Compute t-Student kernel for each point
    # i,j range over points; k ranges over dims
    for k in 1:dimensions(te), i in 1:no_items(te)
        @inbounds sum_X[i] += X(te)[i, k] * X(te)[i, k] # Squared norm
    end

    for j in 1:no_items(te), i in 1:no_items(te)
        @inbounds K[i,j] = sum_X[i] + sum_X[j]
        for k in 1:dimensions(te)
            # K[i,j] = ((sqdist(i,j)/α + 1)) ^ (-(α+1)/2),
            # which is exactly the numerator of p_{i,j} in the lower right of
            # t-STE paper page 3.
            # The proof follows because sqdist(a,b) = (a-b)(a-b) = a^2+b^2-2ab
            @inbounds K[i,j] += -2 * X(te)[i,k] * X(te)[j,k]
        end
        @inbounds Q[i,j] = (1 + K[i,j] / α(te)) ^ -1
        @inbounds K[i,j] = (1 + K[i,j] / α(te)) ^ ((α(te) + 1) / -2)
    end

    # Compute probability (or log-prob) for each triplet
    nthreads::Int64 = Threads.nthreads()
    work_ranges = partition_work(no_triplets(te), nthreads)

    # Define costs and gradients for each thread
    Cs = zeros(Float64, nthreads, )
    ∇Cs = [zeros(Float64, no_items(te), dimensions(te)) for _ = 1:nthreads]
    
    Threads.@threads for tid in 1:nthreads
        Cs[tid] = thread_kernel(te, K, Q, ∇Cs[tid], constant, work_ranges[tid])
    end

    C += sum(Cs)
    ∇C = ∇Cs[1]
    
    for i in 2:length(∇Cs)
        ∇C .+= ∇Cs[i]
    end

    for i in 1:dimensions(te), n in 1:no_items(te)
        # The 2λX is for regularization: derivative of L2 norm
        @inbounds ∇C[n,i] = - ∇C[n, i] + 2*λ(te) * X(te)[n, i]
    end

    return C, ∇C
end

function thread_kernel(te::tSTE,
                        K::Array{Float64,2},
                        Q::Array{Float64,2},
                        ∇C::Array{Float64,2},
                        constant::Float64,
                        triplets_range::UnitRange{Int64})

    C::Float64 = 0.0
    
    for t in triplets_range
        @inbounds triplets_A = te.triplets[t, 1]
        @inbounds triplets_B = te.triplets[t, 2]
        @inbounds triplets_C = te.triplets[t, 3]

        # This is exactly p_{ijk}, which is the equation in the lower-right of page 3 of the t-STE paper.
        @inbounds P = K[triplets_A, triplets_B] / (K[triplets_A, triplets_B] + K[triplets_A, triplets_C])
        C += -log(P)

        for i in 1:dimensions(te)
            # Calculate the gradient of *this triplet* on its points.
            @inbounds A_to_B = ((1 - P) * Q[triplets_A, triplets_B] * (X(te)[triplets_A, i] - X(te)[triplets_B, i]))
            @inbounds A_to_C = ((1 - P) * Q[triplets_A, triplets_C] * (X(te)[triplets_A, i] - X(te)[triplets_C, i]))

            @inbounds ∇C[triplets_A, i] +=   constant * (A_to_C - A_to_B)
            @inbounds ∇C[triplets_B, i] +=   constant *  A_to_B
            @inbounds ∇C[triplets_C, i] += - constant *  A_to_C
        end
    end
    return C
end

# function gradient(X::Array{Float64,1}, 
#                no_items::Int64,
#                dimensions::Int64,
#                no_triplets::Int64,
#                triplets::Array{Int64,2},
#                λ::Real,
#                α::Real)::Tuple{Float64,Array{Float64,2}}

#     @assert dimensions(te) == 1

#     C, ∇C = gradient(reshape(X, size(X,1), 1),
#                   no_items,
#                   dimensions,
#                   no_triplets,
#                   triplets,
#                   λ,
#                   α)

#     return C, ∇C
# end