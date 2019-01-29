struct tSTE <: TripletEmbedding
    triplets::Array{Int64,2}
    dimensions::Int64
    params::Dict{Symbol,Real}
    X::Embedding
    no_triplets::Int64
    no_items::Int64


    function tSTE(
        triplets::Array{Int64,2},
        dimensions::Int64,
        params::Dict{Symbol,Real},
        X::Embedding)
        
        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)
        
        if size(triplets, 2) != 3
            throw(ArgumentError("Triplets do not have three values"))
        end

        if sort(unique(triplets)) != 1:n
            throw(ArgumentError("Triplet values must have all elements in the range 1:no_items"))
        end         

        if dimensions < 1
            throw(ArgumentError("Dimensions must be >= 1"))
        end
    
        if no_items != size(X.X, 1)
            throw(ArgumentError("Number of elements in triplets does not match the embedding dimension"))
        end

        if size(X.X, 2) != dimensions
            throw(ArgumentError("dimensions and embedding dimensions do not match"))
        end

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

        new(triplets, dimensions, params, X, no_triplets, no_items)
    end

    function tSTE(
        triplets::Array{Int64,2},
        dimensions::Int64,
        params::Dict{Symbol,Real})

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)

        X0 = Embedding(randn(maximum(triplets), dimensions))
        new(triplets, dimensions, params, X0, no_triplets, no_items)
    end

    function tSTE(
        triplets::Array{Int64,2},
        params::Dict{Symbol,Real},
        X::Embedding)

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)

        dimensions = size(X,2)
        new(triplets, dimensions, params, X, no_triplets, no_items)
    end
end

function gradient(te::tSTE)

    P::Float64 = 0.0
    C::Float64 = 0.0 + te.params[:λ] * sum(te.X.X.^2) # Initialize cost including l2 regularization cost

    sum_X = zeros(Float64, te.no_items, )
    K = zeros(Float64, te.no_items, te.no_items)
    Q = zeros(Float64, te.no_items, te.no_items)

    A_to_B::Float64 = 0.0
    A_to_C::Float64 = 0.0
    constant::Float64 = (te.params[:α] + 1) / te.params[:α]

    triplets_A::Int64 = 0
    triplets_B::Int64 = 0
    triplets_C::Int64 = 0

    # Compute t-Student kernel for each point
    # i,j range over points; k ranges over dims
    for k in 1:te.dimensions, i in 1:te.no_items
        @inbounds sum_X[i] += te.X.X[i, k] * te.X.X[i, k] # Squared norm
    end

    for j in 1:te.no_items, i in 1:te.no_items
        @inbounds K[i,j] = sum_X[i] + sum_X[j]
        for k in 1:te.dimensions
            # K[i,j] = ((sqdist(i,j)/α + 1)) ^ (-(α+1)/2),
            # which is exactly the numerator of p_{i,j} in the lower right of
            # t-STE paper page 3.
            # The proof follows because sqdist(a,b) = (a-b)(a-b) = a^2+b^2-2ab
            @inbounds K[i,j] += -2 * te.X.X[i,k] * te.X.X[j,k]
        end
        @inbounds Q[i,j] = (1 + K[i,j] / te.params[:α]) ^ -1
        @inbounds K[i,j] = (1 + K[i,j] / te.params[:α]) ^ ((te.params[:α] + 1) / -2)
    end

    # Compute probability (or log-prob) for each triplet
    nthreads::Int64 = Threads.nthreads()
    work_ranges = partition_work(te.no_triplets, nthreads)

    # Define costs and gradients for each thread
    Cs = zeros(Float64, nthreads, )
    ∇Cs = [zeros(Float64, te.no_items, te.dimensions) for _ = 1:nthreads]
    
    Threads.@threads for tid in 1:nthreads
        Cs[tid] = thread_kernel(te, K, Q, ∇Cs[tid], constant, work_ranges[tid])
    end

    C += sum(Cs)
    ∇C = ∇Cs[1]
    
    for i in 2:length(∇Cs)
        ∇C .+= ∇Cs[i]
    end

    for i in 1:te.dimensions, n in 1:te.no_items
        # The 2λX is for regularization: derivative of L2 norm
        @inbounds ∇C[n,i] = - ∇C[n, i] + 2*te.params[:λ] * te.X.X[n, i]
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

        for i in 1:te.dimensions
            # Calculate the gradient of *this triplet* on its points.
            @inbounds A_to_B = ((1 - P) * Q[triplets_A, triplets_B] * (te.X.X[triplets_A, i] - te.X.X[triplets_B, i]))
            @inbounds A_to_C = ((1 - P) * Q[triplets_A, triplets_C] * (te.X.X[triplets_A, i] - te.X.X[triplets_C, i]))

            @inbounds ∇C[triplets_A, i] +=   constant * (A_to_C - A_to_B)
            @inbounds ∇C[triplets_B, i] +=   constant *  A_to_B
            @inbounds ∇C[triplets_C, i] += - constant *  A_to_C
        end
    end
    return C
end

# function gradient(X::Array{Float64,1}, 
#                te.no_items::Int64,
#                te.dimensions::Int64,
#                no_triplets::Int64,
#                triplets::Array{Int64,2},
#                λ::Real,
#                α::Real)::Tuple{Float64,Array{Float64,2}}

#     @assert te.dimensions == 1

#     C, ∇C = gradient(reshape(X, size(X,1), 1),
#                   te.no_items,
#                   te.dimensions,
#                   no_triplets,
#                   triplets,
#                   λ,
#                   α)

#     return C, ∇C
# end