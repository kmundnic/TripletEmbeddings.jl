struct GNMDS <: TripletEmbedding
    loss::Symbol
    triplets::Array{Int64,2}
    dimensions::Int64
    X::Embedding
    no_triplets::Int64
    no_items::Int64

    function GNMDS(
        loss::String,
        triplets::Array{Int64,2},
        dimensions::Int64,
        X::Embedding)
        
        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)

        @check_embedding_conditions
        
        new(loss, triplets, dimensions, X, no_triplets, no_items)
    end

    """
    Constructor that initializes with a random (normal)
    embedding.
    """
    function GNMDS(
        loss::Symbol,
        triplets::Array{Int64,2},
        dimensions::Int64)

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)

        if dimensions > 0
            X = Embedding(0.0001*randn(maximum(triplets), dimensions))
        else
            throw(ArgumentError("Dimensions must be >= 1"))
        end

        @check_embedding_conditions

        new(loss, triplets, dimensions, X, no_triplets, no_items)
    end

    """
    Constructor that takes an initial condition.
    """
    function GNMDS(
        loss::Symbol,
        triplets::Array{Int64,2},
        X::Embedding)

        no_triplets::Int64 = size(triplets,1)
        no_items::Int64 = maximum(triplets)
        dimensions = size(X,2)

        @check_embedding_conditions

        new(loss, triplets, dimensions, X, no_triplets, no_items)
    end
end

function hinge_kernel(te::GNMDS, D::Array{Float64,2})

    slack = zeros(no_triplets(te))
    
    for t in 1:no_triplets(te)
        @inbounds i = triplets(te)[t,1]
        @inbounds j = triplets(te)[t,2]
        @inbounds k = triplets(te)[t,3]

        @inbounds slack[t] = max(D[i,j] + 1 - D[i,k], 0)
    end
    return slack
end

function distances(te::GNMDS)
    sum_X = zeros(Float64, no_items(te), )
    D = zeros(Float64, no_items(te), no_items(te))

    # Compute normal density kernel for each point
    # i,j range over points; k ranges over dimensions
    for k in 1:dimensions(te), i in 1:no_items(te)
        @inbounds sum_X[i] += X(te)[i,k] * X(te)[i,k]
    end

    for j in 1:no_items(te), i in 1:no_items(te)
        @inbounds D[i,j] = sum_X[i] + sum_X[j]
        for k in 1:dimensions(te)
            @inbounds D[i,j] += -2 * X(te)[i,k] * X(te)[j,k]
        end
    end
    return D
end

function gradient(te::GNMDS)

    D = distances(te)
    slack = hinge_kernel(te, D)
    C = sum(slack)

    # We use single thread here since it is faster than
    # multithreading in Julia v1.0.3
    # We leave the multithreaded version of gradient_kernel
    # for the future
    ∇C = gradient_kernel(te, slack)

    return C, ∇C

end

function gradient_kernel(te::GNMDS,
                         slack::Array{Float64,1})
    
    violations = zeros(Bool, no_triplets(te))
    @. violations = slack > 0
    ∇C = zeros(Float64, no_items(te), dimensions(te))

    for t in eachindex(violations)
        if violations[t]
            # We obtain the indices for readability
            @inbounds i = triplets(te)[t,1]
            @inbounds j = triplets(te)[t,2]
            @inbounds k = triplets(te)[t,3]

            for d in 1:dimensions(te)
                @inbounds dx_j = X(te)[i,d] - X(te)[j,d]
                @inbounds dx_k = X(te)[i,d] - X(te)[k,d]
                
                @inbounds ∇C[i,d] +=   2 * (dx_j - dx_k)
                @inbounds ∇C[j,d] += - 2 * dx_j
                @inbounds ∇C[k,d] +=   2 * dx_k
            end
        end
    end

    return ∇C
end

function gradient_kernel!(te::GNMDS,
                         violations::Array{Bool,1},
                         ∇C::Array{Float64,2},
                         triplets_range::UnitRange{Int64})
    
    for t in triplets_range
        if violations[t]
            @inbounds i = triplets(te)[t,1]
            @inbounds j = triplets(te)[t,2]
            @inbounds k = triplets(te)[t,3]

            for d in 1:dimensions(te)
                @inbounds dx_j = X(te)[i,d] - X(te)[j,d]
                @inbounds dx_k = X(te)[i,d] - X(te)[k,d]
                
                @inbounds ∇C[i,d] +=   2 * (dx_j - dx_k)
                @inbounds ∇C[j,d] += - 2 * dx_j
                @inbounds ∇C[k,d] +=   2 * dx_k
            end
        end
    end

    return ∇C
end