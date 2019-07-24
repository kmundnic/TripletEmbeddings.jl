# include("ndgrid.jl")

function load_data()::Vector{Float64}
    return rand(300, )
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

function subset(triplets::Array{Int64,2}, fraction::Real)
    @assert !isempty(triplets)
    
    amount = floor(Int64, fraction * size(triplets,1))
    
    triplets = triplets[shuffle(1:end),:]
    return triplets[1:amount, :]

end

function subset!(triplets::Array{Int64,2}, number_of_triplets::Real)
    @assert !isempty(triplets)
    @assert number_of_triplets > 2.0
    
    S = triplets[1:floor(Int64, number_of_triplets), :]
    triplets = triplets[ceil(Int64, number_of_triplets):end, :]

    return S
end

function distances(te::TripletEmbedding)
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

function distances(X::Array{Float64,1}, no_items::Int64)::Array{Float64,2}
    return distances(reshape(X, size(X,1), 1), no_items)
end

function distances(X::Array{Float64,2}, no_items::Int64)::Array{Float64,2}
    D = zeros(Float64, no_items, no_items)

    for j = 1:no_items, i = j:no_items
        @inbounds D[i,j] = norm(X[i,:] - X[j,:])
    end

    return D + D'
end

function distances(X::Array{Int64,1}; no_items::Int64=length(X))::Array{Int64,2}
    D = zeros(Int64, no_items, no_items)

    for j = 1:no_items, i = j:no_items
        @inbounds D[i,j] = abs(X[i] - X[j])
    end

    return D + D'
end

function triplet_violations(te::TripletEmbedding)
    return triplet_violations(X(te),
        triplets(te);
        no_triplets = no_triplets(te),
        no_items = no_items(te),
        no_dims = dimensions(te))
end

function triplet_violations(X::Array{Float64,2},
                            triplets::Array{Int64,2};
                            no_triplets = size(triplets,1)::Int64,
                            no_items = maximum(triplets)::Int64,
                            no_dims = size(X,2)::Int64)
    "This function is more exact in floating point operations than 
    triplet_violations, but it is a bit slower and uses double the memory"

    D = zeros(Float64, no_items, no_items)

    for j = 1:no_items, i = j:no_items
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

function correct_triplets(te::TripletEmbedding)
    return correct_triplets(X(te), triplets(te))
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

function scale(data::Array{Float64,1}, te::TripletEmbedding; MSE::Bool=false)
    if te.dimensions == 1
        return scale(data, dropdims(X(te); dims=2); MSE=MSE)
    else
        error("Cannot scale an embedding with dimensions > 1. Use procrustes instead")
    end
end

function scale!(data::Array{Float64,1}, te::TripletEmbedding)
    if te.dimensions == 1
        te.X.X[:,1] = scale(data, dropdims(te.X.X; dims=2))
    else
        error("Cannot scale an embedding with dimensions > 1. Use procrustes instead")
    end
end

function scale(data::Vector{T}, X::Array{Float64,1}; MSE::Bool=false) where T <: Real
    # We solve the scaling problem by min || aX - data - b||^2,
    # where (a,b) are the scale and offset parameters
    @assert size(data) == size(X)

    a, b = [X -ones(size(X))]\data

    if MSE
        return a*X .- b, norm(a*X .- b .- data)^2
    else
        return a*X .- b
    end
end