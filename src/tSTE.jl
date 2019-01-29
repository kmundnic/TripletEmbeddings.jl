function compute(te::TripletEmbedding;
              use_log::Bool = true,
              verbose::Bool = true,
              max_iter::Int64 = 1000,
              debug::Bool = false)

    @assert max_iter >= 10

    C::Float64 = Inf                                # Cost
    ∇C = zeros(Float64, te.no_items, te.dimensions) # Gradient

    tolerance::Float64 = 1e-7 # convergence tolerance
    η::Float64 = 2.0          # learning rate
    best_C::Float64 = Inf     # best error obtained so far
    best_X = te.X.X           # best embedding found so far
    
    if debug
        iteration_Xs = zeros(Float64, te.no_items, te.dimensions, max_iter)
    end

    # Perform main iterations
    no_iterations::Int64 = 0
    no_increments::Int64 = 0
    percent_violations::Float64 = 0.0
    violations = zeros(Bool, te.no_items, )

    while no_iterations < max_iter && no_increments < 5
        no_iterations += 1
        old_C = C

        # Calculate gradient descent and cost
        C, ∇C = ∇tSTE(te)

        te.X.X = te.X.X - (η / te.no_triplets * te.no_items) * ∇C

        if C < best_C
            best_C = C
            best_X = te.X.X
        end

        # Save each iteration if indicated
        if debug
            iteration_Xs[:,:,no_iterations] = te.X.X
        end

        # Update learning rate
        if old_C > C + tolerance
            no_increments = 0
            η *= 1.01
        else
            no_increments += 1
            η *= 0.5
        end

        # Print out progress
        if verbose && (no_iterations % 10 == 0)
            violations, percent_violations = triplet_violations(te)
            @printf("iter # = %d, cost = %.2f, violations = %.2f %%\n", no_iterations, C, 100*percent_violations)
        end
    end

    if !verbose
        _, percent_violations = triplet_violations(te, no_triplets=te.no_triplets, no_items=te.no_items, dimensions=te.dimensions)
    end

    if debug
        return iteration_Xs[:,:,1:no_iterations], percent_violations
    else
        te.X.X = best_X
        return te, percent_violations
    end
end

function ∇tSTE(te::tSTE)

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
        Cs[tid] = tSTE_thread_kernel(work_ranges[tid], te, K, Q, ∇Cs[tid], constant,)
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

function tSTE_thread_kernel(triplets_range::UnitRange{Int64},
                            te::tSTE,
                            K::Array{Float64,2},
                            Q::Array{Float64,2},
                            ∇C::Array{Float64,2},
                            constant::Float64)
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

# function ∇tSTE(X::Array{Float64,1}, 
#                te.no_items::Int64,
#                te.dimensions::Int64,
#                no_triplets::Int64,
#                triplets::Array{Int64,2},
#                λ::Real,
#                α::Real)::Tuple{Float64,Array{Float64,2}}

#     @assert te.dimensions == 1

#     C, ∇C = ∇tSTE(reshape(X, size(X,1), 1),
#                   te.no_items,
#                   te.dimensions,
#                   no_triplets,
#                   triplets,
#                   λ,
#                   α)

#     return C, ∇C
# end