function compute(te::TripletEmbedding;
                    verbose::Bool = true,
                    max_iter::Int64 = 1000,
                    debug::Bool = false)
    println("Computing embedding of type $(typeof(te))")

    @assert max_iter >= 10

    C::Float64 = Inf                                  # Cost
    ∇C = zeros(Float64, no_items(te), dimensions(te)) # Gradient

    tolerance::Float64 = 1e-7 # convergence tolerance
    η::Float64 = 1.0          # learning rate
    best_C::Float64 = Inf     # best error obtained so far
    best_X = X(te)            # best embedding found so far
    
    if debug
        iteration_Xs = zeros(Float64, no_items(te), dimensions(te), max_iter)
    end

    # Perform main iterations
    no_iterations::Int64 = 0
    no_increments::Int64 = 0
    percent_violations::Float64 = 0.0
    violations = zeros(Bool, no_items(te), )

    while no_iterations < max_iter && no_increments < 5
        no_iterations += 1        
        old_C = C

        # Calculate gradient descent and cost
        C, ∇C = gradient(te)

        # Update the embedding according to the gradient
        te.X.X = X(te) - (η / no_triplets(te) * no_items(te)) * ∇C

        if C < best_C
            best_C = C
            best_X = X(te)
        end

        # Save each iteration if indicated
        if debug
            iteration_Xs[:,:,no_iterations] = X(te)
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
        _, percent_violations = triplet_violations(te, no_triplets=no_triplets(te), no_items=no_items(te), dimensions=te(dimensions))
    end

    if debug
        return iteration_Xs[:,:,1:no_iterations], percent_violations
    else
        te.X.X = best_X
        return percent_violations
    end
end

function partition_work(no_triplets::Int64, nthreads::Int64)
    
    ls = range(1, stop=no_triplets, length=nthreads+1)
    
    map(1:nthreads) do i
        a = round(Int64, ls[i])
        if i > 1
            a += 1
        end
        b = round(Int64, ls[i+1])
        a:b
    end
end