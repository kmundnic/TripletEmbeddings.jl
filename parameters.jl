include("Triplets.jl")

using JLD
using MAT

# Set seed
random_seed = 4
srand(random_seed)

# Load data and slice to have 1 sample/second
path = "./data/gt_objective.csv"
data = readdlm(path)
data = data[1:30:end] # slice the data to have 1 sample/second
data = reshape(data, (length(data), 1))

# Model parameters
λ = 0.0   # L2 regularization parameter
α = 2:20  # Degrees of freedom of the t-Student
fraction   = 5 * logspace(-4,-2,10) # Percentage of triplets used to calculate the embedding ∈ [0,1]
# fraction   = 0.05:0.1:0.55 # Percentage of triplets used to calculate the embedding ∈ [0,1]
dimensions = 1             # dimensions of the embedding

# Probability parameters
# μ = 0.55:0.05:0.95
μ = 0.80:0.05:0.95
σ = 0.01
n = size(data)[1]

# Projected t-STE?
projected = false

# Repetitions
repetitions = 30 # To take mean values and compare

mse                = zeros(Float64, length(μ), length(α), length(fraction), repetitions)
triplet_violations = zeros(Float64, length(μ), length(α), length(fraction), repetitions)
iterations         = zeros(Float64, length(μ), length(α), length(fraction), repetitions)


for i in eachindex(μ)
    # Generate success probabilities (in the paper, these are the μ values)
    probability_success = μ[i] + σ*randn(n,n,n) # Probability of success
    probability_success[probability_success .> 1] = 1 # Check whether any values are > 1

    # Generate triplets
    triplets = Triplets.generate_triplets(data, probability_success)
    
    @time for j in eachindex(α), k in eachindex(fraction), l in 1:repetitions
            println("μ = $(μ[i]) | α = $(α[j]) | Percentage = $(fraction[k]) | Repetition = $l")

            @time X, triplet_violations[i,j,k,l], iterations[i,j,k,l] = 
                Triplets.calculate_embedding(triplets, fraction[k]; d=dimensions, α=α[j], λ=λ, project=projected)

            X_scaled, mse[i,j,k,l] = Triplets.scale(data, X)
        
            filename = "parameter_grid_search.jld"
            README = string("Grid search μ = $(μ[i]), 
                                         α = $(α[j]), 
                                         fraction = $(fraction[k]), 
                                         repetitions = $repetitions,
                                          using t-STE on ", Dates.format(Dates.now(), "yyyy-mm-dd_HH.MM.SS"))
            
            print("Saving...")
            save(filename, "MSE", mse,
                           "triplet_violations", triplet_violations,
                           "iterations", iterations,
                           "repetitions", repetitions,
                           "README", README)
            println("saved file")
    end
end

# For figure generation/exploration
README = string("Grid search μ = $(μ), α = $(α), fraction = $(fraction), repetitions = $repetitions using t-STE on ", 
            Dates.format(Dates.now(), "yyyy-mm-dd_HH.MM.SS"))
file = matopen("./results/results.mat","w")
write(file, "mse", mse)
write(file, "triplet_violations", triplet_violations)
write(file, "projected", projected)
write(file, "repetitions", repetitions)
write(file, "README", README);
close(file);