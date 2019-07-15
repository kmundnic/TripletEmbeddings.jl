using Random
using Plots; plotlyjs()
using BenchmarkTools
include("Embeddings.jl")

Random.seed!(4)

# Load data dummy data
data = Embeddings.load_data()

# Generate triplets
triplets = Embeddings.label(data)

# Calculate embedding from triplets
dimensions = 1
params = Dict{Symbol,Real}()
params[:α] = 30 # Degrees of freedom of the t-Student

te = Embeddings.tSTE(triplets, dimensions, params)

# # Compute embeddings
# @time violations = Embeddings.fit!(te; max_iter=50)

# X = Embeddings.scale(data, dropdims(te.X.X, dims=2))
# plot(data, label="Data")
# plot!(X, label="tSTE")
# # plot!(data)

# dimensions = 1
# params = Dict{Symbol,Real}()
# params[:σ] = 1/sqrt(2) # Normal distribution variance

# te = Embeddings.STE(triplets, dimensions, params)

# # Compute embeddings
# @time violations = Embeddings.fit!(te; max_iter=50)

# X = Embeddings.scale(data, dropdims(te.X.X, dims=2))
# plot!(X, label="STE")

# dimensions = 1
# te = Embeddings.SmoothHingeGNMDS(triplets, dimensions)
@time violations = Embeddings.fit!(te; max_iter=50)
# plot(Embeddings.X(te))