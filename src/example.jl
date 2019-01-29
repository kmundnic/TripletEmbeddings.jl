using Random
using Plots; pyplot()
include("Embeddings.jl")

Random.seed!(4)

# Load data and slice to have 1 sample/second
data = Embeddings.load_data()

# Generate triplets
triplets = Embeddings.label(data)

# Calculate embedding from triplets
dimensions = 1
params = Dict{Symbol,Real}()
params[:λ] = 0.0 # L2 regularization parameter
params[:α] = 30 # Degrees of freedom of the t-Student

te = Embeddings.tSTE(triplets, dimensions, params)

# Compute embeddings
@time bla, _ = Embeddings.compute(te; max_iter=50)

# X = Embeddings.scale(data, dropdims(X, dims=2))
# plot(X)
# plot!(data)
