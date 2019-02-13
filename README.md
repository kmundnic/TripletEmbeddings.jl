# TripletEmbeddings.jl
This is a Julia 1.0 implementation of Triplet Embeddings. Currently, the following algorithms are implemented:

  - Stochastic Triplet Embeddings:
    - STE (normal kernel in probabilities)
    - tSTE (t-Student kernel in probabilities)
  - Generalized NMDS with the following loss functions:
    - Hinge loss
    - Smooth hinge loss
    - Logarithmic
    - Exponential

These implementations are based on based on Laurens Van der Maaten's implementation in [MATLAB](https://lvdmaaten.github.io/ste/Stochastic_Triplet_Embedding.html). However, this implementation uses multithreading for the calculation of the gradients, making it the fastest implementation.

# Usage
The following code is a usage example:

```
using Random
using Plots
include("src/Embeddings.jl")

Random.seed!(4)

# Load data dummy data
data = Embeddings.load_data()

# Generate triplets
triplets = Embeddings.label(data)

# Create an embedding with the triplets and compute the embedding
dimensions = 1
params = Dict{Symbol,Real}()
params[:α] = 30 # Degrees of freedom of the t-Student

te = Embeddings.tSTE(triplets, dimensions, params)
@time violations = Embeddings.compute(te; max_iter=50)

params = Dict{Symbol,Real}()
params[:σ] = 1/sqrt(2) # Normal distribution variance

te = Embeddings.STE(triplets, dimensions, params)
@time violations = Embeddings.compute(te; max_iter=50)

te = Embeddings.HingeGNMDS(triplets, dimensions)
@time violations = Embeddings.compute(te; max_iter=50)

# Obtain the embedding from the struct te and plot it
plot(Embeddings.X(te))
```
