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

## Loading the package
There's two ways to use this package:

  1. Download it into some `path`, and then load in Julia using:

  ```julia
	include("path/TripletEmbeddings.jl/src/TripletEmbeddings.jl")
	using .TripletEmbeddings
  ```
  
  2. Download it into some `path`. Add the `path` to `LOAD_PATH` in Julia using `push!(LOAD_PATH, path)`. Then, you can simply use `using TripletEmbeddings`. If you want it added to the path automatically every time you start a Julia session, add `push!(LOAD_PATH, path)` to your `~/.julia/config/startup.jl` file.
  
## Using the package
To use multiple threads, on the command line, write (before opening Julia):

```bash
export JULIA_NUM_THREADS=n
```
where `n` is the number of threads (defaults to 1).

The following code is a usage example:

```julia
using Plots
using Random
using TripletEmbeddings # see instructions above

Random.seed!(4)

# Load data dummy data
data = TripletEmbeddings.load_data()

# Generate triplets
triplets = TripletEmbeddings.label(data)

# Create an embedding with the triplets and compute the embedding
dimensions = 1
params = Dict{Symbol, Real}()
params[:σ] = 1/sqrt(2) # Normal distribution variance

te = TripletEmbeddings.STE(triplets, dimensions, params)
@time violations = TripletEmbeddings.fit!(te; max_iter=50, verbose=true)

TripletEmbeddings.scale!(data, te)
plot(data, label="Data")
plot!(TripletEmbeddings.X(te), label="Embedding") # Can also use te.X.X to access the embedding
```
