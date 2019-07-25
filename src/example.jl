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