module Embeddings

	using Printf
	using Random
	using Statistics
	using LinearAlgebra
	using DelimitedFiles

	export tSTE, Embedding

    abstract type TripletEmbedding end

    mutable struct Embedding
    	X::Array{Float64,2}
    end

    include("utilities.jl")
    include("compute.jl")
    include("tSTE.jl")

end # module