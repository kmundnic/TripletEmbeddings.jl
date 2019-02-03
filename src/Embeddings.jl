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

    function X(te::TripletEmbedding)
    	return te.X.X
    end

    function no_items(te::TripletEmbedding)
    	return te.no_items
    end

    function no_triplets(te::TripletEmbedding)
    	return te.no_triplets
    end

    function dimensions(te::TripletEmbedding)
    	return te.dimensions
    end

    function triplets(te::TripletEmbedding)
    	return te.triplets
    end

    include("utilities.jl")
    include("compute.jl")
    include("tSTE.jl")

end # module