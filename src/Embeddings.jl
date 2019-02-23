module Embeddings

    using Printf
	using Random
	using Distances
	using Statistics
	using LinearAlgebra
	using DelimitedFiles
	using Base.Threads

	abstract type TripletEmbedding end
	
	"""
	This is a wraper for a 2-dimensional array containing the
	computed embedding. We define this wraper so that we can 
	have a mutable field within the defined triplet embedding.
	"""
	mutable struct Embedding
    	X::Array{Float64,2}
    end

	"""
	The @def macro let's us populate any kind of triplet embedding
	with the necessary fields.
	Note that for many types of embeddings, only the parameters
	within params will change, and we will define the appropriate
	loss function.
	"""
	macro def(name, definition)
		return quote
			macro $(esc(name))()
				esc($(Expr(:quote, definition)))
			end
		end
	end

	"""
	We define a macro @add_embedding_fields with all common field
	types for all possible embeddings
	"""
	@def add_embedding_fields begin
	    triplets::Array{Int64,2}
	    dimensions::Int64
	    params::Dict{Symbol,Real}
		X::Embedding
	    no_triplets::Int64
	    no_items::Int64
	end

	"""
	We define a macro @check_embedding_conditions with all common conditions
	for fields of all possible embeddings
	"""
	@def check_embedding_conditions begin
        if size(triplets, 2) != 3
            throw(ArgumentError("Triplets do not have three values"))
        end

        if sort(unique(triplets)) != 1:no_items
            throw(ArgumentError("Triplet values must have all elements in the range 1:no_items"))
        end         

        if dimensions < 1
            throw(ArgumentError("Dimensions must be >= 1"))
        end
    
        if no_items != size(X.X, 1)
            throw(ArgumentError("Number of elements in triplets does not match the embedding dimension"))
        end

        if size(X.X, 2) != dimensions
            throw(ArgumentError("dimensions and embedding dimensions do not match"))
        end
	end

    function triplets(te::TripletEmbedding)
    	return te.triplets
    end

    function dimensions(te::TripletEmbedding)
    	return te.dimensions
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

    include("utilities.jl")
    include("compute.jl")
    include("tSTE.jl")
    include("STE.jl")
    include("GNMDS.jl")
    include("CKL.jl")

end # module