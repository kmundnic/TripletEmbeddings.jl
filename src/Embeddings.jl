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

    struct tSTE <: TripletEmbedding
		triplets::Array{Int64,2}
    	dimensions::Int64
    	params::Dict{Symbol,Real}
    	X::Embedding

    	function tSTE(
    		triplets::Array{Int64,2},
    		dimensions::Int64,
    		params::Dict{Symbol,Real},
    		X::Embedding)
    		
    		n::Int64 = maximum(triplets)
    		
			if size(triplets, 2) != 3
				throw(ArgumentError("Triplets do not have three values"))
			end

    		if sort(unique(triplets)) != 1:n
    			throw(ArgumentError("Triplet values must have all elements in the range 1:n (with n being the total number of elements)"))
    		end    		

    		if dimensions < 1
    			throw(ArgumentError("Dimensions must be >= 1"))
    		end
    	
			if n != size(X.X, 1)
				throw(ArgumentError("Number of elements in triplets does not match the embedding dimension"))
			end

    		if size(X.X, 2) != dimensions
    			throw(ArgumentError("dimensions and embedding dimensions do not match"))
    		end

    		if !haskey(params, :α)
    			throw(ArgumentError("params has no key :α"))
    		elseif params[:α] <= 1
    			throw(ArgumentError("Parameter α must be > 1"))
    		end

			if !haskey(params, :λ)
				throw(ArgumentError("params has no key :λ"))
			elseif params[:λ] < 0
				throw(ArgumentError("Regularizer λ must be >= 0"))
			end

    		new(triplets, dimensions, params, X)
    	end

    	function tSTE(
    		triplets::Array{Int64,2},
    		dimensions::Int64,
    		params::Dict{Symbol,Real})

    		X0 = Embedding(randn(maximum(triplets), dimensions))
    		new(triplets, dimensions, params, Embedding(X0))
    	end

    	function tSTE(
    		triplets::Array{Int64,2},
    		params::Dict{Symbol,Real},
    		X::Embedding)

    		dimensions = size(X,2)
    		new(triplets, dimensions, params, X)
    	end
    end

    include("utilities.jl")
    include("tSTE.jl")

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

end # module