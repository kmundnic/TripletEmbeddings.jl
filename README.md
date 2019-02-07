# TripletEmbeddings.jl
This is a Julia 1.0 implementation of Triplet Embeddings in Julia. Currently, Stochastic Triplet Embeddings (STE) and t-Student Stochastic Triplet Embeddings (t-STE) are implemented.

These implementations are based on based on Michael Wilber's implementation of [tSTE in Cython](www.github.com/gcr/cython_tste) and Laurens Van der Maaten's implementation in [MATLAB](https://lvdmaaten.github.io/ste/Stochastic_Triplet_Embedding.html). However, this implementation uses multithreading for the calculation of the gradients, making it the fastest implementation.