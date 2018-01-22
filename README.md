# projected-tSTE
This code accompanies the paper Mundnich, K., Booth, B., Narayanan, Shrikanth, "Towards continuous-time annotations using projected stochastic triplet embeddings", submitted to ICASSP 2018.

This paper proposes the use of projected gradient descent when using tSTE to reconstruct 1-dimensional embeddings. The (convex) projection is performed using a median filter that filters the embeddings in the first _n_ iterations.

## Installation
Requirements:

- Julia v0.6 or greater with the following packages (earlier versions might work as well):
	- ToeplitzMatrices v0.3.0
	- DataFrames v0.11.4
	- Distances v0.5.0
	- Convex v0.5.0
	- SCS v0.3.3
	- PyCall v1.15.0
	- JLD v0.8.3
	- MAT v0.4.0
- Python v2.7
	- Numpy v0.19 or greater
	- Cython v0.25.2

In MacOS 10.10 or later, the easiest way is to install Anaconda Python, and download the Julia .dmg file from www.julialang.org/downloads

Once installed, go to the cy_tste folder and run:

	python setup.py build_ext install
	
to install cy_tste globally. Triplets.jl contains the code that uses `PyCall` to use Cython from Julia.

## Experiments
To run the experiments, open a Julia REPL and run:
	
	include('parameters.jl')
	
Or from a terminal, just run

	julia parameters.jl
	
Figures are generated running the `.m` files under the folder /figures.

