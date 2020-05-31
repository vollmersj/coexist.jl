using ModelingToolkit
using DifferentialEquations

# initial state
nAge=9
nHS=8
nIso=4
nI=4
nTest=4

# yeah
include("utils.jl")
param=(nAge=nAge, nHS=nHS, nIso=nIso, nTest=nTest)

include("diseaseProg.jl")

ssize=(nAge, nHS, nIso, nTest)

using PyCall
using Test

PyCall.pyimport_conda("pandas", "pandas")
PyCall.pyimport_conda("numpy", "numpy")
PyCall.pyimport_conda("scipy", "scipy")
PyCall.pyimport_conda("dask", "dask")
PyCall.pyimport_conda("cloudpickle", "cloudpickle")
PyCall.pyimport_conda("distributed", "distributed")
PyCall.pyimport_conda("xlrd", "xlrd")

py"""
import os
exec(open(os.path.join(os.getcwd(), "coexist_python", "model_COVID_testing-ODEoutput.py")).read())
"""

function dydt_Complete(stateTensor_flattened,p,t)
	stateTensor = reshape(stateTensor_flattened, (nTest, nIso, nHS, nAge))
	dydt = zeros(size(stateTensor)...)
	trTensor_complete = zeros((nTest, nIso, nHS, nTest, nIso, nHS, nAge))
	trTensor_diseaseProgression = trFunc_diseaseProgression(;param...)
	for k in 1:4
		shape = size(trTensor_diseaseProgression[:,k,:,:])
		expand_dims = reshape(trTensor_diseaseProgression[:,k,:,:],
					(shape[1:end-2]..., 1, shape[end-1:end]...)) # equal to np.exapand_dims
		# slice = trTensor_complete[:,k,:,:,k,:,:]
		# @einsum view[m,l,j,i] := slice[l,m,l,j,i]
		view = einsum("ijlml->ijlm", trTensor_complete[:,k,:,:,k,:,:])
		view .+= expand_dims
		trTensor_complete[:,k,:,:,k,:,:] = _einsum12(trTensor_complete[:,k,:,:,k,:,:], view)
		# @einsum slice[l,m,l,j,i] = view[m,l,j,i]
		# trTensor_complete[:,k,:,:,k,:,:] = slice
	end

	# @einsum to_be_modified[l,k,j,i] := trTensor_complete[l,k,j,l,k,j,i]
	to_be_modified = einsum("ijkljkl->ijkl", trTensor_complete)
	to_be_modified .-= einsum("...jkl->...", trTensor_complete)
	# @einsum trTensor_complete[l,k,j,l,k,j,i] = view[l,k,j,i]
	trTensor_complete = _einsum11(trTensor_complete, to_be_modified)


	dydt = einsum("ijkl,ijklmnp->imnp", stateTensor, trTensor_complete)
	return reshape(dydt, prod(size(dydt)))
end

state = 50*ones(9*8*4*4)
r = dydt_Complete(state,0,nothing)

@testset "dydt_Complete" begin
	@test r == py"out"
end

prob = ODEProblem(dydt_Complete,state,(0.0,80.0),p=nothing)
sol = solve(prob,Tsit5(),reltol=1e-3,abstol=1e-3)
sol = convert(Array, sol)
sol = reshape(sol, (4,4,8,9, Int64(prod(size(sol))/(4*4*8*9))))

number_of_deaths = sum(sol[:,:,8,:,:], dims=[1,2,3]) # the last dim is time
number_of_deaths = reshape(number_of_deaths, prod(size(number_of_deaths)))
plot(1:prod(size(number_of_deaths)), number_of_deaths)
