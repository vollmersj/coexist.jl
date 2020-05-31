using ModelingToolkit
using DifferentialEquations
using Einsum

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

function dydt_Complete(t, stateTensor_flattened)
	stateTensor = reshape(stateTensor_flattened, (nTest, nIso, nHS, nAge))
	dydt = zeros(size(stateTensor)...)
	trTensor_complete = zeros((nTest, nIso, nHS, nTest, nIso, nHS, nAge))
	trTensor_diseaseProgression = trFunc_diseaseProgression(;param...)
	for k in 1:4
		_s = size(trTensor_diseaseProgression[:,k,:,:])
		_y = reshape(trTensor_diseaseProgression[:,k,:,:], (_s[1:end-2]..., 1, _s[end-1:end]...))
		_t = trTensor_complete[:,k,:,:,k,:,:]
		@einsum _temp[m,l,j,i] := _t[l,m,l,j,i]
		_temp .+= _y
		@einsum _t[l,m,l,j,i] = _temp[m,l,j,i]
		trTensor_complete[:,k,:,:,k,:,:] = _t
	end

	# @einsum _mod[l,k,j,i] := trTensor_complete[l,k,j,l,k,j,i]
	_mod = einsum("ijkljkl->ijkl", trTensor_complete)
	_mod .-= einsum("...jkl->...", trTensor_complete)
	# @einsum trTensor_complete[l,k,j,l,k,j,i] = _temp[l,k,j,i]
	trTensor_complete = _einsum11(trTensor_complete, _mod)


	dydt = einsum("ijkl,ijklmnp->imnp", stateTensor, trTensor_complete)
	return reshape(dydt, prod(size(dydt)))
end

state = 50*ones(9*8*4*4)
r = dydt_Complete(0, state)

@testset "lol" begin
	@test r == py"out"
end
