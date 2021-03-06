using ModelingToolkit
using DifferentialEquations
#using TensorOperations
#using Einsum




x=1
nAge=9
nHS=8
nIso=4
nI=4
nTest=4

# yeah
include("utils.jl")
param=(nAge=nAge, nHS=nHS, nIso=nIso, nTest=nTest)

# include("diseaseProg.jl")
include("diseaseProg_basic.jl")

# todo replace with inital state tensor
stateTensor= 100*ones((nAge, nHS, nIso, nTest))
dstateTensor=100*ones((nAge, nHS, nIso, nTest))
trTensor_complete = zeros((nAge, nHS, nIso, nTest, nHS, nIso, nTest))
  #@einsum dstateTensor[i,m,n,p]:=stateTensor[i,j,k,l]* trTensor_complete[i,j,k,l,m,n,p]

  # if we want to use TensorOperations which seems maintained this is the way to go
  # does not work @tensor ddstateTensor[1,m,n,p]:=stateTensor[1,j,k,l]* trTensor_complete[1,j,k,l,m,n,p]
  # for i=1:nAge
  #
  #   v=view(ddstateTensor,i,:,:,:)
  #   @tensor v[m,n,p]=stateTensor[1,j,k,l]* trTensor_complete[1,j,k,l,m,n,p]
  # end
ssize=(nAge, nHS, nIso, nTest)

# Playground
# #dydt = np.einsum('ijkl,ijklmnp->imnp', stateTensor, trTensor_complete) # contract the HS axis, keep age
# A=ones((3,3))
# b=ones(3)
# C=ones((3,3))
# @einsum C[1,i]=A[i,j]*b[j]
# c=view(C,1,:)

# # https://tutorials.sciml.ai/html/introduction/03-optimizing_diffeq_code.html
# c[:]=zeros(3)
# C

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
# print(os.getcwd())
exec(open(os.path.join(os.getcwd(), "coexist_python", "model_COVID_testing-ODEoutput.py")).read())
"""

function f_vec!(dstateTensor_vec,stateTensor_vec,p,t)
  trTensor_diseaseProgression=trFunc_diseaseProgression(param...)
  stateTensor=reshape(stateTensor_vec,(nTest, nIso, nHS, nAge))
  dstateTensor = zeros(size(stateTensor)...)
  trTensor_complete = zeros((nTest, nIso, nHS, nTest, nIso, nHS, nAge))

  # Efficient Python version
  # # Get disease condition updates with no isolation or test transition ("diagonal along those")
  #    for k1 in [0,1,2,3]:
  #        np.einsum('ijlml->ijlm',
  #            trTensor_complete[:,:,k1,:,:,k1,:])[:] += np.expand_dims(
  #                trTensor_diseaseProgression[:,:,k1,:]
  #                ,[2])  # AUTOMATICALLY broadcast all non-hospitalised disease progression is same
  # @cast from TensorCast only works with two repeeated index
  # https://discourse.julialang.org/t/einstein-convention-for-slicing-or-how-to-view-diagonals-in-tensors/38553/3
  # maybe with CartesianIndex or @views
  for iso=1:nIso
    for ag=1:nAge
      for hs=1:nHS
        for tst=1:nTest
          for nhs=1:nHS #new health state
              trTensor_complete[ag,hs,iso,tst,nhs,iso,tst]+=trTensor_diseaseProgression[ag,hs,iso,nhs]
          end
        end
      end
    end
  end
  @testset "Checking intermediate state" begin
    @test py"test1" == trTensor_diseaseProgression
    @test py"test" == trTensor_diseaseProgression
  end

  for ag=1:nAge
    for hs=1:nHS
      for iso=1:nIso
        for tst=1:nTest
          for nhs=1:nHS
            for niso=1:nIso
              for ntst=1:nTest
                dstateTensor[ag,hs,iso,tst]+=stateTensor[1,nhs,niso,ntst,]* trTensor_complete[ag,nhs,niso,ntst,hs,iso,tst]
              end
            end
          end
        end
      end
    end
  end
  dstate[:]=reshape(dstateTensor,(nIso*nAge*nHS*nTest))
end
#= This one is close
  for ag=1:nAge
    for hs=1:nHS
      for iso=1:nIso
        for tst=1:nTest
          for nhs=1:nHS
            for niso=1:nIso
              for ntst=1:nTest
                dstateTensor[ag,hs,iso,ntst]+=stateTensor[tst,nhs,niso,ntst]* trTensor_complete[ag,hs,iso,tst,nhs,niso,ntst]
              end
            end
          end
        end
      end
    end
  end
=#

n=prod([ssize...])
println(n)
dstate= 50*ones(n)
state= ones(n)
println(size(state))
out = f_vec!(dstate,state,nothing,0.0)
@testset "Testing output" begin
	@test py"out" == out
end
# println(out)
# println(size(out))

# prob = ODEProblem(f_vec!,100*ones(n),(0.0,80.0),p=nothing)
# # sys = modelingtoolkitize(prob)
# dstate=100*ones(n)
# state=100*ones(n)
# f_vec!(dstate,state,[],0.0)

# sol = solve(prob,Tsit5())
# println(sol)

# sum to get all infacted sum(stateTensor[:,:,8,:],[1,2,3])



#using Pkg
#Pkg.add(PackageSpec(url="https://github.com/JuliaDiffEq/SciPyDiffEq.jl", rev="master"))
# using SciPyDiffEq
# SciPyDiffEq.RK23


#
# fun = lambda t,y: dydt_Complete(t,y, **kwargs),
# t_span=(0.,total_days),
# y0 = cur_stateTensor,
# method='RK23',
# t_eval=range(total_days),
# rtol = 1e-3, #default 1e-3
# atol = 1e-3, # default 1e-6
# SciPyDiffEq code is compiling forever
# sol = solve(prob,SciPyDiffEq.RK23(),rtol=1.0e-3,atol=1.0e-3,saveat=1.0*[0:80;])
