using ModelingToolkit
using DifferentialEquations
using TensorOperations
using Einsum

## lets try tensor states that just one dimensional ODE for each component
# Attempt1
# Health states (S, E and D are fixed to 1 dimension)
nI_symp = 2 # number of sympyomatic infected states
nI = 2+nI_symp # number of total infected states (disease stages), the +2 are Exposed and I_nonsymptomatic
nR = 2 # number of recovery states (antibody development post-disease, IgM and IgG are two stages)
nHS = 2+nI+nR # number of total health states, the +2: S, D are suspectible and dead

# Age groups (risk groups)
nAge = 9 # In accordance w Imperial #13 report (0-9, 10-19, ... 70-79, 80+)

# Isolation states
nIso = 4 # None/distancing, Case isolation, Hospitalised, Hospital staff

# Testing states
nTest = 4 # untested/negative, Virus positive, Antibody positive, Both positive


#@variables t
#@variables stateTensor[1:nAge,1:nHS,1:nIso,1:nTest][t]
#@variables dstateTensor[1:nAge,1:nHS,1:nIso,1:nTest][t]

@derivatives D'~t

eqs = [D(stateTensor) ~stateTensor]

u0 = [stateTensor => ones((nAge, nHS, nIso, nTest))]
sys = ODESystem(eqs)
tspan = (0.0,100.0)
prob = ODEProblem(sys,u0,tspan,p;jac=true,sparse=true)


## Attemp 2 via

function f!(dstateTensor,stateTensor,p,t)
      dstateTensor[:]=stateTensor
end

prob = ODEProblem(f!,ones((nAge, nHS, nIso, nTest)),(0.0,2.0),p=nothing)
sys = modelingtoolkitize(prob)
sol = solve(prob,Tsit5())

# yeah
include("utils.jl")
include("diseaseProg_basic.jl")


stateTensor= 100*ones((nAge, nHS, nIso, nTest))
dstateTensor=100*ones((nAge, nHS, nIso, nTest))
trTensor_complete = zeros((nAge, nHS, nIso, nTest, nHS, nIso, nTest))

f!(dstateTensor,stateTensor,nothing,0.0)
function f!(dstateTensor,stateTensor,p,t)
  trTensor_diseaseProgression=trFunc_diseaseProgression()

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
#  trTensor_complete = zeros((nAge, nHS, nIso, nTest, nHS, nIso, nTest))
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
  dstateTensor=zeros((nAge, nHS, nIso, nTest))

  for i=1:nAge

     v=view(dstateTensor,i,:,:,:)
     @tensor v[m,n,p]=stateTensor[1,j,k,l]* trTensor_complete[1,j,k,l,m,n,p]
   end
  #@einsum dstateTensor[i,m,n,p]:=stateTensor[i,j,k,l]* trTensor_complete[i,j,k,l,m,n,p]

  # if we want to use TensorOperations which seems maintained this is the way to go
  # does not work @tensor ddstateTensor[1,m,n,p]:=stateTensor[1,j,k,l]* trTensor_complete[1,j,k,l,m,n,p]
  # for i=1:nAge
  #
  #   v=view(ddstateTensor,i,:,:,:)
  #   @tensor v[m,n,p]=stateTensor[1,j,k,l]* trTensor_complete[1,j,k,l,m,n,p]
  # end
end

prob = ODEProblem(f!,ones((nAge, nHS, nIso, nTest)),(0.0,2.0),p=nothing)
sys = modelingtoolkitize(prob)
sol = solve(prob,Tsit5())






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
