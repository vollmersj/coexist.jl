using ModelingToolkit, OrdinaryDiffEq, Base.Threads

# yeah
include("utils.jl")
param=(nAge=nAge, nHS=nHS, nIso=nIso, nTest=nTest)

#include("diseaseProg.jl")
include("diseaseProg_basic.jl")

# todo replace with inital state tensor
stateTensor= 100*ones((nAge, nHS, nIso, nTest))
dstateTensor=100*ones((nAge, nHS, nIso, nTest))


ssize=(nAge, nHS, nIso, nTest)


nIso=4
nTest=4
nHS=8
nAge=9

ssize=(nAge, nHS, nIso, nTest)
param=(nAge=nAge,nIso=nIso,nTest=nTest)
function f!(dstateTensor,stateTensor_vec,p,t)
  trTensor_complete = zeros(Operation,(nAge, nHS, nIso, nTest, nHS, nIso, nTest))
  trTensor_diseaseProgression=trFunc_diseaseProgression(param...)
  for tst=1:nTest, iso=1:nIso, nhs=1:nHS, ntst=1:nTest, niso=1:nIso, hs=1:nHS, ag=1:nAge
    trTensor_complete[ag,hs,iso,tst,nhs,iso,tst]+=trTensor_diseaseProgression[ag,hs,iso,nhs]
  end
  dstateTensor .= false
  for tst=1:nTest, iso=1:nIso, nhs=1:nHS, ntst=1:nTest, niso=1:nIso, hs=1:nHS, ag=1:nAge
    dstateTensor[ag,hs,iso,tst]+= stateTensor[1,nhs,niso,ntst] * trTensor_complete[ag,nhs,niso,ntst,hs,iso,tst]
  end
end
trTensor_complete = zeros(Operation,(nAge, nHS, nIso, nTest, nHS, nIso, nTest))
trTensor_diseaseProgression=trFunc_diseaseProgression(param...)

f!(dstateTensor,stateTensor,nothing,0.0)
u0= 50*ones(nAge, nHS, nIso, nTest)
prob = ODEProblem(f!,u0,(0.0,80.0))
sys = modelingtoolkitize(prob)
simplified_exs = simplify.(sys.eqs) # Can take a bit, but gives more optimized code
simplifiedsys = ODESystem(simplified_exs,sys.iv,sys.states,sys.ps)
generate_function(simplifiedsys,multithread=true) # Generate the Julia code to save




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
