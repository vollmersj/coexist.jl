using ModelingToolkit
using DifferentialEquations
using Plots
using BenchmarkTools
using Coexist
using Parameters

@with_kw mutable struct dydt_Complete
  # initial state
  nAge=9
  nHS=8
  nIso=4
  nI=4
  nTest=4
  ssize=(nAge, nHS, nIso, nTest)
  param=(nAge=nAge, nHS=nHS, nIso=nIso, nTest=nTest)
  trFunc_diseaseProgression=Coexist.trFunc_diseaseProgression()
end

function (f::dydt_Complete)(stateTensor_flattened, p, t)
  T = eltype(stateTensor_flattened)
  stateTensor = reshape(stateTensor_flattened, (f.nTest, f.nIso, f.nHS, f.nAge))
  dydt = zeros(T,size(stateTensor)...)
  trTensor_complete = zeros(T,(f.nTest, f.nIso, f.nHS, f.nTest, f.nIso, f.nHS, f.nAge))
  trTensor_diseaseProgression = f.trFunc_diseaseProgression(;f.param...)
  for k in 1:4
    shape = size(trTensor_diseaseProgression[:,k,:,:])
    expand_dims = reshape(trTensor_diseaseProgression[:,k,:,:],
          (shape[1:end-2]..., 1, shape[end-1:end]...)) # equal to np.exapand_dims
    view = Coexist.einsum("ijlml->ijlm", trTensor_complete[:,k,:,:,k,:,:],T)
    view .+= expand_dims
    trTensor_complete[:,k,:,:,k,:,:] = Coexist._einsum12(trTensor_complete[:,k,:,:,k,:,:], view, T)
  end

  to_be_modified = Coexist.einsum("ijkljkl->ijkl", trTensor_complete, T)
  to_be_modified .-= Coexist.einsum("...jkl->...", trTensor_complete, T)
  trTensor_complete = Coexist._einsum11(trTensor_complete, to_be_modified, T)

  dydt = Coexist.einsum("ijkl,ijklmnp->imnp", stateTensor, trTensor_complete, T)
  return vec(dydt)
end


@with_kw mutable struct solveSystem
  timeSpan=(0.0, 80.0)
  p=nothing
  dydt_Complete=dydt_Complete()
end

function (f::solveSystem)(state, timeSpan=(0.0, 80.0), p=nothing)
  f.timeSpan=timeSpan
  f.p=p
  println("solveSystem")
  dstate = similar(state)

  # @variables t u[1:1152](t)
  # @derivatives D'~t
  # dydt_symbolic = f.dydt_Complete(u,0,nothing)
  # dydt_symbolic = simplify.(dydt_symbolic)
  # sys = ODESystem(D.(u) .~ dydt_symbolic,t,u,[])
  # computed_fn = generate_function(sys)[2]
  #
  # fn = eval(computed_fn)

  ## open(joinpath(@__DIR__,"..","src","generated_dydt.jl"), "w") do io
  ##    write(io, "const dydt = $computed_f")
  ## end

  # prob = ODEProblem(fn,state,(0.0,80.0),p=nothing)
  prob = ODEProblem(f.dydt_Complete,state,(0.0,80.0),p=nothing)
	sol = solve(prob,Tsit5(),reltol=1e-3,abstol=1e-3)
	sol = convert(Array, sol)
	reshape(sol, (4,4,8,9, length(sol) ÷ (4*4*8*9)))
end

state = 50*ones(9*8*4*4)

function solveSystem_unoptimized(
	state,
	timeSpan = (0.0, 80.0),
	p=nothing
	)
	prob = ODEProblem(dydt_Complete,state,(0.0,80.0),p=nothing)
	sol = solve(prob,Tsit5(),reltol=1e-3,abstol=1e-3)
	sol = convert(Array, sol)
	return reshape(sol, (4,4,8,9, length(sol) ÷ (4*4*8*9)))
end

soln = solveSystem()(state)

# using BenchmarkTools
# dydt_benchmark = @benchmark dydt_Complete(state,0,nothing)
# f_benchmark = @benchmark f(dstate,state,0,nothing)
# solveSystem_unoptimized_benchmark = @benchmark solveSystem_unoptimized(state)
# solveSystem_benchmark = @benchmark solveSystem(state)
#
# println("BENCHMARK OF f")
# display(dydt_benchmark)
# display(f_benchmark)
# println()
# println("BENCHMARK OF solveSystem")
# display(solveSystem_unoptimized_benchmark)
# display(solveSystem_benchmark)
# println()
