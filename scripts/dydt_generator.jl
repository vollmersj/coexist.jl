using ModelingToolkit
using DifferentialEquations
using Plots
using BenchmarkTools
using coexist

# initial state
nAge=9
nHS=8
nIso=4
nI=4
nTest=4
param=(nAge=nAge, nHS=nHS, nIso=nIso, nTest=nTest)
ssize=(nAge, nHS, nIso, nTest)

function dydt_Complete(stateTensor_flattened,p,t)
	T = eltype(stateTensor_flattened)
	stateTensor = reshape(stateTensor_flattened, (nTest, nIso, nHS, nAge))
	dydt = zeros(T,size(stateTensor)...)
	trTensor_complete = zeros(T,(nTest, nIso, nHS, nTest, nIso, nHS, nAge))
	trTensor_diseaseProgression = coexist.trFunc_diseaseProgression(;param...)
	for k in 1:4
		shape = size(trTensor_diseaseProgression[:,k,:,:])
		expand_dims = reshape(trTensor_diseaseProgression[:,k,:,:],
					(shape[1:end-2]..., 1, shape[end-1:end]...)) # equal to np.exapand_dims
		# slice = trTensor_complete[:,k,:,:,k,:,:]
		# @einsum view[m,l,j,i] := slice[l,m,l,j,i]
		view = coexist.einsum("ijlml->ijlm", trTensor_complete[:,k,:,:,k,:,:],T)
		view .+= expand_dims
		trTensor_complete[:,k,:,:,k,:,:] = coexist._einsum12(trTensor_complete[:,k,:,:,k,:,:], view, T)
		# @einsum slice[l,m,l,j,i] = view[m,l,j,i]
		# trTensor_complete[:,k,:,:,k,:,:] = slice
	end

	# @einsum to_be_modified[l,k,j,i] := trTensor_complete[l,k,j,l,k,j,i]
	to_be_modified = coexist.einsum("ijkljkl->ijkl", trTensor_complete, T)
	to_be_modified .-= coexist.einsum("...jkl->...", trTensor_complete, T)
	# @einsum trTensor_complete[l,k,j,l,k,j,i] = view[l,k,j,i]
	trTensor_complete = coexist._einsum11(trTensor_complete, to_be_modified, T)


	dydt = coexist.einsum("ijkl,ijklmnp->imnp", stateTensor, trTensor_complete, T)
	return vec(dydt)
end

state = 50*ones(9*8*4*4)
dstate = similar(state)

using ModelingToolkit
@variables t u[1:1152](t)
@derivatives D'~t

dydt_symbolic = dydt_Complete(u,0,nothing)
dydt_symbolic = simplify.(dydt_symbolic)
sys = ODESystem(D.(u) .~ dydt_symbolic,t,u,[])
computed_f = generate_function(sys)[2]

open(joinpath(@__DIR__,"..","src","generated_dydt.jl"), "w") do io
   write(io, "const dydt = $computed_f")
end

f = eval(computed_f)

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

function solveSystem(
	state,
	timeSpan = (0.0, 80.0),
	p=nothing
	)
	prob = ODEProblem(f,state,(0.0,80.0),p=nothing)
	sol = solve(prob,Tsit5(),reltol=1e-3,abstol=1e-3)
	sol = convert(Array, sol)
	reshape(sol, (4,4,8,9, length(sol) ÷ (4*4*8*9)))
end

using BenchmarkTools
dydt_benchmark = @benchmark dydt_Complete(state,0,nothing)
f_benchmark = @benchmark f(dstate,state,0,nothing)
solveSystem_unoptimized_benchmark = @benchmark solveSystem_unoptimized(state)
solveSystem_benchmark = @benchmark solveSystem(state)

println("BENCHMARK OF f")
display(dydt_benchmark)
display(f_benchmark)
println()
println("BENCHMARK OF solveSystem")
display(solveSystem_unoptimized_benchmark)
display(solveSystem_benchmark)
println()
