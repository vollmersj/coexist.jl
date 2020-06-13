using Coexist, PyCall
Coexist.pyCoexist_setup()

pydir = joinpath(@__DIR__,"..", "coexist_python", "model_COVID_testing-ODEoutput.py")
py"""
import os
exec(open(os.path.join($pydir)).read())
"""

state = 50*ones(9*8*4*4)
dstate = similar(state)

function solveSystem(
	state,
	timeSpan = (0.0, 80.0),
	p=nothing
	)
	prob = ODEProblem(Coexist.dydt,state,(0.0,80.0),p=nothing)
	sol = solve(prob,Tsit5(),reltol=1e-3,abstol=1e-3)
	sol = convert(Array, sol)
	return reshape(sol, (4,4,8,9, Int64(prod(size(sol))/(4*4*8*9))))
end

dydt_benchmark = @benchmark Coexist.dydt(dstate,state,0,nothing)
solveSystem_benchmark = @benchmark solveSystem(state)
py_dydt_benchmark = @benchmark py"solveSystem(state, 80, **paramDict_current)"
py_solveSystem_benchmark = @benchmark py"solveSystem(state, 80, **paramDict_current)"

println("BENCHMARK OF dydt_Complete")
display(dydt_benchmark)
println()
println("BENCHMARK OF solveSystem")
display(solveSystem_benchmark)
println()
println("BENCHMARK OF py_dydt_Complete")
display(py_dydt_benchmark)
println()
println("BENCHMARK OF py_solveSystem")
display(py_solveSystem_benchmark)
println()
