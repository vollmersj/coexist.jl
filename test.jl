using PyCall #env using Conda
using Test

include("julia/diseaseProg.jl")

py"""
path = "/Users/aa25desh/Downloads/coexit/coexist-julia/test.py" #change path according?
exec(open(path).read())
"""

intial_state = (nAge = 9, nHS = 8 ,nIso = 4)

@testset "diseaseprog" begin
    @test  py"trFunc_diseaseProgression()"==trFunc_diseaseProgression(;intial_state...)
end
