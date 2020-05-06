using PyCall #env using Conda
using Test

include("julia/diseaseProg.jl")

py"""
import os
relative_path = os.getcwd()
path = os.path.join(relative_path, 'coexist-julia/test.py')
exec(open(path).read())
"""

intial_state = (nAge = 9, nHS = 8 ,nIso = 4)

@testset "diseaseprog" begin
    @test  py"trFunc_diseaseProgression()"==trFunc_diseaseProgression(;intial_state...)
end
