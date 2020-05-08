using PyCall #env using Conda
using Test

include("julia/diseaseProg.jl")

py"""
import os
relative_path = os.getcwd()
path = os.path.join(relative_path, 'test.py')
# print(path)
exec(open(path).read())
"""

initial_state = (nAge=9, nHS=8, nIso=4, nI=4)

@testset "diseaseprog" begin
    @test  py"trFunc_diseaseProgression()"==trFunc_diseaseProgression(;initial_state...)
end

@testset "hospitaladmission" begin
	@test py"trFunc_HospitalAdmission()"==trFunc_HospitalAdmission(;initial_state...)
end
