using PyCall #env using Conda
using Test

include("julia/diseaseProg.jl")

py"""
import os
relative_path = os.getcwd()
path = os.path.join(relative_path, 'test.py')
exec(open(path).read())
"""

initial_state = (nAge=9, nHS=8, nIso=4, nI=4)

@testset "DiseaseProg & HospitalAdmission" begin
    @test  py"np.transpose(trFunc_diseaseProgression())"==
	trFunc_diseaseProgression(;initial_state...)
	@test py"np.transpose(trFunc_HospitalAdmission())"≈ # Approximate equality?
	trFunc_HospitalAdmission(;initial_state...)
	@test py"np.transpose(trFunc_HospitalDischarge())"≈ 
	trFunc_HospitalDischarge(;initial_state...)
end
