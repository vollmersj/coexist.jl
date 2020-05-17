using coexist, PyCall#env using Conda
using Test

include("../src/diseaseProg.jl")
# Install scikit-learn if not installed
PyCall.pyimport_conda("pandas", "pandas")
PyCall.pyimport_conda("numpy", "numpy")
PyCall.pyimport_conda("scipy", "scipy")
PyCall.pyimport_conda("dask", "dask")
PyCall.pyimport_conda("cloudpickle", "cloudpickle")
PyCall.pyimport_conda("distributed", "distributed")

py"""
import os
relative_path = os.getcwd()
path = os.path.join(relative_path, 'test.py')
exec(open(path).read())
"""

initial_state = (nAge=9, nHS=8, nIso=4, nI=4, nTest=4)
t = 10 # Time within simulation

@testset "DiseaseProg & HospitalAdmission" begin
    @test  py"np.transpose(trFunc_diseaseProgression())"==
	trFunc_diseaseProgression(;initial_state...)
	@test py"np.transpose(trFunc_HospitalAdmission())"≈ # Approximate equality?
	trFunc_HospitalAdmission(;initial_state...)
	@test py"np.transpose(trFunc_HospitalDischarge())"≈
	trFunc_HospitalDischarge(;initial_state...)
	@test py"np.transpose(trFunc_travelInfectionRate_ageAdjusted(10))"≈
	trFunc_travelInfectionRate_ageAdjusted(t)
	@test py"ageSocialMixingBaseline" ≈ ageSocialMixingBaseline
	@test py"ageSocialMixingDistancing" ≈ ageSocialMixingDistancing
	@test transpose(einsum("ijk,j->ik", stateTensor[3:end,1,2:(4+1),:], transmissionInfectionStage)*(ageSocialMixingBaseline.-ageSocialMixingDistancing))≈
	py"np.matmul(ageSocialMixingBaseline-ageSocialMixingDistancing,np.einsum('ijk,j->ik',stateTensor[:,1:(4+1),0,2:], transmissionInfectionStage))"
	@test py"trFunc_newInfections_Complete(stateTensor=stateTensor,policySocialDistancing=False, policyImmunityPassports=True)"≈
	permutedims(trFunc_newInfections_Complete(stateTensor,false,true;initial_state...),[3,2,1])
end
