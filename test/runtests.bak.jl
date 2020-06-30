using Coexist, PyCall#env using Conda
using Test
using Dates
using DataFrames
Coexist.pyCoexist_setup()

py"""
import os
relative_path = os.getcwd()
path = os.path.join(relative_path, 'test.py')
exec(open(path).read())
"""

## Setup for testing
initial_state = (nAge=9, nHS=8, nIso=4, nI=4, nTest=4, nR=2)

# `trFunc_travelInfectionRate_ageAdjusted`
t = 10 # Time within simulation

# `trFunc_testCapacity`
rTime = Date("2020-05-25", "yyyy-mm-dd") # Real Time
py"""
_trFunc_testCapacity = __trFunc_testCapacity
"""

# `inpFunc_testSpecifications`
_other, _name, __truePosHealthState = py"inpFunc_testSpecifications()"
other = Coexist.inpFunc_testSpecifications(;initial_state...)
name = convert(Array, select(other, :Name))
name = reshape(name, size(name)[1])
_truePosHealthState = [[] for i in 1:size(__truePosHealthState)[1]]
for i in 1:24
	_len = size(__truePosHealthState[i])[1]
	_temp = [0 for i in 1:_len]
	for j in 1:size(__truePosHealthState[i])[1]
		_temp[j] = __truePosHealthState[i][j]
	end
	_truePosHealthState[i] = _temp
end
truePosHealthState = convert(Array, select(other, :TruePosHealthState))
truePosHealthState = reshape(truePosHealthState, size(truePosHealthState)[1])

other = select(other, Not(:Name))
other = select(other, Not(:TruePosHealthState))
other = convert(Array, other)

# End of setup

@testset "DiseaseProg & HospitalAdmission" begin
    @test  py"np.transpose(trFunc_diseaseProgression())"==
	Coexist.trFunc_diseaseProgression(;initial_state...)
	@test py"np.transpose(trFunc_HospitalAdmission())"≈ # Approximate equality?
	Coexist.trFunc_HospitalAdmission(;initial_state...)
	@test py"np.transpose(trFunc_HospitalDischarge())"≈
	Coexist.trFunc_HospitalDischarge(;initial_state...)
	@test py"np.transpose(trFunc_travelInfectionRate_ageAdjusted(10))"≈
	Coexist.trFunc_travelInfectionRate_ageAdjusted(t)
	@test py"_trFunc_testCapacity"==
	Coexist.trFunc_testCapacity(Date("2020-05-25", "yyyy-mm-dd"))

	# inpFunc_testSpecifications
	@test _other == other
	@test _name == name
	@test _truePosHealthState == truePosHealthState

	@test py"ageSocialMixingBaseline" ≈ Coexist.ageSocialMixingBaseline
	@test py"ageSocialMixingDistancing" ≈ Coexist.ageSocialMixingDistancing
	@test transpose(Coexist.einsum("ijk,j->ik", Coexist.stateTensor[3:end,4,2:(4+1),:], Coexist.transmissionInfectionStage, eltype(Coexist.transmissionInfectionStage))*(Coexist.ageSocialMixingBaseline.-Coexist.ageSocialMixingDistancing))≈
	py"np.matmul(ageSocialMixingBaseline-ageSocialMixingDistancing,np.einsum('ijk,j->ik',stateTensor[:,1:(4+1),3,2:], transmissionInfectionStage))"
	@test py"trFunc_newInfections_Complete(stateTensor=stateTensor,policySocialDistancing=False, policyImmunityPassports=True)"≈
	permutedims(Coexist.trFunc_newInfections_Complete(Coexist.stateTensor,false,true;initial_state...),[3,2,1])
end
