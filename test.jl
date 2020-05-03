using PyCall
using Test
py"""
exec(open('model_COVID_testing_initial.py').read())
"""
@testset "diseaseprog" begin
    @test  py"sq"(2)==4
end
