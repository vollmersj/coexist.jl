# Library Imports
using Parameters
using Test

## Existing Function
function adjustRatesByAge_KeepAverageRateTest(rate; agePopulationRatio=agePopulationRatio,
                                                ageRelativeAdjustment::Array=[],
                                                maxOutRate::Float64=10.0)
    if rate == 0
        return fill(0, size(ageRelativeAdjustment))
    end
    if rate >= maxOutRate
        @warn("covidTesting::adjustRatesByAge_KeepAverageRate Input rate $rate >
                     maxOutRate $maxOutRate, returning input rates")
        return rate*(fill(1, size(ageRelativeAdjustment)))
    end
    out = fill(0, size(ageRelativeAdjustment))
    out[1] = maxOutRate + 1
    while sum(out .>= maxOutRate) > 0
        corrFactor = sum(agePopulationRatio ./ (1 .+ ageRelativeAdjustment))
        out =  rate * (1 .+ ageRelativeAdjustment) * corrFactor
        if sum(out .>= maxOutRate) > 0
            @warn("covidTesting::adjustRatesByAge_KeepAverageRate Adjusted rate
                   larger than $maxOutRate encountered, reducing ageAdjustment
                   variance by 10%")
            tmp_mean = sum(ageRelativeAdjustment)/length(ageRelativeAdjustment)
            ageRelativeAdjustment = tmp_mean .+ sqrt(0.9)*(
                                            ageRelativeAdjustment .- tmp_mean)
        end
    end
    return out
end

## Callable struct

@with_kw mutable struct adjustRatesByAge_KeepAverageRate
	agePopulationRatio::Float64=agePopulationRatio
	ageRelativeAdjustment::Array=[]
	maxOutRate::Float64=10.0
end
#	maxOutRate::Float64=10.0

function (f::adjustRatesByAge_KeepAverageRate)(rate; agePopulationRatio=agePopulationRatio,
	    ageRelativeAdjustment::Array=[],
	    maxOutRate::Float64=10.0
	)
	f.agePopulationRatio = agePopulationRatio
	f.ageRelativeAdjustment = ageRelativeAdjustment
	f.maxOutRate = maxOutRate
	if rate == 0
        return fill(0, size(f.ageRelativeAdjustment))
    end
    if rate >= f.maxOutRate
        @warn("covidTesting::adjustRatesByAge_KeepAverageRate Input rate $rate >
                     maxOutRate $(f.maxOutRate), returning input rates")
        return rate*(fill(1, size(f.ageRelativeAdjustment)))
    end
    out = fill(0, size(f.ageRelativeAdjustment))
    out[1] = f.maxOutRate + 1
    while sum(out .>= f.maxOutRate) > 0
        corrFactor = sum(f.agePopulationRatio ./ (1 .+ f.ageRelativeAdjustment))
        out =  rate * (1 .+ f.ageRelativeAdjustment) * corrFactor
        if sum(out .>= f.maxOutRate) > 0
            @warn("covidTesting::adjustRatesByAge_KeepAverageRate Adjusted rate
                   larger than $(f.maxOutRate) encountered, reducing ageAdjustment
                   variance by 10%")
            tmp_mean = sum(f.ageRelativeAdjustment)/length(f.ageRelativeAdjustment)
            f.ageRelativeAdjustment = tmp_mean .+ sqrt(0.9)*(
                                            f.ageRelativeAdjustment .- tmp_mean)
        end
    end
    return out
end

@testset "Function vs Callable Struct" begin
	@test adjustRatesByAge_KeepAverageRateTest(10; ageRelativeAdjustment=[1,2,3]) ==
		adjustRatesByAge_KeepAverageRate()(10; ageRelativeAdjustment=[1,2,3])
end
