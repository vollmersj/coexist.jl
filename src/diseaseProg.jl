# Library Imports
using DataFrames
using CSVFiles
using LinearAlgebra
using Dates
import StatsFuns: logistic, gammapdf

#Based on England data (CHESS and NHS England)
# I want a way to keep this as the "average" disease progression,
# but modify it such that old people have less favorable outcomes (as observed)
# But correspondingly I want people at lower risk to have more favorable outcome on average
const MODULE_DIR = dirname(@__FILE__)
const DATA_DIR = joinpath(MODULE_DIR, "..", "data")
# For calculations see data_cleaning_py.ipynb, calculations from NHS England dataset as per 05 Apr
relativeDeathRisk_given_COVID_by_age = [-0.99742186, -0.99728639, -0.98158438,
                                        -0.9830432 , -0.82983414, -0.84039294,
                                         0.10768979,  0.38432409,  5.13754904]
nTest, nIso, nHS, nAge = 4, 4, 8, 9
stateTensor = ones((nTest, nIso, nHS, nAge))
#ageRelativeDiseaseSeverity = np.array([-0.8, -0.6, -0.3, -0.3, -0.1, 0.1, 0.35, 0.4, 0.5])
# FIXED (above) - this is a guess, find data and fix
#ageRelativeRecoverySpeed = np.array([0.2]*5+[-0.1, -0.2, -0.3, -0.5])
#TODO - this is a guess, find data and fix
ageRelativeRecoverySpeed = zeros(9)# For now we make it same for everyone, makes calculations easier
# For calculations see data_cleaning_py.ipynb, calculations from NHS England dataset as per 05 Apr
caseFatalityRatioHospital_given_COVID_by_age = [0.00856164, 0.03768844, 0.02321319,
            0.04282494, 0.07512237, 0.12550367, 0.167096  , 0.37953452, 0.45757006]

agePopulationTotal = 1000*[8044.056, 7642.473, 8558.707, 9295.024, 8604.251,
                                      9173.465, 7286.777, 5830.635, 3450.616]

function _agePopulationRatio(agePopulationTotal)
    agePopulationTotal *= 55.98/66.27
    return agePopulationTotal/sum(agePopulationTotal)
end

function einsum(str, a, b)
    if str=="ijl,j->i"
        return _einsum1(a, b)
    elseif str=="ijk,j->ik"
        return _einsum2(a, b)
    elseif  str=="ijkl,j->i"
        return _einsum3(a, b)
    elseif  str=="ijkl,ijklmnp->imnp"
        return _einsum5(a, b)
    end
end

function einsum(str, a)
   if str=="ijlml->ijlm"
      return _einsum4(a)
  elseif str=="ijkj->ijk"
      return _einsum6(a)
  elseif str=="iklkl->ikl"
      return _einsum7(a)
  elseif str=="ijljl->ijl"
      return _einsum7(a)
  elseif str=="ijkljkm->ijklm"
      return _einsum8(a)
  elseif str=="ijkljkl->ijkl"
      return _einsum9(a)
  elseif str=="...jkl->..."
      return _einsum10(a)
  end
end

function _einsum1(a, b) #'ijl,j->i'
    i,_,j = size(a)
    p = zeros(i,j)
    for i=1:i, j=1:j
		p[i,j] += dot(a[i,:,j], b)
    end
    return sum(p, dims=1)
end

function _einsum2(a, b) #'ijk,j->ik'
    i,_,j = size(a)
    p = zeros(i,j)
    for i=1:i, j=1:j
		p[i,j] = dot(a[i,:,j], b)
    end
    return p
end

function _einsum3(a, b) #'ijkl,j->i'
    _,j,_,i = size(a)
    p = zeros(j,i)
    for i=1:i, j=1:j
		p[j,i] += sum(a[:,j,:,i]*b)
    end
    return sum(p, dims=1)
end

function _einsum4(a) #'ijlml->ijlm'
    l,m,l,j,i = size(a)
    p = zeros(m,l,j,i)
    for i=1:i, j=1:j, l=1:l
        p[:,l,j,i] = a[l,:,l,j,i]
    end
    return p
end
function _einsum5(a, b) #'ijkl,ijklmnp->imnp'
    p, n, m, l, k, j, i = size(b)
    dydt = zeros(p,n,m,i)
    for i=1:i, j=1:j, k=1:k, l=1:l, m=1:m, n=1:n, p=1:p
        dydt[p,n,m,i] += a[l,k,j,i] * b[p,n,m,l,k,j,i]
    end
    return dydt
end

function _einsum6(a) # 'ijkj->ijk'
    _,k,j,i = size(a)
    p = zeros(k,j,i)
    for j=1:j, i=1:i
        p[:,j,i] = a[j,:,j,i]
    end
    return p
end

function _einsum7(a) # 'ijljl->ijl' & 'iklkl->ikl'
    _,_,l,k,i = size(a)
    p = zeros(l,k,i)
    for i=1:i, k=1:k, l=1:l
        p[l,k,i] = a[l,k,l,k,i]
    end
    return p
end

function _einsum8(a) # 'ijkljkm->ijklm'
    m,_,_,l,k,j,i = size(a)
    p = zeros(m,l,k,j,i)
    for i=1:i, j=1:j, k=1:k, l=1:l, m=1:m
        p[m,l,k,j,i] = a[m,k,j,l,k,j,i]
    end
    return p
end

function _einsum9(a) # 'ijkljkl->ijkl'
    _,_,_,l,k,j,i = size(a)
    p = zeros(l,k,j,i)
    for i=1:i, j=1:j, k=1:k, l=1:l
        p[l,k,j,i] = a[l,k,j,l,k,j,i]
    end
    return p
end

function _einsum11(a, b) # for setting the values in einsum 10
	_,_,_,l,k,j,i = size(a)
    for i=1:i, j=1:j, k=1:k, l=1:l
        a[l,k,j,l,k,j,i] = b[l,k,j,i]
    end
    return a
end

function _einsum10(a) # '...jkl->...'
    return reshape(sum(a, dims=[1,2,3]),size(a)[4:end])
end

agePopulationRatio = _agePopulationRatio(agePopulationTotal)

function trFunc_diseaseProgression(
         ageRelativeRecoverySpeed::Array =
         ageRelativeRecoverySpeed,
         caseFatalityRatioHospital_given_COVID_by_age::Array=
         caseFatalityRatioHospital_given_COVID_by_age,
         nonsymptomatic_ratio::Float64 = 0.86,
                                   # number of days between measurable events
         infect_to_symptoms::Float64 = 5.0,
                                   #symptom_to_death = 16.;
         symptom_to_recovery::Float64= 10.0, # 20.5; #unrealiticly long for old people
         symptom_to_hospitalisation::Float64 = 5.76,
         hospitalisation_to_recovery::Float64 = 14.51,
         IgG_formation::Float64 = 15.0,
                                   # Age related parameters
                                   # for now we'll assume that all hospitalised cases are known (overall 23% of hospitalised COVID patients die. 9% overall case fatality ratio)
                                   # Unknown rates to estimate
         nonsymp_to_recovery::Float64 = 15.0,
         inverse_IS1_IS2::Float64 = 4.0;
         kwargs...)
    # Now we have all the information to build the age-aware multistage SIR model transition matrix
    # The full transition tensor is a sparse map from the Age x HealthState x isolation state to HealthState,
    # and thus is a 4th order tensor itself, representing a linear mapping
    # from "number of people aged A in health state B and isolation state C to health state D.
    #agePopulationRatioByTotal = _agePopulationRatio(agePopulationTotal)
    nAge, nHS, nIso = kwargs[:nAge], kwargs[:nHS], kwargs[:nIso]
    #relativeDeathRisk_given_COVID_by_age = [:relativeDeathRisk_given_COVID_by_age]
    trTensor_diseaseProgression = zeros((nHS, nIso, nHS, nAge))
    # Use basic parameters to regularise inputs
    E_IS1 = 1.0/infect_to_symptoms
    # Numbers nonsymptomatic is assumed to be 86% -> E->IN / E-IS1 = 0.86/0.14
    E_IN = 0.86/0.14 * E_IS1
    # Nonsymptomatic recovery
    IN_R1 = 1.0/nonsymp_to_recovery
    IS1_IS2  = 1.0/inverse_IS1_IS2
    IS2_R1 = 1.0/(symptom_to_recovery - inverse_IS1_IS2)
    R1_R2 = 1.0/IgG_formation

    # Disease progression matrix # TODO - calibrate (together with transmissionInfectionStage)
    # rows: from-state, cols: to-state (non-symmetric!)
    # - this represent excess deaths only, doesn't contain baseline deaths!

    # Calculate all non-serious cases that do not end up in hospitals.
    # Note that we only have reliable death data from hospitals (NHS England),
    # so we do not model people dieing outside hospitals
    diseaseProgBaseline = [
    # to: E,    IN,    IS1,   IS2,    R1,      R2,     D
          0.0  E_IN  E_IS1    0       0        0       0    # from E
          0    0      0       0     IN_R1      0       0    # from IN
          0    0      0    IS1_IS2    0        0       0    # from IS1
          0    0      0       0     IS2_R1     0       0    # from IS2
          0    0      0       0       0       R1_R2    0    # from R1
          0    0      0       0       0        0       0    # from R2
          0    0      0       0       0        0       0    # from D
    ]

    diseaseProgBaseline = transpose(diseaseProgBaseline)
    # TODO can be improved
    # vcat(fill.(x, v)...) ???
    ageAdjusted_diseaseProgBaseline = deepcopy(cat(repeat([diseaseProgBaseline],
                                                              nAge)..., dims=3))
    # Modify all death and R1 rates:
    for ii in range(1, stop = size(ageAdjusted_diseaseProgBaseline)[2])
        # Adjust death rate by age dependent disease severity  ??? check the key args
        ageAdjusted_diseaseProgBaseline[end, ii, :] = adjustRatesByAge_KeepAverageRate(
                                            ageAdjusted_diseaseProgBaseline[end, ii, 1],
                             agePopulationRatio=_agePopulationRatio(agePopulationTotal),
                              ageRelativeAdjustment=relativeDeathRisk_given_COVID_by_age
                              )
        # Adjust recovery rate by age dependent recovery speed
        ageAdjusted_diseaseProgBaseline[end - 2, ii, :] = adjustRatesByAge_KeepAverageRate(
                                            ageAdjusted_diseaseProgBaseline[end - 2, ii, 1],
                                                      agePopulationRatio=agePopulationRatio,
                                             ageRelativeAdjustment=ageRelativeRecoverySpeed
                                             )
    end
    ageAdjusted_diseaseProgBaseline_Hospital = deepcopy(ageAdjusted_diseaseProgBaseline)
    # Calculate hospitalisation based rates, for which we do have data. Hospitalisation can end up with deaths
    # Make sure that the ratio of recoveries in hospital honour the case fatality ratio appropriately
    # IS2 -> death
    ageAdjusted_diseaseProgBaseline_Hospital[end, 4, :] =
                     ageAdjusted_diseaseProgBaseline_Hospital[end - 2, 4, :] .* ( # IS2 -> recovery
                                  caseFatalityRatioHospital_given_COVID_by_age./(  # multiply by cfr / (1-cfr) to get correct rate towards death
                            1 .-  caseFatalityRatioHospital_given_COVID_by_age) )

    #TODO - time to death might be incorrect overall without an extra delay state, especially for young people
    # Non-hospitalised disease progression
    for i1 in [1, 2, 4]
        trTensor_diseaseProgression[2:end, i1, 2:end, :] = ageAdjusted_diseaseProgBaseline
    end
    # hospitalised disease progression
    trTensor_diseaseProgression[2:end, 3, 2:end, :] = ageAdjusted_diseaseProgBaseline_Hospital
    return trTensor_diseaseProgression
end


# Population (data from Imperial #13 ages.csv/UK)
#agePopulationTotal = 1000*[8044.056, 7642.473, 8558.707, 9295.024, 8604.251,
#                                      9173.465, 7286.777, 5830.635, 3450.616]
#agePopulationTotal = 1000.*pd.read_csv("https://raw.githubusercontent.com/ImperialCollegeLondon/covid19model/master/data/ages.csv").iloc[3].values[2:]

# Currently: let's work with england population only instead of full UK, as NHS England + CHESS data is much clearer than other regions
#agePopulationTotal *= 55.98/66.27 # (google england/uk population 2018, assuming age dist is similar)
#agePopulationRatio = agePopulationTotal/sum(agePopulationTotal)

agePopulationRatio = _agePopulationRatio(agePopulationTotal)

function adjustRatesByAge_KeepAverageRate(rate; agePopulationRatio=agePopulationRatio,
                                                ageRelativeAdjustment::Array=nothing,
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

# Getting Hospitalised
# -----------------------------------
#ageHospitalisationRateBaseline

# Larger data driver approaches, with age distribution, see data_cleaning_R.ipynb for details

ageHospitalisationRateBaseline = DataFrame(load(joinpath(DATA_DIR,
 "clean_hosp-epis-stat-admi-summ-rep-2015-16-rep_table_6.csv")))[:, end]
ageHospitalisationRecoveryRateBaseline = DataFrame(load(joinpath(DATA_DIR,
 "clean_10641_LoS_age_provider_suppressed.csv")))[:, end]
ageHospitalisationRecoveryRateBaseline = 1.0 ./ ageHospitalisationRecoveryRateBaseline

# Calculate initial hospitalisation (occupancy), that will be used to initialise the model
initBaselineHospitalOccupancyEquilibriumAgeRatio = ageHospitalisationRateBaseline ./
                                                    (ageHospitalisationRateBaseline +
                                                    ageHospitalisationRecoveryRateBaseline) #? check the calculations

# Take into account the NHS work-force in hospitals that for our purposes count
# as "hospitalised S" population, also unaffected by quarantine measures
ageNhsClinicalStaffPopulationRatio = DataFrame(load(joinpath(DATA_DIR,
      "clean_nhsclinicalstaff.csv")))[:,end]

# Extra rate of hospitalisation due to COVID-19 infection stages
# TODO - find / estimate data on this (unfortunately true rates are hard to get due to many unknown cases)
# Symptom to hospitalisation is 5.76 days on average (Imperial #8)

infToHospitalExtra = Array([1e-4, 1e-3, 2e-2, 1e-2])

# For calculations see data_cleaning_py.ipynb, calculations from CHESS dataset as per 05 Apr
relativeAdmissionRisk_given_COVID_by_age = [-0.94886625, -0.96332087, -0.86528671,
                                           -0.79828999, -0.61535305, -0.35214767,
                                            0.12567034,  0.85809052,  3.55950368]

riskOfAEAttandance_by_age = [0.41261361, 0.31560648, 0.3843979 ,
                             0.30475704, 0.26659415,0.25203475,
                             0.24970244, 0.31549102, 0.65181376]

function trFunc_HospitalAdmission(
         ageHospitalisationRateBaseline::Array=
         ageHospitalisationRateBaseline,
         infToHospitalExtra::Array=infToHospitalExtra,
         ageRelativeExtraAdmissionRiskToCovid::Array=
         relativeAdmissionRisk_given_COVID_by_age .*
         riskOfAEAttandance_by_age;
         kwargs...
         )
    nAge, nHS, nI = kwargs[:nAge], kwargs[:nHS], kwargs[:nI]

    trTensor_HospitalAdmission = zeros((nHS, nAge))

    ageAdjusted_infToHospitalExtra = deepcopy(cat(repeat([infToHospitalExtra],
                                                             nAge)..., dims=2))
    for ii in range(1, stop = size(ageAdjusted_infToHospitalExtra)[1])
        ageAdjusted_infToHospitalExtra[ii, :] = adjustRatesByAge_KeepAverageRate(
                     infToHospitalExtra[ii],
                     ageRelativeAdjustment=ageRelativeExtraAdmissionRiskToCovid
                    )
    end
    # Add baseline hospitalisation to all non-dead states
    trTensor_HospitalAdmission[1:end-1, :] .+= reshape(ageHospitalisationRateBaseline,
                                            (1, size(ageHospitalisationRateBaseline)...))
    # Add COVID-caused hospitalisation to all infeted states
    #(TODO: This is a summation fo rates for independent processes, should be correct, but check)
    trTensor_HospitalAdmission[2:(nI+1), :] .+= ageAdjusted_infToHospitalExtra
    return trTensor_HospitalAdmission
end


function trFunc_HospitalDischarge(
    ageHospitalisationRecoveryRateBaseline::Array=
    ageHospitalisationRecoveryRateBaseline,
    dischargeDueToCovidRateMultiplier::Float64=3.0;
    kwargs...
    )
    nAge, nHS = kwargs[:nAge], kwargs[:nHS]
    trTensor_HospitalDischarge = zeros((nHS, nAge))
    # Baseline discharges apply to all non-symptomatic patients (TODO: take into account testing state!)
    trTensor_HospitalDischarge[1:3, :] .+= transpose(
                                     ageHospitalisationRecoveryRateBaseline)

    # No discharges for COVID symptomatic people from the hospital until they recover
    # TODO - check with health experts if this is correct assumption; probably also depends on testing state
    trTensor_HospitalDischarge[4:5, :] .= 0.0
    trTensor_HospitalDischarge[6:7, :] .= dischargeDueToCovidRateMultiplier .*
                            transpose(ageHospitalisationRecoveryRateBaseline)
    return trTensor_HospitalDischarge
end


# Overall new infections include within quarantine and hospital infections
# ------------------------------------------------------------------------

ageSocialMixingBaseline = DataFrame(load(joinpath(DATA_DIR,
"socialcontactdata_UK_Mossong2008_social_contact_matrix.csv")))[:,2:end]
ageSocialMixingBaseline = convert(Matrix, ageSocialMixingBaseline)
ageSocialMixingBaseline = (ageSocialMixingBaseline.+
                           transpose(ageSocialMixingBaseline))/2.0
ageSocialMixingBaseline = ageSocialMixingBaseline
ageSocialMixingDistancing = DataFrame(load(joinpath(DATA_DIR,
"socialcontactdata_UK_Mossong2008_social_contact_matrix_with_distancing.csv")))[:,2:end]
ageSocialMixingDistancing = convert(Matrix, ageSocialMixingDistancing)
ageSocialMixingDistancing = (ageSocialMixingDistancing.+
                             transpose(ageSocialMixingDistancing))/2.0
ageSocialMixingDistancing = ageSocialMixingDistancing

ageSocialMixingIsolation = fill(0.0, size(ageSocialMixingBaseline))

elevatedMixingRatioInHospital = 3.0

withinHospitalSocialMixing = elevatedMixingRatioInHospital *
                           sum(ageSocialMixingBaseline * agePopulationRatio)

transmissionInfectionStage = [0.001, 0.1, 0.6, 0.5]
"""
All new infections, given infected people on all different isolation states (normal, home, hospital)
We use the following assumptions:

- Infectiousness only depends on infection stage, not age or location

- Hospitalised people are assumed to only mix with other hospitalised people (this includes staff!),
in a non-age-dependent manner: withinHospitalSocialMixing

If policySocialDistancing is True
- Non- and home-isolated people mix with non- and home isolated via ageSocialMixingDistancing (averaging interactions)

If policySocialDistancing is False, we assume home-isolation is taken more seriously, but with little effect on non-isolated people
- Non-isolated people mix with each other via ageSocialMixingBaseline, and with home-isolated people via ageSocialMixingIsolation
- Home-isolated people do not mix with each other

This separation will help disentangle the effects of simply a blanket lessening of social distancing
(keeping the policy True but with less effective ageSocialMixingDistancing matrix),
vs case isolation (policy = False, but with serious ageSocialMixingIsolation)
"""
function trFunc_newInfections_Complete(
         stateTensor,
         policySocialDistancing::Bool, # True / False, no default because it's important to know which one we use at any moment!
         policyImmunityPassports::Bool, # True / False, no default because it's important to know which one we use at any moment!
         ageSocialMixingBaseline::Array=
         ageSocialMixingBaseline,
         ageSocialMixingDistancing::Array=
         ageSocialMixingDistancing,
         ageSocialMixingIsolation::Array=
         ageSocialMixingIsolation,
         withinHospitalSocialMixing::Float64=
         withinHospitalSocialMixing,
         transmissionInfectionStage::Array=
         transmissionInfectionStage;
         kwargs...
         )

    nTest, nIso, nHS, nAge, nI = kwargs[:nTest], kwargs[:nIso], kwargs[:nHS],
                                 kwargs[:nAge] , kwargs[:nI]
    ageIsoContractionRate = zeros((nTest, nIso, nAge))
    # Add non-hospital infections
    #--------------------------------
    curNonIsolatedSocialMixing = policySocialDistancing ? ageSocialMixingDistancing : ageSocialMixingBaseline
    # Add baseline interactions only between non-isolated people
    for k1 in [1, 4]
        for k2 in [1, 4]
            ageIsoContractionRate[:,k1,:] .+= reshape(
                  ( einsum("ijl,j->i",
                    stateTensor[:,k2,2:(nI+1),:], transmissionInfectionStage) * # all infected in non-isolation
                    curNonIsolatedSocialMixing
                  ),
                (1, size(curNonIsolatedSocialMixing)[1]...)
            )
        end
    end
    if policyImmunityPassports
        # If the immunity passports policy is on, everyone who tested antibody positive, can roam freely
        # Therefore replace the interactions between people with testingState = 2 with ageSocialMixingBaseline
        # we do this by using the distributive property of matrix multiplication, and adding extra interactions
        # "ageSocialMixingBaseline"-"curNonIsolatedSocialMixing" with each other (this is zero if no social distancing!)
        # TODO - this is a bit hacky?, but probably correct - double check though!
        for k1 in [1, 4]
            for k2 in [1, 4]
                ageIsoContractionRate[3:end,k1,:] .+=
                         einsum("ijk,j->ik",
                         stateTensor[3:end,k2,2:(nI+1),:], transmissionInfectionStage)* # all infected in non-isolation
                         (ageSocialMixingBaseline.-curNonIsolatedSocialMixing)
            end
        end
    end
    # Add isolation interactions only between isolated and non-isolated people
    # non-isolated contracting it from isolated
    for k1 in [1, 4]
        ageIsoContractionRate[:,k1,:] .+= reshape(
              (einsum("ijl,j->i",
               stateTensor[:,2,2:(nI+1),:], transmissionInfectionStage)* # all infected in isolation
               ageSocialMixingIsolation
              ),
            (1, size(ageSocialMixingIsolation )[1]...)
        )
    end
    # isolated contracting it from non-isolated
    for k1 in [1, 4]
        ageIsoContractionRate[:,1,:] .+= reshape(
               (einsum("ijl,j->i",
                stateTensor[:,k1,2:(nI+1),:], transmissionInfectionStage)* # all infected in non-hospital, non-isolation
                ageSocialMixingIsolation
               ),
            (1, size(ageSocialMixingIsolation )[1]...)
        )
    end
        # isolated cannot contracting it from another isolated
    # Add in-hospital infections (of hospitalised patients, and staff)
    #--------------------------------
    # (TODO - within hospitals we probably want to take into effect the testing state;
    #      tested people are better isolated and there's less mixing)
    ageIsoContractionRate[:,3:end,:] .+= reshape(
                  withinHospitalSocialMixing *
                  einsum("ijkl,j->i",
                 stateTensor[:,3:end,2:(nI+1),:], transmissionInfectionStage), # all infected in hospital (sick or working)
        (1, 1, size(stateTensor)[end]...)
        )
    return ageIsoContractionRate/sum(stateTensor) # Normalise the rate by total population
end

function trFunc_travelInfectionRate_ageAdjusted(
	     t::Int64, # Time within simulation
	     travelMaxTime::Int64 = 200,
	     travelBaseRate::Float64 = 5e-4, # How many people normally travel back to the country per day # TODO - get data
	     travelDecline_mean::Float64 = 15.0,
	     travelDecline_slope::Float64 = 1.0,
	     travelInfection_peak::Float64 = 1e-1,
	     travelInfection_maxloc::Float64 = 10.0,
	     travelInfection_shape::Float64 = 2.0;
	     kwargs...
         )
	tmpTime = [0:1:travelMaxTime-1;]
	# nAge x T TODO get some realistic data on this
	travelAgeRateByTime = travelBaseRate .* (agePopulationRatio * transpose(1 .- map(logistic,
                        (tmpTime .- travelDecline_mean) ./ travelDecline_slope)))

    # 1 x T TODO get some realistic data on this, maybe make it age weighted
    _scale = travelInfection_maxloc / (travelInfection_shape-1)
    travelContractionRateByTime = map(x -> gammapdf(travelInfection_shape, 1.0,
                                            x/_scale), tmpTime)
    travelContractionRateByTime ./= _scale
    travelContractionRateByTime ./= max(travelContractionRateByTime...)
    travelContractionRateByTime .*= travelInfection_peak

    if t >= size(travelAgeRateByTime)[end]
        return zeros(size(travelAgeRateByTime)[1])
    else
        return travelAgeRateByTime[:,t+1] * travelContractionRateByTime[t+1]
    end
end

# Test parameters
# ---------------


# assumptions about practical (not theoretical, see discrapancy in PCR!) parameters of tests
# TODO - but particular data and references from lit (or estimates based on previous similar tests)

# TODO - MANUAL! - this function is VERY specific to current health state setup, and needs to be manually edited if number of health states change
function inpFunc_testSpecifications(
	PCR_FNR_I1_to_R2::Array = [ 0.9,  0.4, 0.15, 0.35, 0.5, 0.8],
    PCR_FPR::Float64 = 0.01,
    antigen_FNR_I1_to_R2::Array = [ 0.95, 0.6, 0.35, 0.45, 0.6, 0.9],
    antigen_FPR::Float64 = 0.1,
    antibody_FNR_I1_to_R2::Array = [0.99, 0.85, 0.8, 0.65, 0.3, 0.05],
    antibody_FPR_S_to_I4::Array = [0.05, 0.04, 0.03, 0.02, 0.01];
    kwargs...
    )

    nHS, nI, nR = kwargs[:nHS], kwargs[:nI], kwargs[:nR]

    testSpecifications = DataFrame()
    testSpecifications.Name = vcat(
                    ["PCR" for i in 1:nHS],
                    ["Antigen" for i in 1:nHS],
                    ["Antibody" for i in 1:nHS]
                    )
    testSpecifications.OutputTestState = vcat(
                    [1 for i in 1:2*nHS],
                    [2 for i in 1:nHS]
                    )
    testSpecifications.TruePosHealthState = vcat(
                    [[i for i in 1:nI] for j in 1:2*nHS],
                    [[i for i in nI+1:nI+nR] for j in 1:nHS]
                    )
    # In some health states some people are true negatives and some are true positives! (No, makes litte sense to use, just account for it in FPR? Only matters for test makers...)
    # testSpecifications['AmbiguousPosHealthState'] = [np.arange(nI+1, nI+nR+1)]*nHS + [np.arange(nI+1, nI+nR+1)]*nHS + [np.arange(1, nI+1)]*nHS # what information state does a pos test transition you to.

    testSpecifications.InputHealthState = vcat(0:nHS-1, 0:nHS-1, 0:nHS-1)

    # These numbers below are "defaults" illustrating the concept, but are modified by the inputs!!!

    testSpecifications.FalseNegativeRate = [ # ratio of positive (infected / immune) people missed by the test
        # For each health stage:
        #  S -> I1 (asymp) -> I2 (mild symp) -> I3 (symp, sick) -> I4 (symp, less sick) -> R1 / R2 (IgM, IgG avail) -> D

        # PCR
            0.,   0.9,            0.4,           0.15,                0.35,              0.5, 0.8,   0.,

        # Antigen
            0.,   0.95,           0.6,           0.35,                0.45,              0.6, 0.9,   0.,

        # Antibody
            0.,   0.99,           0.85,          0.8,                 0.65,              0.3, 0.05,  0.
    ]

    testSpecifications[2:7, :FalseNegativeRate] .= PCR_FNR_I1_to_R2
    testSpecifications[10:15, :FalseNegativeRate] .= antigen_FNR_I1_to_R2
    testSpecifications[18:23, :FalseNegativeRate] .= antibody_FNR_I1_to_R2


    testSpecifications.FalsePositiveRate = [ # ratio of negative (non-infected or not immune) people deemed positive by the test
        # PCR
        0.01, 0.,0.,0.,0., 0.01, 0.01, 0.,

        # Antigen
        0.1, 0.,0.,0.,0., 0.1, 0.1, 0.,

        # Antibody
        0.05, 0.04, 0.03, 0.02, 0.01, 0., 0., 0.
    ]

    testSpecifications[1, :FalsePositiveRate] = PCR_FPR
    testSpecifications[6:7, :FalsePositiveRate] = PCR_FPR
    testSpecifications[9, :FalsePositiveRate] = antigen_FPR
    testSpecifications[14:15, :FalsePositiveRate] = antigen_FPR
    testSpecifications[17:21, :FalsePositiveRate] .= antibody_FPR_S_to_I4

    return testSpecifications
end

function trFunc_testCapacity(
    realTime::Date, # time within simulation (day)
    # PCR capacity - initial
    testCapacity_pcr_phe_total::Float64 = 1e4,
    testCapacity_pcr_phe_inflexday::Date = Date("2020-03-25", "yyyy-mm-dd"),
    testCapacity_pcr_phe_inflexslope::Float64 = 5.0,

    # PCR capacity - increased
    testCapacity_pcr_country_total::Float64 = 1e5,
    testCapacity_pcr_country_inflexday::Date = Date("2020-04-25", "yyyy-mm-dd"),
    testCapacity_pcr_country_inflexslope::Float64 = 10.0,

    # Antibody / antigen capacity
    testCapacity_antibody_country_firstday::Date = Date("2020-04-25", "yyyy-mm-dd"),

    testCapacity_antibody_country_total::Float64 = 5e6,
    testCapacity_antibody_country_inflexday::Date = Date("2020-05-20", "yyyy-mm-dd"),
    testCapacity_antibody_country_inflexslope::Float64 = 20.0,

	testCapacity_antigenratio_country::Float64 = 0.7;

	kwargs...
	)

    # Returns a dictionary with test names and number available at day "t"
    outPCR = (
        #phe phase
        testCapacity_pcr_phe_total * logistic(Dates.value(Day(realTime-testCapacity_pcr_phe_inflexday))/testCapacity_pcr_phe_inflexslope)
        +
        #whole country phase
        testCapacity_pcr_country_total * logistic(Dates.value(Day(realTime-testCapacity_pcr_country_inflexday))/testCapacity_pcr_country_inflexslope)
    )


    if realTime<testCapacity_antibody_country_firstday
        outAntiTotal = 0.0
    else
        outAntiTotal = (
            testCapacity_antibody_country_total * logistic(Dates.value(Day(realTime-testCapacity_antibody_country_inflexday))/testCapacity_antibody_country_inflexslope)
		)
	end

    return Dict([
        ("PCR", outPCR),
        ("Antigen", outAntiTotal*testCapacity_antigenratio_country),
        ("Antibody", outAntiTotal*(1-testCapacity_antigenratio_country))
	]) # Tuples can be used instead (using dictionary to make it identical to python code
end
