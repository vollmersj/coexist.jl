# Packages
# Basic packages
import numpy as np

from scipy import integrate, stats, spatial
from scipy.special import expit, binom

import pandas as pd
import xlrd # help read excel files directly from source into pandas

import copy
import warnings

# Building parameter/computation graph
import inspect
from collections import OrderedDict

# OS/filesystem tools
import time
from datetime import datetime
import random
import string
import os
import shutil
import sys
import cloudpickle

# Distributed computing tools
import dask
import distributed
from dask.distributed import Client
from dask.distributed import as_completed
import itertools


# State Dimensions
# Health states (S, E and D are fixed to 1 dimension)
nI_symp = 2 # number of sympyomatic infected states
nI = 2+nI_symp # number of total infected states (disease stages), the +2 are Exposed and I_nonsymptomatic
nR = 2 # number of recovery states (antibody development post-disease, IgM and IgG are two stages)
nHS = 2+nI+nR # number of total health states, the +2: S, D are suspectible and dead

# Age groups (risk groups)
nAge = 9 # In accordance w Imperial #13 report (0-9, 10-19, ... 70-79, 80+)

# Isolation states
nIso = 4 # None/distancing, Case isolation, Hospitalised, Hospital staff

# Testing states
nTest = 4 # untested/negative, Virus positive, Antibody positive, Both positive


stateTensor = np.ones((nAge, nHS, nIso, nTest))


# Population (data from Imperial #13 ages.csv/UK)
agePopulationTotal = 1000.*np.array([8044.056,7642.473,8558.707,9295.024,8604.251,9173.465,7286.777,5830.635,3450.616])
#agePopulationTotal = 1000.*pd.read_csv("https://raw.githubusercontent.com/ImperialCollegeLondon/covid19model/master/data/ages.csv").iloc[3].values[2:]
# Currently: let's work with england population only instead of full UK, as NHS England + CHESS data is much clearer than other regions
agePopulationTotal *= 55.98/66.27 # (google england/uk population 2018, assuming age dist is similar)
agePopulationRatio = agePopulationTotal/np.sum(agePopulationTotal)

# Helper function to adjust average rates to age-aware rates
def adjustRatesByAge_KeepAverageRate(rate, ageRelativeAdjustment, agePopulationRatio=agePopulationRatio, maxOutRate=10):
    """This is a helper function and wont be picked up as a model parameter!"""
    if rate == 0:
        return np.zeros_like(ageRelativeAdjustment)
    if rate >= maxOutRate:
        warnings.warn("covidTesting::adjustRatesByAge_KeepAverageRate Input rate {} > maxOutRate {}, returning input rates".format(rate, maxOutRate))
        return rate*np.ones_like(ageRelativeAdjustment)
    out = np.zeros_like(ageRelativeAdjustment)
    out[0] = maxOutRate+1 # just to start the while loop below
    while np.sum(out>=maxOutRate)>0:
        corrFactor = np.sum(agePopulationRatio/(1+ageRelativeAdjustment))
        out =  rate * (1+ageRelativeAdjustment) * corrFactor
        if np.sum(out>=maxOutRate)>0:
            warnings.warn("covidTesting::adjustRatesByAge_KeepAverageRate Adjusted rate larger than {} encountered, reducing ageAdjustment variance by 10%".format(maxOutRate))
            tmp_mean = np.mean(ageRelativeAdjustment)
            ageRelativeAdjustment = tmp_mean + np.sqrt(0.9)*(ageRelativeAdjustment-tmp_mean)
    return out

# For calculations see data_cleaning_py.ipynb, calculations from NHS England dataset as per 05 Apr
relativeDeathRisk_given_COVID_by_age = np.array([-0.99742186, -0.99728639, -0.98158438, -0.9830432 , -0.82983414,
       -0.84039294,  0.10768979,  0.38432409,  5.13754904])

#ageRelativeDiseaseSeverity = np.array([-0.8, -0.6, -0.3, -0.3, -0.1, 0.1, 0.35, 0.4, 0.5]) # FIXED (above) - this is a guess, find data and fix
#ageRelativeRecoverySpeed = np.array([0.2]*5+[-0.1, -0.2, -0.3, -0.5]) # TODO - this is a guess, find data and fix
ageRelativeRecoverySpeed = np.array([0.]*9) # For now we make it same for everyone, makes calculations easier

# For calculations see data_cleaning_py.ipynb, calculations from NHS England dataset as per 05 Apr
caseFatalityRatioHospital_given_COVID_by_age = np.array([0.00856164, 0.03768844, 0.02321319, 0.04282494, 0.07512237,
       0.12550367, 0.167096  , 0.37953452, 0.45757006])

def trFunc_diseaseProgression(
    # Basic parameters to adhere to
    nonsymptomatic_ratio = 0.86,

    # number of days between measurable events
    infect_to_symptoms = 5.,
    #symptom_to_death = 16.,
    symptom_to_recovery = 10., # 20.5, #unrealiticly long for old people
    symptom_to_hospitalisation = 5.76,
    hospitalisation_to_recovery = 14.51,
    IgG_formation = 15.,

    # Age related parameters
    # for now we'll assume that all hospitalised cases are known (overall 23% of hospitalised COVID patients die. 9% overall case fatality ratio)
    caseFatalityRatioHospital_given_COVID_by_age = caseFatalityRatioHospital_given_COVID_by_age,
    ageRelativeRecoverySpeed = ageRelativeRecoverySpeed,

    # Unknown rates to estimate
    nonsymp_to_recovery = 15.,
    inverse_IS1_IS2 = 4.,


    **kwargs
    ):
    # Now we have all the information to build the age-aware multistage SIR model transition matrix
    # The full transition tensor is a sparse map from the Age x HealthState x isolation state to HealthState,
        # and thus is a 4th order tensor itself, representing a linear mapping
        # from "number of people aged A in health state B and isolation state C to health state D.
    trTensor_diseaseProgression = np.zeros((nAge, nHS, nIso, nHS))
    # Use basic parameters to regularise inputs
    E_IS1 = 1./infect_to_symptoms
    # Numbers nonsymptomatic is assumed to be 86% -> E->IN / E-IS1 = 0.86/0.14
    E_IN = 0.86/0.14 * E_IS1

    # Nonsymptomatic recovery
    IN_R1 = 1./nonsymp_to_recovery

    IS1_IS2  = 1./inverse_IS1_IS2

    IS2_R1 = 1./(symptom_to_recovery-inverse_IS1_IS2)

    R1_R2 = 1./IgG_formation


    # Disease progression matrix # TODO - calibrate (together with transmissionInfectionStage)
    # rows: from-state, cols: to-state (non-symmetric!)
    # - this represent excess deaths only, doesn't contain baseline deaths!

    # Calculate all non-serious cases that do not end up in hospitals.
    # Note that we only have reliable death data from hospitals (NHS England), so we do not model people dieing outside hospitals
    diseaseProgBaseline = np.array([
    # to: E,   IN,   IS1,   IS2,    R1,   R2,   D
        [  0 , E_IN, E_IS1,    0,   0,     0,   0   ], # from E
        [  0,   0,     0,   0,    IN_R1,   0,   0   ], # from IN
        [  0 ,  0,     0, IS1_IS2,  0,     0,    0 ], # from IS1
        [  0 ,  0,     0,    0,  IS2_R1,   0,   0  ], # from IS2
        [  0 ,  0,     0,    0,    0,    R1_R2,  0   ], # from R1
        [  0 ,  0,     0,    0,    0,     0,   0   ], # from R2
        [  0 ,  0,     0,    0,    0,     0,   0   ] # from D
    ])

    ageAdjusted_diseaseProgBaseline = copy.deepcopy(np.repeat(diseaseProgBaseline[np.newaxis],nAge,axis=0))
    # Modify all death and R1 rates:
    for ii in range(ageAdjusted_diseaseProgBaseline.shape[1]):
        # Adjust death rate by age dependent disease severity
        ageAdjusted_diseaseProgBaseline[:,ii,-1] = adjustRatesByAge_KeepAverageRate(
            ageAdjusted_diseaseProgBaseline[0,ii,-1],
            ageRelativeAdjustment=relativeDeathRisk_given_COVID_by_age
        )

        # Adjust recovery rate by age dependent recovery speed
        ageAdjusted_diseaseProgBaseline[:,ii,-3] = adjustRatesByAge_KeepAverageRate(
            ageAdjusted_diseaseProgBaseline[0,ii,-3],
            ageRelativeAdjustment=ageRelativeRecoverySpeed,
            agePopulationRatio=agePopulationRatio
        )

    ageAdjusted_diseaseProgBaseline_Hospital = copy.deepcopy(ageAdjusted_diseaseProgBaseline)
    # Calculate hospitalisation based rates, for which we do have data. Hospitalisation can end up with deaths
    # Make sure that the ratio of recoveries in hospital honour the case fatality ratio appropriately
    # IS2 -> death
    ageAdjusted_diseaseProgBaseline_Hospital[:,3,-1] = (
        # IS2 -> recovery
        ageAdjusted_diseaseProgBaseline_Hospital[:,3,-3] * (
            # multiply by cfr / (1-cfr) to get correct rate towards death
            caseFatalityRatioHospital_given_COVID_by_age/(
                 1 -  caseFatalityRatioHospital_given_COVID_by_age)
        )
    )


    # TODO - time to death might be incorrect overall without an extra delay state, especially for young people

    # Non-hospitalised disease progression
    for i1 in [0,1,3]:
        trTensor_diseaseProgression[:,1:,i1,1:] = ageAdjusted_diseaseProgBaseline

    # hospitalised disease progression
    trTensor_diseaseProgression[:,1:,2,1:] = ageAdjusted_diseaseProgBaseline_Hospital


    return trTensor_diseaseProgression

# Larger data driver approaches, with age distribution, see data_cleaning_R.ipynb for details
ageHospitalisationRateBaseline = pd.read_csv('data/clean_hosp-epis-stat-admi-summ-rep-2015-16-rep_table_6.csv', sep=',').iloc[:,-1].values
ageHospitalisationRecoveryRateBaseline = 1./pd.read_csv('data/clean_10641_LoS_age_provider_suppressed.csv', sep=',').iloc[:,-1].values

# Calculate initial hospitalisation (occupancy), that will be used to initialise the model
initBaselineHospitalOccupancyEquilibriumAgeRatio = ageHospitalisationRateBaseline/(ageHospitalisationRateBaseline+ageHospitalisationRecoveryRateBaseline)


# Take into account the NHS work-force in hospitals that for our purposes count as "hospitalised S" population,
# also unaffected by quarantine measures
ageNhsClinicalStaffPopulationRatio = pd.read_csv('data/clean_nhsclinicalstaff.csv', sep=',').iloc[:,-1].values

# Extra rate of hospitalisation due to COVID-19 infection stages
# TODO - find / estimate data on this (unfortunately true rates are hard to get due to many unknown cases)
# Symptom to hospitalisation is 5.76 days on average (Imperial #8)

infToHospitalExtra = np.array([1e-4, 1e-3, 2e-2, 1e-2])

# We do know at least how age affects these risks:

# For calculations see data_cleaning_py.ipynb, calculations from CHESS dataset as per 05 Apr
relativeAdmissionRisk_given_COVID_by_age = np.array([-0.94886625, -0.96332087, -0.86528671, -0.79828999, -0.61535305,
       -0.35214767,  0.12567034,  0.85809052,  3.55950368])

riskOfAEAttandance_by_age = np.array([0.41261361, 0.31560648, 0.3843979 , 0.30475704, 0.26659415,
       0.25203475, 0.24970244, 0.31549102, 0.65181376])


# Build the transition tensor from any non-hospitalised state to a hospitalised state
# (being in home quarantine is assumed to affect only the infection probability [below], not the hospitalisation probability)
# caseIsolationHospitalisationRateAdjustment = 1.

# This function takes as input the number of people in given age and health state, and in any non-hospitalised state
# and returns the number of people staying in the same age and health state,
# but now hospitalised (the rest of people remain in whatever state they were in)

def trFunc_HospitalAdmission(
    ageHospitalisationRateBaseline = ageHospitalisationRateBaseline,
    infToHospitalExtra = infToHospitalExtra,
    ageRelativeExtraAdmissionRiskToCovid = relativeAdmissionRisk_given_COVID_by_age * riskOfAEAttandance_by_age,

    **kwargs
    ):

    # This tensor will pointwise multiply an nAge x nHS slice of the stateTensor
    trTensor_HospitalAdmission = np.zeros((nAge, nHS))

    ageAdjusted_infToHospitalExtra = copy.deepcopy(np.repeat(infToHospitalExtra[np.newaxis],nAge,axis=0))
    for ii in range(ageAdjusted_infToHospitalExtra.shape[1]):
        # Adjust death rate by age dependent disease severity
        ageAdjusted_infToHospitalExtra[:,ii] = adjustRatesByAge_KeepAverageRate(
            infToHospitalExtra[ii],
            ageRelativeAdjustment=ageRelativeExtraAdmissionRiskToCovid
        )

    # Add baseline hospitalisation to all non-dead states
    trTensor_HospitalAdmission[:,:-1] += np.expand_dims(ageHospitalisationRateBaseline,-1)

    # Add COVID-caused hospitalisation to all infected states (TODO: This is summation of rates for independent processes, should be correct, but check)
    trTensor_HospitalAdmission[:,1:(nI+1)] += ageAdjusted_infToHospitalExtra

    return trTensor_HospitalAdmission


def trFunc_HospitalDischarge(
    ageHospitalisationRecoveryRateBaseline = ageHospitalisationRecoveryRateBaseline,
    dischargeDueToCovidRateMultiplier = 3.,

    **kwargs
    ):

    trTensor_HospitalDischarge = np.zeros((nAge, nHS))

    # Baseline discharges apply to all non-symptomatic patients (TODO: take into account testing state!)
    trTensor_HospitalDischarge[:, :3] += ageHospitalisationRecoveryRateBaseline[:,np.newaxis]

    # No discharges for COVID symptomatic people from the hospital until they recover
    # TODO - check with health experts if this is correct assumption; probably also depends on testing state
    trTensor_HospitalDischarge[:, 3:5] = 0.
    trTensor_HospitalDischarge[:, 5:7] = dischargeDueToCovidRateMultiplier * ageHospitalisationRecoveryRateBaseline[:,np.newaxis]

    return trTensor_HospitalDischarge

ageSocialMixingBaseline = pd.read_csv('data/socialcontactdata_UK_Mossong2008_social_contact_matrix.csv', sep=',').iloc[:,1:].values


ageSocialMixingBaseline = (ageSocialMixingBaseline+ageSocialMixingBaseline.T)/2.

ageSocialMixingDistancing = pd.read_csv('data/socialcontactdata_UK_Mossong2008_social_contact_matrix_with_distancing.csv', sep=',').iloc[:,1:].values

ageSocialMixingDistancing = (ageSocialMixingDistancing+ageSocialMixingDistancing.T)/2.

ageSocialMixingIsolation = np.zeros_like(ageSocialMixingBaseline)

elevatedMixingRatioInHospital = 3.0

withinHospitalSocialMixing = elevatedMixingRatioInHospital * np.sum(np.dot(agePopulationRatio, ageSocialMixingBaseline))

transmissionInfectionStage = np.array([0.001, 0.1, 0.6, 0.5])

def trFunc_newInfections_Complete(
    stateTensor,
    policySocialDistancing, # True / False, no default because it's important to know which one we use at any moment!
    policyImmunityPassports, # True / False, no default because it's important to know which one we use at any moment!
    ageSocialMixingBaseline = ageSocialMixingBaseline,
    ageSocialMixingDistancing = ageSocialMixingDistancing,
    ageSocialMixingIsolation = ageSocialMixingIsolation,
    withinHospitalSocialMixing = withinHospitalSocialMixing,
    transmissionInfectionStage = transmissionInfectionStage,

    **kwargs
):

    ageIsoContractionRate = np.zeros((nAge, nIso, nTest))


    # Add non-hospital infections
    #--------------------------------

    curNonIsolatedSocialMixing = ageSocialMixingDistancing if policySocialDistancing else ageSocialMixingBaseline

    # Add baseline interactions only between non-isolated people
    for k1 in [0,3]:
        for k2 in [0,3]:
            ageIsoContractionRate[:,k1,:] += np.expand_dims(
                np.matmul(
                    curNonIsolatedSocialMixing,
                    np.einsum('ijl,j->i',
                        stateTensor[:,1:(nI+1),k2,:], transmissionInfectionStage) # all infected in non-isolation
                ),
                axis=1
            )

    if policyImmunityPassports:
        # If the immunity passports policy is on, everyone who tested antibody positive, can roam freely
        # Therefore replace the interactions between people with testingState = 2 with ageSocialMixingBaseline
        # we do this by using the distributive property of matrix multiplication, and adding extra interactions
        # "ageSocialMixingBaseline"-"curNonIsolatedSocialMixing" with each other (this is zero if no social distancing!)
        # TODO - this is a bit hacky?, but probably correct - double check though!
        for k1 in [0,3]:
            for k2 in [0,3]:
                ageIsoContractionRate[:,k1,2:] += np.matmul(
                        ageSocialMixingBaseline-curNonIsolatedSocialMixing,
                        np.einsum('ijk,j->ik',
                            stateTensor[:,1:(nI+1),k2,2:], transmissionInfectionStage) # all infected in non-isolation
                    )
    # Add isolation interactions only between isolated and non-isolated people
    # non-isolated contracting it from isolated
    for k1 in [0,3]:
        ageIsoContractionRate[:,k1,:] += np.expand_dims(
            np.matmul(
                ageSocialMixingIsolation,
                np.einsum('ijl,j->i',
                    stateTensor[:,1:(nI+1),1,:], transmissionInfectionStage) # all infected in isolation
            ),
            axis=1
        )
    # isolated contracting it from non-isolated
    for k1 in [0,3]:
        ageIsoContractionRate[:,1,:] += np.expand_dims(
            np.matmul(
                ageSocialMixingIsolation,
                np.einsum('ijl,j->i',
                    stateTensor[:,1:(nI+1),k1,:], transmissionInfectionStage) # all infected in non-hospital, non-isolation
            ),
            axis = 1
        )

        # isolated cannot contracting it from another isolated


    # Add in-hospital infections (of hospitalised patients, and staff)
    #--------------------------------
    # (TODO - within hospitals we probably want to take into effect the testing state;
    #      tested people are better isolated and there's less mixing)

    ageIsoContractionRate[:,2:,:] += np.expand_dims(
            withinHospitalSocialMixing *
            np.einsum('ijkl,j->i',
                stateTensor[:,1:(nI+1),2:,:], transmissionInfectionStage), # all infected in hospital (sick or working)
        axis = (1,2))


    return ageIsoContractionRate/np.sum(stateTensor) # Normalise the rate by total population
