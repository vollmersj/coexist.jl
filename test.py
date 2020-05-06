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


stateTensor = np.zeros((nAge, nHS, nIso, nTest))


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


    return np.transpose(trTensor_diseaseProgression)
