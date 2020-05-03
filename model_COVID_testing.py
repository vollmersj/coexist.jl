#!/usr/bin/env python
# coding: utf-8

# # COVID-19 model for policy makers in the United Kingdom

#  <p align="center">
#  <img src="images/dynamicalModel.png" width="70%">
#  </p>

# We use an extended [SEIR model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model) to capture available information both about the disease progression, as well as how accessible various disease states are by testing. Being tested might cause a transition in the <span style="display: inline-block;background-color:#FFFCCC">Testing State</span>; the rate of such a transition depends both on the <span style="display: inline-block;background-color:#D1E2FF">Health State</span> as well the parameters of the test used.
# 
# Due to age being both an extra risk factor as well as a potential way for decision makers to clearly apply policies to only parts of the total population, we include it directly in our model, and most transition rates are strongly influenced by <span style="display: inline-block;background-color:#FFD4D1">Age State</span>.
# 
# Finally, the main policy making tool as well as conundrum nowadays is the implementation of quarantining and social distancing in order to keep hospitals and medical staff under tolerable levels of pressure. We represent <span style="display: inline-block;background-color:#C2EDC0">Isolation States</span> to investigate the effects of various interventions on policy targets relating to hospitalisation rates and economic freedom, while describing the different health outcomes via the <span style="display: inline-block;background-color:#D1E2FF">Health States</span>.
# 

# ### Health states and Disease progression
# 
# Susceptible people may be Exposed to the virus by mixing with other infected people (E,A,I<sub>1</sub> and I<sub>2</sub>).
# 
# They may get through the infection Asymptomatic and eventually Recover, or become symptomatic and more Infectious, spreading the disease, and potentially Dying or Recovering.
# 
# Recovered people develop more effective antibodies against the virus, and are considered immune<sup>*</sup>.
# 
#     
# | State | Description | Testing |
# | ----- | ----------- | -------- |
# | S | Susceptible | Negative |
# | E | Exposed | Very weakly virus positive
# | A | Asymptomatic | Weakly virus positive
# | I<sub>1</sub> | Symptomatic early | Strongly virus positive
# | I<sub>2</sub> | Symptomatic late | Medium virus positive <br>Weakly IgM antibody positive
# | R<sub>1</sub> | Recovered early | IgM antibody positive
# | R<sub>2</sub> | Recovered late | IgM/IgG antibody positive
# | D | COVID-related death | May be virus or antibody positive
# 
# 
# <sub><sup>*</sup>We plan to consider partial / short-term immunity, see further discussion in [Continued research](#header_contd).</sub>

# # Code
# 
# The below consists of the implementation of the model described above, with extended comments including assumptions, sources of data and areas to improve upon

# In[1]:


# Use available data up until this day; cutoff is important due to more recent data being less complete.
CONST_DATA_CUTOFF_DATE = "20200414"


# # Packages and Helper functions
# 
# To preserve the single-notebook formulation, we include all packages used as well as subfunctions here
# 
# To skip to the start of model implementation, <a href="#modelImplementation">click here</a>!

# In[2]:


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


# In[3]:


# Regroup various age-group representations into our internal one, and vice versa
def regroup_by_age(
    inp, # first dimension is ages, others dont matter.
    fromAgeSplits, toAgeSplits, maxAge=100., maxAgeWeight = 5.):
    fromAgeSplits = np.concatenate([np.array([0]), fromAgeSplits, np.array([maxAge])]) # Add a zero at beginning for calculations
    toAgeSplits = np.concatenate([np.array([0]), toAgeSplits, np.array([maxAge])]) # Add inf at end for calculations
    def getOverlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))
    out = np.zeros((len(toAgeSplits)-1,)+inp.shape[1:])
    for from_ind in range(1, len(fromAgeSplits)):
        # Redistribute to the new bins by calculating how many years in from_ind-1:from_ind falls into each output bin
        cur_out_distribution = (
        [getOverlap(toAgeSplits[cur_to_ind-1:cur_to_ind+1],fromAgeSplits[from_ind-1:from_ind+1])  for cur_to_ind in range(1, len(toAgeSplits))]
        )
        
        if cur_out_distribution[-1] > 0:
            cur_out_distribution[-1] = maxAgeWeight # Define the relative number of ages if we have to distribute between second to last and last age groups

        cur_out_distribution = cur_out_distribution/np.sum(cur_out_distribution)
        
        for to_ind in range(len(out)):
            out[to_ind] += cur_out_distribution[to_ind] * inp[from_ind-1]
            
    return out


# PARAMETER DICTIONARIES AND TABLES
# -----------------------------------------------------------------------------------------

# Build the nested parameter/computation graph of a single function.
def build_paramDict(cur_func):
    """
    This function iterates through all inputs of a function, 
    and saves the default argument names and values into a dictionary.
    
    If any of the default arguments are functions themselves, then recursively (depth-first) adds an extra field to
    the dictionary, named <funcName + "_params">, that contains its inputs and arguments.
    
    The output of this function can then be passed as a "kwargs" object to the highest level function, 
    which will then pass the parameter values to the lower dictionary levels appropriately
    """
    
    paramDict = OrderedDict()
    
    allArgs = inspect.getfullargspec(cur_func)
    
    # Check if there are any default parameters, if no, just return empty dict
    if allArgs.defaults is None:
        return paramDict
    
    
    for argname, argval in zip(allArgs.args[-len(allArgs.defaults):], allArgs.defaults):
        # Save the default argument
        paramDict[argname] = argval
        # If the default argument is a function, inspect it for further 
        
        if callable(argval):
            # print(argname)
            paramDict[argname+"_params"] = build_paramDict(argval)

    return paramDict


# Do a mapping between dictionary and parameter table row and vice versa (for convenient use)

# Flatten the dictionary into a table with a single row (but many column):
def paramDict_toTable(paramDict):
    paramTable = pd.DataFrame()
    def paramDictRecurseIter(cur_table, cur_dict, preString):
        # Iterate through the dictionary to find all keys not ending in "_params", 
        # and add them to the table with name <preString + key>
        # 
        # If the key doesn end in "_params", then append the key to preString, in call this function on the value (that is a dict)
        for key, value in cur_dict.items():
            if key.endswith("_params"):
                paramDictRecurseIter(cur_table, value, preString+key+"_")
            else:
                paramTable[preString+key] = [value]
                
        # For the rare case where we want to keep an empty dictionary, the above for cycle doesn't keep it
        if len(cur_dict)==0:
            paramTable[preString] = [OrderedDict()]
                
        return cur_table
    
    return paramDictRecurseIter(paramTable, paramDict, preString="")


def paramTable_toDict(paramTable, defaultDict=None):
    # enable to pass a default dict (if paramTable is incomplete), in which we'll just add / overwrite the values
    paramDict = defaultDict if defaultDict is not None else OrderedDict() 
    def placeArgInDictRecurse(argName, argVal, cur_dict):
        # Find all "_params_" in the argName, and for each step more and more down in the dictionary
        strloc = argName.find("_params_")
        if strloc == -1:
            # We're at the correct level of dictionary
            cur_dict[argName] = argVal
            return cur_dict
        else:
            # step to the next level of dictionary
            nextKey = argName[:strloc+len("_params_")-1]
            nextArgName = argName[strloc+len("_params_"):]
            if not nextKey in cur_dict:
                cur_dict[nextKey] = OrderedDict()
            placeArgInDictRecurse(nextArgName, argVal, cur_dict[nextKey])
            
        return cur_dict
            
    for key in paramTable.columns:
        paramDict = placeArgInDictRecurse(key, paramTable.at[0,key], paramDict)
        
    return paramDict


# <a name="modelImplementation"></a>
# # Model implementation
# 
# The model is governed by these two main tensors:
# - State tensor: a 4th order tensor containing axes:
#     - Age groups
#     - Health states
#     - Isolation states
#     - Testing states
#     
#     In our extended state tensor formulation, we also keep track of not just people currently in each state, but also people newly arriving to each state, as a large number of available datasets refer to "new cases" or "new hospitalisations" each day, rather than current state occupancies normally represented by ODE solutions.   
#     
#     
# - Dynamic state transition rate tensor
#     - Rates that govern all allowed transitions between states
#     - Dynamically recomputed every iteration, based on number of infected, their current social mixing and number and types of tests available, amongst other variables.
#     - Briefly:
#         - No transition between age groups
#         - No transitions between testing states without performing a test
#         - No transitions into S or out of D and R_IgG health states
#     - Allowed transitions are as showcased in the model image above
#     - Represented by a 7th order, sparse tensor, containing all transitions except age (unimportant due to relatively short time scales compared to coarse age grouping)
#     

# ```
# NOTICE
# ------
# THE "MODEL IMPLEMENTATION" SECTION CONTAINS A LARGE NUMBER OF PARAMETER VALUES SET TO A DEFAULT VALUE.
# THESE ARE LARGELY INFORMED BY DATA, BUT NOT FIXED!
# THEY ARE VARIED DURING THE FITTING OF THE MODEL, ACCORDING TO HOW UNCERTAIN WE ARE IN THE PARAMETER
# ```
# 
# The priors are defined <a href="#defineEnsemblePriors">below the model</a>. Although many of our uncertain / weak assumptions are signalled by "TODO" comments, we feel that the overall conclusions would not be affected by finding better parameter values, especially given our fitting and exploration approach.
# 

# ### The state tensor

# In[4]:


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


# ### Transition rates (1 / day)
# 
# The full transition rate structure is an 8th order tensor, 
# mapping from any 4D state in the state tensor, to any other 4D state in the state tensor
# 
# However, many of these transitions are non-sensical (ie a 3 year old cannot suddenly become 72, or a dead person be infected again), therefore during the construction of the full model below, we fill in the rates on all "allowed" transitions.
# 
# We attempt to do so based on existing data either describing particular rates (like COVID-related hospitalisation),
# or data that helps constrain the product or ratios of multiple rates (such as the R0, or the case fatality ratio [noting this latter depends on testing policy and test availability]).
# 
# Further, to massively reduce the number of weakly constrained parameters, we will approximate many of the weakly correlated transition rates as rank 1 (uncorrelated) matrices. For example the rate of hospitalisation for a patient at a given age and stage of infection will be computed as a product of two indepent rates, one based purely on the age (older people are generally more at risk of hospitalisation), and the other purely on how far the patient has progressed into the disease. This allows us to estimate more of required parameters from available published data.
# 
# There of course still is a lot of uncertainty about how the virus behaves, and all of the data that we use is likely uncomplete and noisy. In order to better represent the things we do not know, we use advanced machine learning techniques, and investigate many possible scenarios (settings of parameters) and for all analyses we retain all plausible scenarios (various parameter settings that explain the available data well enough).
# 
# Any policies we suggest for the near future are investigated for all plausible scenarios, such that decision makers know how likely each one will work as expected in these uncertain times. We further note that as we progress further into the pandemic, the number of plausible scenarios reduces more and more, enabling us to see the way out clearer and clearer.
# 
# 

# In[5]:


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
            #print(out)
            tmp_mean = np.mean(ageRelativeAdjustment)
            ageRelativeAdjustment = tmp_mean + np.sqrt(0.9)*(ageRelativeAdjustment-tmp_mean)
    return out


# ## Getting infected

# In[7]:


# Getting infected
# ------------------
# We wish to calibrate overall infection rates to match 
# - previous R0 estimates, 
# - available age-attack-ratios,



# Age-dependent mixing affects state transition S -> I1 (data available eg Imperial #13 report)
# The mixing-related data is nowhere to be found!
# This is an Age x Age symmetric matrix, showing which groups mix with which other ones.
# Data from DOI: 10.1097/EDE.0000000000001047 via http://www.socialcontactdata.org/tools/ interactive tool in data folder
# This is assumed to be contacts per day (but may need to be time-rescaled)
ageSocialMixingBaseline = pd.read_csv('data/socialcontactdata_UK_Mossong2008_social_contact_matrix.csv', sep=',').iloc[:,1:].values
ageSocialMixingDistancing = pd.read_csv('data/socialcontactdata_UK_Mossong2008_social_contact_matrix_with_distancing.csv', sep=',').iloc[:,1:].values

# Symmetrise these matrices (not sure why they aren't symmetric)
ageSocialMixingBaseline = (ageSocialMixingBaseline+ageSocialMixingBaseline.T)/2.
ageSocialMixingDistancing = (ageSocialMixingDistancing+ageSocialMixingDistancing.T)/2.

# For simplicity, let's assume scenario of perfect isolation in state-issued home quarantine, see commented below for alternatives
ageSocialMixingIsolation = np.zeros_like(ageSocialMixingBaseline) 
#isolationEffectComparedToDistancing = 3. # TODO - find better numbers for proper isolation mixing estimation!
#ageSocialMixingIsolation = ageSocialMixingBaseline/(isolationEffectComparedToDistancing * np.mean(ageSocialMixingBaseline/ageSocialMixingDistancing))



# For the S->I1 transition we also need a product mapping, 
# as the AS->AI1 rate is variable and depend on all AI via social mixing (ages) and transmission rates (I stages)
# this vector is nI long only, calibrated together with other variables to reproduce overall R0
# These numbers should represent rate of transmission given contact [will be multiplied by social mixing matrices]
transmissionInfectionStage = np.array([0.001, 0.1, 0.6, 0.5]) # We vary this during model fitting



# In[8]:


# Getting Infected in the Hospital
# ---------------------------------------

# The general experience is that infections spread faster in a hospital environment, 
# we capture this intuition with an age-independent but increased "social Mixing" amongst hospital patients and staff

# TODO - This requires further data-driven calibration!

# Capture S->I1 within hospital, given the number of total infected inside hospitals
elevatedMixingRatioInHospital = 3. # TODO - fact check this number, atm just set based on intuition
# Called "Nosocomial viral infection", some data: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5414085/
# HAP: hospital acquired pneumonia, apparently quite common
# more data: https://cmr.asm.org/content/14/3/528
# on covid-19: https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(20)30073-6/fulltext "Nosocomial infection risk among health-care workers and patients has been identified as a research gap to be prioritised in the next few months by WHO."
withinHospitalSocialMixing = elevatedMixingRatioInHospital * np.sum(np.dot(agePopulationRatio, ageSocialMixingBaseline))


# In[9]:


# Also add new infected from travelling based on time-within-simulation

# TODO - get real travel data to make these numbers more realistic. For now based on the following assumptions:
# - people's age distribution in travel is square of the usual age distribution
# - travel rates declined from a base rate as a sigmoid due to border closures, with given mean and slope
# - infection rates due to travel are modelled as a gamma pdf over time, with given peak value, loc, and scale parameter
def trFunc_travelInfectionRate_ageAdjusted(
    t, # Time (int, in days) within simulation
    
    travelMaxTime = 200,
    travelBaseRate = 5e-4, # How many people normally travel back to the country per day # TODO - get data
    travelDecline_mean = 15.,
    travelDecline_slope = 1.,
    
    travelInfection_peak = 1e-1,
    travelInfection_maxloc = 10.,
    travelInfection_shape = 2.,

    **kwargs
):
    
    tmpTime = np.arange(travelMaxTime)
    # nAge x T TODO get some realistic data on this
    travelAgeRateByTime = travelBaseRate * np.outer(agePopulationRatio, 1-expit((tmpTime-travelDecline_mean)/travelDecline_slope))
    
    
    # 1 x T TODO get some realistic data on this, maybe make it age weighted
    travelContractionRateByTime = stats.gamma.pdf(tmpTime, a=travelInfection_shape, loc=0., scale=travelInfection_maxloc/(travelInfection_shape-1))
    travelContractionRateByTime = travelInfection_peak*travelContractionRateByTime/np.max(travelContractionRateByTime)

    
    if t >= travelAgeRateByTime.shape[-1]:
        return np.zeros(travelAgeRateByTime.shape[0])
    else:
        return travelAgeRateByTime[:,int(t)] * travelContractionRateByTime[int(t)]


# In[10]:


# Overall new infections include within quarantine and hospital infections
# ------------------------------------------------------------------------

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
    


# ## Hospitalisation and hospital staff
# 
# Disease progression in severe cases naturally leads to hospitalisation before death. 
# One of the important policy questions we wish to estimate is how many people at any one time would require a hospital bed during the treatment of their disease.
# 
# Hospitalisation is generally a simple situation modeling wise. People with either symptomatic infection (I3-...In states), or for other sicknesses (baseline hospitalisation) end up in hospital. People in S health state may return to non-hospitalised S state, however people in (informed, see later) I state generally remain in hospital until they recovered or dead.
# 
# Home quarantine / social distancing is a different situation. Unlike other reports, here we do not (yet) wish to disentagle the effects of individual quarantine operations (school closures, working from home, social distancing), but rather investigate the effects of current full lockdown (coming into effect on 24 March in the UK), versus testing-based informed individual quarantining.
# 
# Numerically:
# 
# - People in home isolation change their social mixing patterns. The overall social mixing matrix between people in no isolation and home has been estimated via the http://www.socialcontactdata.org/tools/ software, see details in the data_cleaning notebook, this will determine the S->I transition overall.
# 
# - People in hospitals (sick) dramatically reduce their contacts outside the hospital, but increase the chance of transmission within the hospitalised community. For the purpose of this simulation, hospital staff will also in effect be suspecitble to higher risk of infection due to "hospitalised" patients and they also keep their normal interaction. 
# 
# - Reported numbers regarding pressure on the health system will report both COVID-19 and non-COVID-19 patients
# 

# In[11]:


# Getting Hospitalised
# ---------------------------------------

# Describe the transitions to-from hospitals 
# Note that this implementation will assume that hospitalisation takes an extra day,
# due to the discrete nature of the simulation, might need to be re-thought. 
# -> if simulation of a single day is done in multiple steps (first disease progression, then potential hospitalisation),
#.    then this problem is avoided. Can do the same with testing.

# Further we assume that hospitalisation does not change health state, 
# but if happens in a non-S state, then it persists until R1 or D 
# (this may need to be relaxed for early untested I states, where the hospitalisation is not COVID-related).

# Hospitalisation mainly depends on disease severity
# Baseline hospitalisation rate (Data from Scotland: https://www.isdscotland.org/Health-Topics/Hospital-Care/Publications/Acute-Hospital-Publication/data-summary/)
#hospitalisationRateBaseline = 261278./(91.*(5.425*10**6)) # hospitalisation / (period in days * population) -> frac of pop hospitalised per day
#hospitalisationRecoveryRateBaseline = 1./4.2 # inverse of mean length of stay in days

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

# Recovery rates (hospital discharge)
# ------------------------------------

# Higher-than-normal discharge rate for people who recovered (as they were likely to be in hospital mostly due to the virus)
# TODO - check with health experts if this is correct assumption; probably also depends on testing state

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




# TODO - think of how the latest changes (no prenatal care, no elective surgeries, etc) changed the default hospitalisation rate
#trTensor_HospitalAdmission[:,5]
    
# TODO!!! - adjust disease progression transitions so that it shifts direct death probabilities to hospitalised death probabilities    


# ## Disease progression
# 
#  - assumed to be strictly age and infection stage dependent distributions (progression rates), doesn't depend on other people
#  - distinct states represent progression, not necessarly time, but only forward progression is allowed, and the inverse of rates represent average number of days in progression
#  - there is a small chance of COVID death from every state, but we assume death is most often preceeded by hospitalisation
#  - there is a chance of recovery (and becoming immunised) from every state
# 
# We wish to calibrate these disease progression probabilities to adhere to observed data / earlier models
# - serial interval distribution suggests time-to-transmission of Gamma(6.5 days, 0.62) MODEL [Imperial #13]
# Symptom progression (All params with relatively wide confidence intervals)
# - infect-to-symptom onset is assumed 5 days mean MODEL [AceMod, https://arxiv.org/pdf/2003.10218.pdf]
# - symptom-to-death is 16 days DATA_WEAK [Imperial #8]
# - symptom-to-discharge is 20.5 days DATA_WEAK [Imperial #8]
# - symptom-to-hospitalisation is 5.76 days DATA_WEAK [Imperial #8]
# - hospitalisation-to-recovery is 14.51 days DATA_WEAK [Imperial #8]
# all the above in Imperial #8 is largely age dependent. Raw data available in data/ImperialReport8_subset_international_cases_2020_03_11.csv
# 

# In[12]:


# Based on England data (CHESS and NHS England)



# I want a way to keep this as the "average" disease progression, but modify it such that old people have less favorable outcomes (as observed)
# But correspondingly I want people at lower risk to have more favorable outcome on average

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


# ## Testing
# 
# In this section we describe multiple types of tests (PCR, antigen and antibody), and estimate their sensitivity and specificity in different health stages. These are thought to be the same for patients of all ages, and isolation states at this time.
# 
# We then model the transitions to other testing states, which are largely policy-based.
# 
# To model the current data (up to 03 April 2020):
# - only PCR tests have been done in the UK
# - PCR tests are thought to be carried out almost exclusively on symptomatic patients, to determine if their symptoms are caused by SARS-CoV2 or some other infection (this helps us determine the baseline ILI symptoms in practice, to predict true negative rates of the tests given the SARS-infected vs non-SARS-infected (but ILI symptom producing) populations).
# 
# One aim of this complete model is to enable policy makers to make decisions now, based on predicted test availability in the future, therefore most testing-related concerns will be hypotheticals. That said, we aim to accurately model the tests' capabilities based on extensive literature research, and also aim to bring stable policy-level outcomes despite the actual numbers may be inaccurate.
# 
# Two important questions answered by integrating this section into the epidemiology model above will be:
#     
#     1. In what ratio we should produce antibody and antigen lateral flow immunoassay tests? They require the same production capabilities and reagents, there is a question ideally suited to the policy making level
#     
#     2. At what level of testing capabilities (PCR, antigen and antibody) can the country lessen the complete lockdown, without risking lives or overburdening the NHS?
#     
#     
#     
# API:
# 
# - trFunc_testing(stateTensor, t, policyFunc, testSpecifications, trFunc_testCapacity):
#     - This is the main transition rate function, it returns transition rates from and to all testing states
# 
# - policyFunc
#     - Returns a testing policy about what states are tested with how many of which test
#     
# - testSpecifications
#     - Details the FPR/FNR of individual tests given the health state
#     
# - trFunc_testCapacity(t)
#     - outputs how many tests are available at time t of the different test types modelled

# In[13]:


# Test parameters
# ---------------


# assumptions about practical (not theoretical, see discrapancy in PCR!) parameters of tests
# TODO - but particular data and references from lit (or estimates based on previous similar tests)

# TODO - MANUAL! - this function is VERY specific to current health state setup, and needs to be manually edited if number of health states change
def inpFunc_testSpecifications(
    PCR_FNR_I1_to_R2 = np.array([ 0.9,  0.4, 0.15, 0.35, 0.5, 0.8]),
    PCR_FPR = 0.01,
    antigen_FNR_I1_to_R2 = np.array([ 0.95, 0.6, 0.35, 0.45, 0.6, 0.9]),
    antigen_FPR = 0.1,
    antibody_FNR_I1_to_R2 = np.array([0.99, 0.85, 0.8, 0.65, 0.3, 0.05]),
    antibody_FPR_S_to_I4 =  np.array([0.05, 0.04, 0.03, 0.02, 0.01])
    ):
    
    
    testSpecifications = pd.DataFrame(
    columns=["Name"],#, "Infection stage"],#, "Sensitivity", "Specificity"],
    
    data = (
        ["PCR"] * nHS +
        ["Antigen"] * (nHS) +
        ["Antibody"] * (nHS))
    )

    testSpecifications['OutputTestState'] = [1]*nHS + [1]*nHS + [2]*nHS # what information state does a pos test transition you to.

    testSpecifications['TruePosHealthState'] = [np.arange(1,nI+1)]*nHS + [np.arange(1,nI+1)]*nHS + [np.arange(nI+1,nI+nR+1)]*nHS # what information state does a pos test transition you to.

    # In some health states some people are true negatives and some are true positives! (No, makes litte sense to use, just account for it in FPR? Only matters for test makers...)
    # testSpecifications['AmbiguousPosHealthState'] = [np.arange(nI+1, nI+nR+1)]*nHS + [np.arange(nI+1, nI+nR+1)]*nHS + [np.arange(1, nI+1)]*nHS # what information state does a pos test transition you to.

    testSpecifications['InputHealthState'] = list(np.tile(range(nHS),3))

    # These numbers below are "defaults" illustrating the concept, but are modified by the inputs!!!
    
    testSpecifications['FalseNegativeRate'] = [ # ratio of positive (infected / immune) people missed by the test
        # For each health stage:
        #  S -> I1 (asymp) -> I2 (mild symp) -> I3 (symp, sick) -> I4 (symp, less sick) -> R1 / R2 (IgM, IgG avail) -> D

        # PCR
            0.,   0.9,            0.4,           0.15,                0.35,              0.5, 0.8,   0.,

        # Antigen
            0.,   0.95,           0.6,           0.35,                0.45,              0.6, 0.9,   0.,

        # Antibody
            0.,   0.99,           0.85,          0.8,                 0.65,              0.3, 0.05,  0.
    ]
    
    
    testSpecifications.loc[1:6,'FalseNegativeRate'] = PCR_FNR_I1_to_R2
    testSpecifications.loc[9:14,'FalseNegativeRate'] = antigen_FNR_I1_to_R2
    testSpecifications.loc[17:22,'FalseNegativeRate'] = antibody_FNR_I1_to_R2
    
    

    testSpecifications['FalsePositiveRate'] = [ # ratio of negative (non-infected or not immune) people deemed positive by the test
        # PCR
        0.01, 0.,0.,0.,0., 0.01, 0.01, 0.,

        # Antigen
        0.1, 0.,0.,0.,0., 0.1, 0.1, 0.,

        # Antibody
        0.05, 0.04, 0.03, 0.02, 0.01, 0., 0., 0.        
    ]
    
    testSpecifications.loc[0,'FalsePositiveRate'] = PCR_FPR
    testSpecifications.loc[5:6,'FalsePositiveRate'] = PCR_FPR
    testSpecifications.loc[8,'FalsePositiveRate'] = antigen_FPR
    testSpecifications.loc[13:14,'FalsePositiveRate'] = antigen_FPR
    testSpecifications.loc[16:20,'FalsePositiveRate'] = antibody_FPR_S_to_I4
    
    return testSpecifications


# In[14]:


inpFunc_testSpecifications()


# In[15]:


# TODO - think if we should introdce an "autopsy test" posthumously, categorising people as tested after death? 
# How is this done, is there data on its sens/spec?

# Testing capacity
# ----------------

# Assumptions about the testing capacity available at day d of the simulation 

# For PCR - we will model this (for now, for fitting we'll plug in real data!), as the sum of two sigmoids:
#   - initial stage of PHE ramping up its limited capacity (parameterised by total capacity, inflection day and slope of ramp-up)
#   - second stage of non-PHE labs joining in and ramping up capacity (this hasn't happened yet, but expected soon! same parameterisation)

# For the antigen / antibody tests we define a single sigmoidal capacity curve (starting later than PCR, but with potentially much higher total capacity)
# We further define a ratio between the production of the two, due to them requiring the same capabilities.


def trFunc_testCapacity(
    realTime, # time within simulation (day)
    
    # PCR capacity - initial
    testCapacity_pcr_phe_total = 1e4,
    testCapacity_pcr_phe_inflexday = pd.to_datetime("2020-03-25", format="%Y-%m-%d"),
    testCapacity_pcr_phe_inflexslope = 5.,

    # PCR capacity - increased
    testCapacity_pcr_country_total = 1e5,
    testCapacity_pcr_country_inflexday = pd.to_datetime("2020-04-25", format="%Y-%m-%d"),
    testCapacity_pcr_country_inflexslope = 10,
    
    # Antibody / antigen capacity
    testCapacity_antibody_country_firstday = pd.to_datetime("2020-04-25", format="%Y-%m-%d"),
    
    testCapacity_antibody_country_total = 5e6,
    testCapacity_antibody_country_inflexday = pd.to_datetime("2020-05-20", format="%Y-%m-%d"),
    testCapacity_antibody_country_inflexslope = 20,
    
    testCapacity_antigenratio_country = 0.7,
    
    **kwargs
             
):

    # Returns a dictionary with test names and number available at day "t"
    
    outPCR = (
        #phe phase
        testCapacity_pcr_phe_total * expit((realTime-testCapacity_pcr_phe_inflexday).days/testCapacity_pcr_phe_inflexslope)
        +
        #whole country phase
        testCapacity_pcr_country_total * expit((realTime-testCapacity_pcr_country_inflexday).days/testCapacity_pcr_country_inflexslope)
    )
    
    
    if realTime<testCapacity_antibody_country_firstday:
        outAntiTotal = 0.
    else:
        outAntiTotal = (
            testCapacity_antibody_country_total * expit((realTime-testCapacity_antibody_country_inflexday).days/testCapacity_antibody_country_inflexslope)
        )
    
    return {
        "PCR": outPCR, 
        "Antigen": outAntiTotal*testCapacity_antigenratio_country, 
        "Antibody": outAntiTotal*(1-testCapacity_antigenratio_country)
    }



# Real life data on test capacity and who got tested
# ---------------------------------------------------

df_CHESS = pd.read_csv("/mnt/efs/data/CHESS_Aggregate20200417.csv").drop(0)
df_CHESS.index = pd.to_datetime(df_CHESS["DateOfAdmission"].values,format="%d-%m-%Y")

# Ignore too old and too recent data points
df_CHESS = df_CHESS.sort_index().drop("DateOfAdmission", axis=1).query('20200309 <= index <= '+CONST_DATA_CUTOFF_DATE)

# Get number of tests per age group
df_CHESS_numTests = df_CHESS.loc[:,df_CHESS.columns.str.startswith("AllAdmittedPatientsTestedForCOVID19")]

# Change age groups to reflect our groupings
df_CHESS_numTests_regroup = pd.DataFrame(data = regroup_by_age(
    inp = df_CHESS_numTests.to_numpy().T,
    fromAgeSplits=np.concatenate([np.array([1,5,15,25]),np.arange(45,85+1,10)]),
    toAgeSplits=np.arange(10,80+1,10)
).T)

df_CHESS_numTests_regroup.index = df_CHESS_numTests.index

def inpFunc_testingDataCHESS_PCR(
    realTime,
    realTestData = df_CHESS_numTests_regroup,
    **kwargs
    ):
    
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))
    
    
    return df_CHESS_numTests_regroup.loc[nearest(df_CHESS_numTests_regroup.index, pd.to_datetime(realTime, format="%Y-%m-%d"))]


        
    


# In[16]:


df_CHESS_numTests_regroup


# In[17]:


# Symptom parameters
# ------------------


# Estimating the baseline ILI-symptoms from earlier studies as well as the success rate of COVID-19 tests

# ILI rate estimate from 2018-19 PHE Surveillance of influenza and other respiratory viruses in the UK report: 
# https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/839350/Surveillance_of_influenza_and_other_respiratory_viruses_in_the_UK_2018_to_2019-FINAL.pdf

# TODO - get actual seasonal symptom rate predictions (from 2020 non-SARS respiratory viruses, this data surely exists)
 # (daily rate estimate from Figure 8 of the report)


# Respiratory diagnosis on hospital admissions (not just ILI, all, TODO - get only ILI?)
# NHS Hosp episode statistics 2018-19, page 12 https://files.digital.nhs.uk/F2/E70669/hosp-epis-stat-admi-summ-rep-2018-19-rep.pdf
# In hospital: 1.1 million respiratory episodes out of 17.1 million total episodes

def f_symptoms_nonCOVID(
    realTime, 
    symptomsIliRCGP = 15./100000., # Symptom rate in general non-hospitalised population
    symptomsRespInHospitalFAEs = 1.1/17.1, # Symptom rate in hospitalised population
    
    **kwargs):
    """
    This function defines the non-COVID ILI symptoms rate in the population at a given t time
    """
    
    
    # TODO, add extra data etc as input. For now: 
    return (symptomsIliRCGP, symptomsRespInHospitalFAEs)



# In[18]:


# Distribute tests amongst (a given subset of) symptomatic people
def distTestsSymp(people, testsAvailable, noncovid_sympRatio, symp_HS = range(3,5), alreadyTestedRate = None):
    """
    distribute tests amongst symptomatic people
    people is nAge x nHS-1 x ... (excluding dead)
    """

    # Calculate noncovid, but symptomatic people
    peopleSymp = copy.deepcopy(people)
    peopleSymp[:, :min(symp_HS)] *= noncovid_sympRatio
    peopleSymp[:, max(symp_HS):] *= noncovid_sympRatio

    # Subtract already tested people
    if alreadyTestedRate is not None:
        peopleSymp -= people*alreadyTestedRate 




    # Check if we already tested everyone with a different test
    if np.sum(peopleSymp)<1e-6:  # avoid numerical instabilities
        return (0.,0.)

    testedRatio = min(1., testsAvailable/np.sum(peopleSymp))


    return (
        # test rate
        testedRatio * (peopleSymp/(people+1e-6)), # avoid dividing by zero
        # tests used to achieve this
        testedRatio * np.sum(peopleSymp)
    )


# In[19]:


# Testing policies (how to distribute available tests)
# ----------------------------------------------------

# Estimate at any one time how many people are getting tested (with which tests) from which health states
def policyFunc_testing_symptomaticOnly(
    stateTensor,
    realTime,

    # Test types (names correspoding to testSpecifications)
    testTypes, # = ["PCR", "Antigen", "Antibody"],
    
    # Test Capacity (dict with names above and numbers available on day t)
    testsAvailable, # = trFunc_testCapacity(t)
    
    # OPTIONAL ARGUMENTS (may be different for different policy functions, should come with defaults!)
    antibody_testing_policy = "hospworker_then_random", 
    # This has these values (for now), {"none", "hospworker_then_random", "virus_positive_only", "virus_positive_only_hospworker_first"}
    
    # Baseline symptoms
    f_symptoms_nonCOVID = f_symptoms_nonCOVID,
    
    distributeRemainingToRandom = True,
    return_testsAvailable_remaining = False,
    
    **kwargs
    ):
    """
    Returns a rate distribution of available test types over age, health and isolation states 
    (although age assumed not to matter here)
    """
   
    # Output nAge x nHS x nIso x nTest x len(testTypes) tensor
    out_testRate = np.zeros(stateTensor.shape+(len(testTypes),))
    
    # Testing capacity is testsAvailable
    
    # Get sympom ratio. [0] - general, [1] - hospitalised
    cur_noncovid_sympRatio = f_symptoms_nonCOVID(realTime, **kwargs["f_symptoms_nonCOVID_params"])
    
    
    
    
    
    # PCR testing
    # -----------
    
    # Hospitalised people get priority over PCR tests 
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,2,0], # hospitalised non-positive people, exclude tested and dead people
        testsAvailable = testsAvailable["PCR"],
        noncovid_sympRatio = cur_noncovid_sympRatio[1]
    )
    
    out_testRate[:,:-1,2,0, testTypes.index("PCR")] += testRate
    testsAvailable["PCR"] -= testsUsed
    
    # Prioritise hospital workers next:
    # TODO: check if we should do this? In UK policy there was a 15% max for hospital worker testing until ~2 April...
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,3,0], 
        testsAvailable = testsAvailable["PCR"],
        noncovid_sympRatio= cur_noncovid_sympRatio[0]
    )
    
    out_testRate[:,:-1,3,0, testTypes.index("PCR")] += testRate
    testsAvailable["PCR"] -= testsUsed
    
    # Distribute PCRs left over the other populations
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,:2,0], 
        testsAvailable = testsAvailable["PCR"],
        noncovid_sympRatio= cur_noncovid_sympRatio[0]
    )
    
    out_testRate[:,:-1,:2,0, testTypes.index("PCR")] += testRate
    testsAvailable["PCR"] -= testsUsed
    
    if distributeRemainingToRandom:
        # Distribute PCRs left over the other populations
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,:,0], 
            testsAvailable = testsAvailable["PCR"],
            noncovid_sympRatio= 1.,
            alreadyTestedRate= out_testRate[:,:-1,:,0, testTypes.index("PCR")]
        )

        out_testRate[:,:-1,:,0, testTypes.index("PCR")] += testRate
        testsAvailable["PCR"] -= testsUsed
    
    
    # Antigen testing
    # ---------------
    
    # Hospitalised people get priority over PCR tests 
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,2,0], # hospitalised non-positive people, exclude tested and dead people
        testsAvailable = testsAvailable["Antigen"],
        noncovid_sympRatio= cur_noncovid_sympRatio[1],
        alreadyTestedRate=out_testRate[:,:-1,2, 0, testTypes.index("PCR")]
    )
    
    out_testRate[:,:-1,2,0, testTypes.index("Antigen")] += testRate
    testsAvailable["Antigen"] -= testsUsed
    
    # Prioritise hospital workers next:
    # TODO: check if we should do this? In UK policy there was a 15% max for hospital worker testing until ~2 April...
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,3,0],
        testsAvailable = testsAvailable["Antigen"],
        noncovid_sympRatio= cur_noncovid_sympRatio[0],
        alreadyTestedRate=out_testRate[:,:-1,3, 0, testTypes.index("PCR")]
    )
    
    out_testRate[:,:-1,3,0, testTypes.index("Antigen")] += testRate
    testsAvailable["Antigen"] -= testsUsed
    
    # Distribute Antigen tests left over the other symptomatic people
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,:2,0],
        testsAvailable = testsAvailable["Antigen"],
        noncovid_sympRatio= cur_noncovid_sympRatio[0],
        alreadyTestedRate=out_testRate[:,:-1,:2, 0, testTypes.index("PCR")]
    )
    
    out_testRate[:,:-1,:2,0, testTypes.index("Antigen")] += testRate
    testsAvailable["Antigen"] -= testsUsed
    
    if distributeRemainingToRandom:
        # Distribute antigen tests left over the other non-symptmatic populations
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,:,0], 
            testsAvailable = testsAvailable["Antigen"],
            noncovid_sympRatio= 1.,
            alreadyTestedRate= out_testRate[:,:-1,:,0, :].sum(-1)
        )

        out_testRate[:,:-1,:,0, testTypes.index("Antigen")] += testRate
        testsAvailable["Antigen"] -= testsUsed
    
    
    # Antibody testing
    # ----------------
    
    if antibody_testing_policy == "hospworker_then_random":
    
        # For now: give to hospital workers first, not taking into account previous tests or symptoms
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,3,:2], 
            testsAvailable = testsAvailable["Antibody"],
            noncovid_sympRatio= 1. # basically workers get antibody tested regardless of symptoms
        )

        out_testRate[:,:-1,3,:2, testTypes.index("Antibody")] += testRate
        testsAvailable["Antibody"] -= testsUsed

        # Afterwards let's just distribute randomly in the rest of the population
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,:3,:2], 
            testsAvailable = testsAvailable["Antibody"],
            noncovid_sympRatio= 1. # basically people get antibody tested regardless of symptoms
        )

        out_testRate[:,:-1,:3,:2, testTypes.index("Antibody")] += testRate
        testsAvailable["Antibody"] -= testsUsed
    
    if antibody_testing_policy == "virus_positive_only_hospworker_first":
        
        # For now: give to hospital workers first, not taking into account previous tests or symptoms
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,3,1], 
            testsAvailable = testsAvailable["Antibody"],
            noncovid_sympRatio= 1. # basically workers get antibody tested regardless of symptoms
        )

        out_testRate[:,:-1,3,1, testTypes.index("Antibody")] += testRate
        testsAvailable["Antibody"] -= testsUsed

        # Afterwards let's just distribute randomly in the rest of the population
        # TODO: Maybe prioratise people who tested positive for the virus before???
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,:3,1], 
            testsAvailable = testsAvailable["Antibody"],
            noncovid_sympRatio= 1. # basically people get antibody tested regardless of symptoms
        )

        out_testRate[:,:-1,:3,1, testTypes.index("Antibody")] += testRate
        testsAvailable["Antibody"] -= testsUsed
        
        
    if antibody_testing_policy == "virus_positive_only":
        
        testRate, testsUsed = distTestsSymp(
            people = stateTensor[:,:-1,:,1], 
            testsAvailable = testsAvailable["Antibody"],
            noncovid_sympRatio= 1. # basically people get antibody tested regardless of symptoms
        )

        out_testRate[:,:-1,:,1, testTypes.index("Antibody")] += testRate
        testsAvailable["Antibody"] -= testsUsed
        
    if antibody_testing_policy == "none":
        out_testRate += 0.
        testsAvailable["Antibody"] -= 0.
    

    
    if return_testsAvailable_remaining:
        return out_testRate, testsAvailable
    
    return out_testRate


# In[20]:


# Define reTesting policy(s) (ie give tests to people in non-0 test states!)
def policyFunc_testing_massTesting_with_reTesting(
    stateTensor,
    realTime,
   
    # Test types (names correspoding to testSpecifications)
    testTypes, # = ["PCR", "Antigen", "Antibody"],
    
    # Test Capacity (dict with names above and numbers available on day t)
    testsAvailable, # = trFunc_testCapacity(t)
    
    # OPTIONAL ARGUMENTS (may be different for different policy functions, should come with defaults!)
    
    basic_policyFunc = policyFunc_testing_symptomaticOnly,
    # This basic policy will: 
    # - do PCRs on symptomatic hospitalised people
    # - do PCRs on symptomatic hospital staff
    # - do PCRs on symptomatic non-hospitalised people
    # If PCRs run out at any stage, we use antigen tests with same priorisation
    
    # Afterwards given fractions of remaining antigen tests are distributed amongst people given these ratios and their earlier testing status:
    #retesting_antigen_viruspos_ratio = 0.1, # find virus false positives
    # UPDATE <- retesting viruspos is same ratio is normal testing, as long as they're not in quarantine already!
    retesting_antigen_immunepos_ratio = 0.05, # find immunity false positives
    # The rest of antigen tests are given out randomly
    
    # Antibody tests are used primarily on people who tested positive for the virus 
    #  (set in basic_policyFunc!, use "virus_positive_only_hospworker_first"!)
    # Afterwards we can use the remaining on either random people (dangerous with many false positives!) 
    # or for retesting people with already positive immune tests to make sure they're still immune,
    # controlled by this ratio:
    retesting_antibody_immunepos_ratio = 1.,
    
    #distributeRemainingToRandom = True, # TODO - otherwise stockpile for future, how?
    return_testsAvailable_remaining = False,   
    
    **kwargs
    ):
    
    # Output nAge x nHS x nIso x nTest x len(testTypes) tensor
    out_testRate = np.zeros(stateTensor.shape+(len(testTypes),))
    
    
    # First distribute tests to symptomatic people as usual:

    # inpArgs change to not distributing tests randomly:
    basic_policyFunc_params_modified = copy.deepcopy(kwargs["basic_policyFunc_params"])
    basic_policyFunc_params_modified["distributeRemainingToRandom"] = False
    basic_policyFunc_params_modified["return_testsAvailable_remaining"] = True
    
    
    # Run the basic policy function with these modified parameters
    out_testRate, testsAvailable = basic_policyFunc(
        stateTensor,
        realTime = realTime,
        testTypes = testTypes,
        testsAvailable = testsAvailable,
        **basic_policyFunc_params_modified
    )
    
    
    # We assume PCRs tend to run out done on symptomatic people in 0 Test state, so no retesting via PCR.
    
    
    # Antigen testing
    # ---------------
       
    # Retesting immune positive people
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,:,2:], # immune positive people
        testsAvailable = testsAvailable["Antigen"] * retesting_antigen_immunepos_ratio,
        noncovid_sympRatio= 1., # set to 1. for ignoring symptom vs non-symptom
    )
    
    out_testRate[:,:-1,:,2:, testTypes.index("Antigen")] += testRate
    testsAvailable["Antigen"] -= testsUsed
    

    # Distribute antigen tests left over the other non-symptmatic populations
    # UPDATE <- here we use tests equally distributed among people with negative or positive previous virus tests, 
    # as long as they are in non-quarantined state (isoState 0) # TODO - hospital worker testing???
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,0,:2], # non-quarantined virus positive people
        testsAvailable = testsAvailable["Antigen"],
        noncovid_sympRatio= 1.,
        alreadyTestedRate= out_testRate[:,:-1,0,:2, testTypes.index("Antigen")] + out_testRate[:,:-1,0,:2, testTypes.index("PCR")]
    )

    out_testRate[:,:-1,0,:2, testTypes.index("Antigen")] += testRate
    testsAvailable["Antigen"] -= testsUsed
    
    
    # Antibody testing
    # -----------------
    # Retesting antibody positive people
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,:,2:], # virus positive people
        testsAvailable = testsAvailable["Antibody"] * retesting_antibody_immunepos_ratio,
        noncovid_sympRatio= 1., # set to 1. for ignoring symptom vs non-symptom
    )
    
    
    # Afterwards let's just distribute randomly in the rest of the population
    testRate, testsUsed = distTestsSymp(
        people = stateTensor[:,:-1,:,:2], 
        testsAvailable = testsAvailable["Antibody"],
        noncovid_sympRatio= 1., # basically people get antibody tested regardless of symptoms
        alreadyTestedRate= out_testRate[:,:-1,:,:2, testTypes.index("Antibody")]
    )

    out_testRate[:,:-1,:,:2, testTypes.index("Antibody")] += testRate
    testsAvailable["Antibody"] -= testsUsed
    
    
    if return_testsAvailable_remaining:
        return out_testRate, testsAvailable
    
    return out_testRate


# In[21]:


def trFunc_testing(
    stateTensor,
    t,
    realStartDate,
    #policyFunc = policyFunc_testing_symptomaticOnly,
    policyFunc = policyFunc_testing_massTesting_with_reTesting,
    inpFunc_testSpecifications = inpFunc_testSpecifications, 
    trFunc_testCapacity = trFunc_testCapacity,
    inpFunc_realData_testCapacity = inpFunc_testingDataCHESS_PCR,
    **kwargs
    ):
    """
    Returns a tensor of rates transitioning to tested states
    """
    trTensor_testTransitions = np.zeros((nAge, nHS, nIso, nTest, nTest))
    
    
    testSpecifications = inpFunc_testSpecifications(**kwargs["inpFunc_testSpecifications_params"])
    
    testTypes = list(set(testSpecifications["Name"]))
    
    
    # Check if we have real data on the administered tests
    
    # Add the current data on within-hospital PCRs carried out already
    curDate = pd.to_datetime(realStartDate, format="%Y-%m-%d") + pd.to_timedelta(int(t), unit="D")
    realData_closest = inpFunc_realData_testCapacity(realTime = curDate, **kwargs["inpFunc_realData_testCapacity_params"])
            
    if realData_closest.name == curDate: # We do have data, just fill it in
        testsAdministeredRate = np.zeros(stateTensor.shape+(len(testTypes),))
        
        
        
        # TODO - fix this very hacky solution accessing symptomatic ratio as a subfunc of the policy func
        noncovid_sympRatio = kwargs["policyFunc_params"]["basic_policyFunc_params"]["f_symptoms_nonCOVID"](curDate, **kwargs["policyFunc_params"]["basic_policyFunc_params"]["f_symptoms_nonCOVID_params"])
        noncovid_sympRatio = noncovid_sympRatio[1] # Use hospitalised patient symptom ratio
        symptomaticRatePerDiseaseState = np.array([noncovid_sympRatio]*stateTensor.shape[1])
        symptomaticRatePerDiseaseState[3:-(nR+1)] = 1. # set the symptomatic ratio of symptomatic states to 1
        symptomaticPeoplePerDiseaseStateInHospital = stateTensor[:,:-1, 2, 0] * np.expand_dims(symptomaticRatePerDiseaseState[:-1], axis=0)
        
        testsAdministeredRate[:,:-1, 2, 0, testTypes.index("PCR")] += (
            np.expand_dims(realData_closest.to_numpy(),1) # true number of tests on given day per age group
            * 
            (symptomaticPeoplePerDiseaseStateInHospital / np.sum(symptomaticPeoplePerDiseaseStateInHospital, axis=-1, keepdims=True)) 
            # Calculate in what ratio we distribute the tests to people along disease states based on symptomatic (age is given in data!)
        )/(stateTensor[:,:-1, 2, 0]+1e-10) # Divide by total people in each state to get testing rate

        
    
    else: # we don't have data, follow our assumed availability and policy curves
        
        # policyFunc returns stateTensor x testTypes tensor of test administration rates 
        testsAdministeredRate = policyFunc(
            stateTensor,
            realTime = curDate,
            testTypes = testTypes,
            testsAvailable = trFunc_testCapacity(realTime = curDate, **kwargs["trFunc_testCapacity_params"]),
            **kwargs["policyFunc_params"]
        )
       
    
    
    # Compute the transition ratio to tested states, given the administered tests
        
    for testType in testTypes:
        # Get the appropriate slices from testsAdmin. and testSpecs
        curTestSpecs = testSpecifications[testSpecifications["Name"]==testType]

        for curTS in range(nTest): 
            # Set output positive test state based on current test state
            if curTS == int(curTestSpecs["OutputTestState"].values[0]):
                # already positive for the given test
                outTS_pos = curTS
            elif curTS == 3:
                # If already positive for both, stay positive
                outTS_pos = 3
            else:
                # Transition 0->1, 0->2, 1->2, 1->3 or 2->3
                outTS_pos = curTS + int(curTestSpecs["OutputTestState"].values[0])
                    
            
            # Where do we go after negative test based on where we are now?
            if curTS == 0:
                # Negatives stay negatives
                outTS_neg = 0
            elif curTS == 3: 
                # go to only virus or antibody positive from both positive
                outTS_neg = 3-int(curTestSpecs["OutputTestState"].values[0])
            elif curTS == int(curTestSpecs["OutputTestState"].values[0]):
                # go to 0 if tested for the one you're positive for
                outTS_neg = 0
            else:
                # stay where you are if you test negative for the one you didnt have anyway
                outTS_neg = curTS
                
            
            
            # Get the transition rates based on current health states
            for curHS in range(nHS):
                # Add the true positives * (1-FNR)
                if curHS in curTestSpecs["TruePosHealthState"].values[0]:
                    trTensor_testTransitions[:,curHS,:, curTS, outTS_pos] += (                        
                        testsAdministeredRate[:,curHS,:,curTS,testTypes.index(testType)] *
                        (1-curTestSpecs[curTestSpecs["InputHealthState"] == curHS]["FalseNegativeRate"].values[0])
                    )

                else:
                # Add the false positives * FPR
                    trTensor_testTransitions[:,curHS,:,curTS, outTS_pos] += (                        
                        testsAdministeredRate[:,curHS,:,curTS,testTypes.index(testType)] *
                        curTestSpecs[curTestSpecs["InputHealthState"] == curHS]["FalsePositiveRate"].values[0]
                    )


                # Add the false negatives (FNR)
                if curHS in curTestSpecs["TruePosHealthState"].values[0]:
                    trTensor_testTransitions[:,curHS,:,curTS,outTS_neg] += (                        
                        testsAdministeredRate[:,curHS,:,curTS,testTypes.index(testType)] *
                        curTestSpecs[curTestSpecs["InputHealthState"] == curHS]["FalseNegativeRate"].values[0]
                    )

                else:
                # Add the true negatives * (1-FNR)
                    trTensor_testTransitions[:,curHS,:,curTS,outTS_neg] += (                        
                        testsAdministeredRate[:,curHS,:,curTS,testTypes.index(testType)] *
                        curTestSpecs[curTestSpecs["InputHealthState"] == curHS]["FalsePositiveRate"].values[0]
                    )
                
    

    return trTensor_testTransitions#, testsAdministeredRate


# ## Quarantine policies
# 
# This section describes alternatives to the social distancing by full lockdown (that is implemented as a change in the socialMixing matrices).
# 
# One alternative is case isolation, either by hospitalisation or by home isolation. We will assume that all non-symptomatic people who test positive are home isolated along with families for nDaysInIsolation days. Symptomatic people have a chance of being immediately hospitalised instead of sent into home isolation

# In[22]:


def trFunc_quarantine_caseIsolation(
    trTensor_complete,
    t,
    trTensor_testing, # This is used to establish who gets tests and how many of those end up positive.
    nDaysInHomeIsolation = 14.,
    timeToIsolation = 0.5, # (days) time from testing positive to actually getting isolated
    # On average this many people get hospitalised (compared to home isolation), but modulated by age (TODO: values > 1? clip for now..)
    symptomHospitalisedRate_ageAdjusted = np.clip(
        adjustRatesByAge_KeepAverageRate(0.3, ageRelativeAdjustment=relativeAdmissionRisk_given_COVID_by_age),
        0.,1.),
    symptomaticHealthStates = [3,4], # TODO - define this in global variable and just pass here!
    **kwargs
    ):
    """
    This function redistributes testing rates, so they dont only create a testing state update, but also an isolation state update
    """

    trTensor_quarantineRate = np.zeros(stateTensor.shape+(nIso,))
    
    trTensor_freshlyVirusPositiveRate_inIso0 = copy.deepcopy(trTensor_testing[:,:,0,:2,1])
    trTensor_freshlyBothPositiveRate_inIso0 = copy.deepcopy(trTensor_testing[:,:,0,2:,3])
    
    
    for curHS in range(stateTensor.shape[1]-1): # ignore dead
        if curHS in symptomaticHealthStates:
            # Send a fraction of people (normal) who are symptomatic and tested positive to hospital, based on their age
            trTensor_quarantineRate[:,curHS,0,:2,2] += (
                (1./timeToIsolation)*symptomHospitalisedRate_ageAdjusted[:,np.newaxis]
                *
                trTensor_freshlyVirusPositiveRate_inIso0[:,curHS]
            )
            trTensor_quarantineRate[:,curHS,0,2:,2] += (
                (1./timeToIsolation)*symptomHospitalisedRate_ageAdjusted[:,np.newaxis]
                *
                trTensor_freshlyBothPositiveRate_inIso0[:,curHS]
            )
            # The rest to home isolation
            trTensor_quarantineRate[:,curHS,0,:2,1] += (
                (1./timeToIsolation)*(1.-symptomHospitalisedRate_ageAdjusted[:,np.newaxis])
                *
                trTensor_freshlyVirusPositiveRate_inIso0[:,curHS]
            )
            trTensor_quarantineRate[:,curHS,0,2:,1] += (
                (1./timeToIsolation)*(1.-symptomHospitalisedRate_ageAdjusted[:,np.newaxis])
                *
                trTensor_freshlyBothPositiveRate_inIso0[:,curHS]
            )

        else:
            # Send all non-symptomatic (normal) who tested freshly positive to home isolation
            trTensor_quarantineRate[:,curHS,0,:2,1] += (
                1./timeToIsolation
                *
                trTensor_freshlyVirusPositiveRate_inIso0[:,curHS]
            )
            trTensor_quarantineRate[:,curHS,0,2:,1] += (
                1./timeToIsolation
                *
                trTensor_freshlyBothPositiveRate_inIso0[:,curHS]
            )
    
    # Release people from home isolation after isolation period
    trTensor_quarantineRate[:,:,1,:,0] = 1./nDaysInHomeIsolation
    
    # Hospitalised people are assumed to be released after recovery, with normal rates (TODO: think if this is correct)
    
    # TODO!!! - importantly, hospital workers are not being home isolated / hospitalised under this policy. 
    # How to keep track of hospital workers who get hospitalised or home isolated themselves, 
    # such that they get back to being hospital workers afterwards? 
    # A simple (slightly incorrect) solution would be to just implement a non-specific "pull" from isoState=0 people to hospital workers to fill up the missing people?
    # But the rate of this pull would be impossible to compute and would still be incorrect. Gotta think more on this.
    
    
    # Update the whole tensor accordingly
    # Make a copy for safety:
    out_trTensor_complete = copy.deepcopy(trTensor_complete)
    
    # First remove all the iso 0->0, test 0,1->1, 2,3->3 transitions (as they're all either hospitalised or sent to home isolation)
    out_trTensor_complete[:,:,0,:2,:,0,1] = 0.
    out_trTensor_complete[:,:,0,2:,:,0,3] = 0.
    
    
    
    # Newly virus positive, newly home-isolated, diagonal in disease state transition
    np.einsum('ijkj->ijk',
        out_trTensor_complete[:,:,0,:2,:,1,1])[:] = trTensor_quarantineRate[:,:,0,:2,1]
    np.einsum('ijkj->ijk',
        out_trTensor_complete[:,:,0,2:,:,1,3])[:] = trTensor_quarantineRate[:,:,0,2:,1]
    
    # Newly virus positive, newly hospitalised, diagonal in disease state transition
    np.einsum('ijkj->ijk',
        out_trTensor_complete[:,:,0,:2,:,2,1])[:] = trTensor_quarantineRate[:,:,0,:2,2]
    np.einsum('ijkj->ijk',
        out_trTensor_complete[:,:,0,2:,:,2,3])[:] = trTensor_quarantineRate[:,:,0,2:,2]
    
    # Home isolated people are "let go" after nDaysInHomeIsolation, without changing disease or testing state
    # (TODO: represent multiple testing / needing negative tests to let go, etc - hard problem!)
    # (UPDATE: multiple testing have now been represented, but for now we'll still let go people based on fixed time rather than negative test, to save tests!)
    np.einsum('ijkjk->ijk',
        out_trTensor_complete[:,:,1,:,:,0,:])[:] = trTensor_quarantineRate[:,:,1,:,0]
    
    
    # Return the full updated tensor (so NOT += outside, but actually =)
    return out_trTensor_complete
    


# ## Full simulation function

# In[23]:


# Function that computes the right side of the non-lin model ODE
def dydt_Complete(t, 
                  stateTensor_flattened, # Might be double the normal size (as first dimension) _withNewOnlyCopy, if debugReturnNewPerDay
                  
                  realStartDate = pd.to_datetime("2020-02-20", format="%Y-%m-%d"),
                  
                  # debug
                  debugTransition = False,
                  debugTimestep = False,
                  debugReturnNewPerDay = True, # Now implemented by default into state iteration
                  
                  # Dimensions
                  nAge=nAge, nHS=nHS, nI=nI, nR=nR, nIso=nIso, nTest=nTest,
                  
                  # Input functions and tensors
                  # ----------------------------
                  
                  # Health state updates
                  trFunc_diseaseProgression = trFunc_diseaseProgression,
                  trFunc_newInfections = trFunc_newInfections_Complete,
                  
                  # Initial incoming travel-based infections (before restrictions)
                  trFunc_travelInfectionRate_ageAdjusted = trFunc_travelInfectionRate_ageAdjusted,
                  
                  # Hospitalisation and recovery
                  trFunc_HospitalAdmission = trFunc_HospitalAdmission,
                  trFunc_HospitalDischarge = trFunc_HospitalDischarge,                  
                  
                  # Policy changes (on social distancing for now) (TODO - possibly make more changes)
                  tStartSocialDistancing = pd.to_datetime("2020-03-23", format="%Y-%m-%d"),
                  tStopSocialDistancing = pd.to_datetime("2025-03-23", format="%Y-%m-%d"),
                  
                  tStartImmunityPassports = pd.to_datetime("2025-03-23", format="%Y-%m-%d"),
                  tStopImmunityPassports = pd.to_datetime("2025-03-23", format="%Y-%m-%d"),
                  
                  tStartQuarantineCaseIsolation = pd.to_datetime("2025-03-23", format="%Y-%m-%d"),
                  tStopQuarantineCaseIsolation = pd.to_datetime("2025-03-23", format="%Y-%m-%d"),
                  trFunc_quarantine = trFunc_quarantine_caseIsolation,
                  
                  # Testing
                  trFunc_testing = trFunc_testing,
                  #policyFunc_testing = policyFunc_testing_symptomaticOnly,
                  #testSpecifications = testSpecifications, 
                  #trFunc_testCapacity = trFunc_testCapacity,
                  #trFunc_testCapacity_param_testCapacity_antigenratio_country = 0.3
                  
                  **kwargs
                  
):
    
    if debugTimestep:
        print(t)
    
    # Initialise return
    if debugReturnNewPerDay: # the input has 2 copies of the state tensor, second copy being the cumulative incomings
        stateTensor = np.reshape(stateTensor_flattened, [2, nAge, nHS, nIso, nTest])[0]
    else:
        stateTensor = np.reshape(stateTensor_flattened, [nAge, nHS, nIso, nTest])
    
    dydt = np.zeros_like(stateTensor)
    
    # Initialise the full transition tensor
    trTensor_complete = np.zeros((nAge, nHS, nIso, nTest, nHS, nIso, nTest))
    

    # Disease condition updates
    # ---------------------------
    trTensor_diseaseProgression = trFunc_diseaseProgression(**kwargs["trFunc_diseaseProgression_params"])
    
    # Get disease condition updates with no isolation or test transition ("diagonal along those")
    for k1 in [0,1,2,3]:
        np.einsum('ijlml->ijlm',
            trTensor_complete[:,:,k1,:,:,k1,:])[:] += np.expand_dims(
                trTensor_diseaseProgression[:,:,k1,:]
                ,[2]) # all non-hospitalised disease progression is same

  
    # Compute new infections (0->1 in HS) with no isolation or test transition ("diagonal along those")
    cur_policySocialDistancing = (
                    t >= (tStartSocialDistancing - realStartDate).days
                )*(
                    t <   (tStopSocialDistancing - realStartDate).days
                )
    cur_policyImmunityPassports = (
                    t >= (tStartImmunityPassports - realStartDate).days
                )*(
                    t <   (tStopImmunityPassports - realStartDate).days
                )
    np.einsum('iklkl->ikl',
        trTensor_complete[:,0,:,:,1,:,:])[:] += (
            trFunc_newInfections(
                stateTensor, 
                policySocialDistancing = cur_policySocialDistancing,
                policyImmunityPassports = cur_policyImmunityPassports,
                **kwargs["trFunc_newInfections_params"]
            ))
    
    # Also add new infected from travelling of healthy people, based on time-within-simulation (this is correct with all (0,0) states, as tested or isolated people dont travel)
    trTensor_complete[:,0,0,0,1,0,0] += trFunc_travelInfectionRate_ageAdjusted(t, **kwargs["trFunc_travelInfectionRate_ageAdjusted_params"])
    
    
    # Hospitalisation state updates
    # -----------------------
    
    # Hospitalisation and recovery rates
    # We assume for now that these only depend on age and disease progression, not on testing state 
    # (TODO - update this given new policies)
    
    # The disease and testing states don't change due to hospitalisation. 
    # Hospital staff is treated as already hospitalised from all aspects expect social mixing, should suffice for now
    # TODO - Could try to devise a scheme in which hospital staff gets hospitalised and some recoveries from hospitalised state go back to hospital staff.
    # TODO - same issue with hospital staff home isolating; that's probably more important question!
    for k1 in [0,1]:
         np.einsum('ijljl->ijl',
            trTensor_complete[:,:,k1,:,:,2,:])[:] += np.expand_dims(
             trFunc_HospitalAdmission(**kwargs["trFunc_HospitalAdmission_params"]),[2])   

    # Add recovery from hospital rates
    # TODO - again here (for now) we assume all discharged people go back to "normal state" instead of home isolation, have to think more on this
    np.einsum('ijljl->ijl',
            trTensor_complete[:,:,2,:,:,0,:])[:] += np.expand_dims(
                 trFunc_HospitalDischarge(**kwargs["trFunc_HospitalDischarge_params"]),[2])     

    
    
    
    
    # Testing state updates
    # ---------------------
    
    # trFunc_testing returns a stateTensor x testStates output 
    #      after the policyFunc assigns tests that are evaluated according to testSpecifications
    
    # Diagonal (no transitions) in age, health state and isolation state 
    # (for now, probably TODO: testing positive correlates with new hospitalisation!)
    trTensor_testing = trFunc_testing(
                                            stateTensor,
                                            t,
                                            realStartDate,
                                            **kwargs["trFunc_testing_params"]
                                        )  
    
    np.einsum('ijkljkm->ijklm',
            trTensor_complete)[:] += trTensor_testing

    
    # Quarantine policy
    # ------------------
    
    # Check if policy is "on"
    if (
            t >= (tStartQuarantineCaseIsolation - realStartDate).days
        )*(
            t <   (tStopQuarantineCaseIsolation - realStartDate).days
        ):
        # New quarantining only happens to people who are transitioning already from untested to virus positive state
        # Therefore here we DO use non-diagonal transitions, and we 
        #     redistribute the transtion rates given the testing (which was previously assumed not to create transition in isolation state)
        trTensor_complete = trFunc_quarantine(
                                                trTensor_complete,
                                                t,
                                                trTensor_testing, 
                                                **kwargs["trFunc_quarantine_params"]
                                            )
        
        

    
    # Final corrections
    # -----------------
    
    
    
    # TODO: simulate aging and normal birth / death (not terribly important on these time scales, but should be quite simple)
    
    
    # Ensure that every "row" sums to 0 by adding to the diagonal (doesn't create new people out of nowhere)
    # Extract (writable) diagonal array and subtract the "row"-sums for each initial state
    np.einsum('ijkljkl->ijkl', trTensor_complete)[:] -= np.einsum('...jkl->...', trTensor_complete) 
    
    
    # Compute the actual derivatives
    dydt = np.einsum('ijkl,ijklmnp->imnp', stateTensor, trTensor_complete) # contract the HS axis, keep age
    
    
    if debugReturnNewPerDay:
        """
            If this is true, instead of returning the real dydt, 
            return only the positive "incoming" number of people to each state, so we can track "new cases"
            This needs some approximations, as follows:
                1. Take the normal transition tensor (with rates potentially > 0)
                2. From all states re-normalise the outgoing rates to sum at most to 1 
                    (if they were less, keep it, if larger, then this represents 
                    “in this day, all people will leave this state, in these ratios to these states”)
                3. Multiply only these outgoing rates with the current state 
                    (so the result wont keep the same number of people as normal, 
                    but only represent the “new incomings” for each state)
        """
        
        trTensor_complete_newOnly = copy.deepcopy(trTensor_complete)
        
        # TODO - Think - this is probably unnecessary actually, artifically reduces "new" rates?
#         # Devide each row by the absolute diagonal rate (that is the sum of the row), but only if its larger than 1
#         trTensor_complete_newOnly /= (
#             np.expand_dims(
#                 np.clip(np.abs(np.einsum('ijkljkl->ijkl', trTensor_complete_newOnly)), a_min=1., a_max=np.inf),
#                 axis=[4,5,6]
#             )
#         )
        
        # Set the diagonals to zero (no preservation, no outgoing, will end up being the incoming only)
        np.einsum('ijkljkl->ijkl', trTensor_complete_newOnly)[:] = 0.
        
        dydt_newOnly = np.einsum('ijkl,ijklmnp->imnp', stateTensor, trTensor_complete_newOnly)
        
        dydt = np.stack([dydt, dydt_newOnly], axis=0)
    
    
    if debugTransition:
        return np.reshape(dydt, -1), trTensor_complete
    
    return np.reshape(dydt, -1)
    


# ## Initialise and run the model

# In[24]:


# Initialise state
stateTensor_init = copy.deepcopy(stateTensor)

# Populate 
stateTensor_init[:,0,0,0] = agePopulationTotal

# Move hospital staff to working in hospital
stateTensor_init[:,0,0,0] -= ageNhsClinicalStaffPopulationRatio * agePopulationTotal
stateTensor_init[:,0,3,0] += ageNhsClinicalStaffPopulationRatio * agePopulationTotal

# Move people to hospital according to baseline occupation (move only from normal people, not hospital staff!)
stateTensor_init[:,0,2,0] += initBaselineHospitalOccupancyEquilibriumAgeRatio * stateTensor_init[:,0,0,0]
stateTensor_init[:,0,0,0] -= initBaselineHospitalOccupancyEquilibriumAgeRatio * stateTensor_init[:,0,0,0]




# Infect some young adults/middle-aged people
# stateTensor_init[2:4,0,0,0] -= 1000.#/np.sum(agePopulationTotal)
# stateTensor_init[2:4,1,0,0] += 1000.#/np.sum(agePopulationTotal)

# BETTER! - People get infected by travel in early stages!


# In[25]:


def solveSystem(stateTensor_init, total_days = 200, samplesPerDay=np.inf, **kwargs):
    # Run the simulation
    
    if kwargs["debugReturnNewPerDay"]: # Keep the second copy as well
        cur_stateTensor = np.reshape(
            np.stack([copy.deepcopy(stateTensor_init), copy.deepcopy(stateTensor_init)], axis=0),-1)
    else:
        cur_stateTensor = np.reshape(copy.deepcopy(stateTensor_init),-1)
    
    if np.isinf(samplesPerDay):
        # Run precise integrator - used for all simulations
        out = integrate.solve_ivp(
            fun = lambda t,y: dydt_Complete(t,y, **kwargs),
            t_span=(0.,total_days),
            y0 = cur_stateTensor,
            method='RK23',
            t_eval=range(total_days),
            rtol = 1e-3, #default 1e-3
            atol = 1e-3, # default 1e-6
        )
        
        out = out.y
        
    else:
        # Run simple Euler method with given step size (1/samplesPerDay) for quickly investigating code behavior
        deltaT = 1./samplesPerDay
        out = np.zeros((np.prod(stateTensor_init.shape),total_days))
                       
        for tt in range(total_days*samplesPerDay):
            if tt % samplesPerDay==0:
                out[:, int(tt/samplesPerDay)] = cur_stateTensor
                       
            cur_stateTensor += deltaT * dydt_Complete((tt*1.)/(1.*samplesPerDay),cur_stateTensor, **kwargs)
            
    
    # Reshape to reasonable format
    if kwargs["debugReturnNewPerDay"]:
        out = np.reshape(out, (2,) + stateTensor_init.shape+(-1,))
    else:
        out = np.reshape(out, stateTensor_init.shape+(-1,))
    
    
    return out


# In[26]:


# # Uncomment below for an example short run of the full model with base parameters and quarantining policy turned on.
# # Takes ~2 mins on single CPU core.

# # Build a dictionary out of arguments with defaults
# paramDict_default = build_paramDict(dydt_Complete)
# paramDict_default["dydt_Complete"] = dydt_Complete
# paramDict_default["INIT_stateTensor_init"] = stateTensor_init

# # Example way to set parameters conveniently, here we start quarantining early based on test results
# paramDict_current = copy.deepcopy(paramDict_default)
# paramDict_current["tStartQuarantineCaseIsolation"] = pd.to_datetime("2020-03-23", format="%Y-%m-%d")


# out1 = solveSystem(
#     stateTensor_init,
#     total_days = 80,
#     **paramDict_current
# )


# # Building a flat model function to use with outside callers
# 
# Having the model defined flexible above, here we make it convenient to use to access any parameters, and to use with arbitrary outside optimisers / distributed workflows.

# In[29]:


# Build a dictionary out of arguments with defaults
paramDict_default = build_paramDict(dydt_Complete)
paramDict_default["dydt_Complete"] = dydt_Complete
paramDict_default["INIT_stateTensor_init"] = stateTensor_init


# In[30]:


paramTable_default = paramDict_toTable(paramDict_default)


# In[31]:


# Define sets of params to differentiate between
paramTypes = OrderedDict()

paramTypes["basic"] = [
    # Definitions
    'debugTransition', 'debugTimestep', 'debugReturnNewPerDay', 
    'nAge', 'nHS', 'nI', 'nR', 'nIso', 'nTest',
    # Real data input
    'trFunc_testing_params_inpFunc_realData_testCapacity',
    'trFunc_testing_params_inpFunc_realData_testCapacity_params_realTestData',
    'tStartSocialDistancing',
    # State initialisation (no assumptions about COVID!)
    'INIT_stateTensor_init'
]

paramTypes["functions"] = [
    'trFunc_diseaseProgression', 
    'trFunc_newInfections',
    'trFunc_travelInfectionRate_ageAdjusted',
    'trFunc_HospitalAdmission',
    'trFunc_HospitalDischarge',
    'trFunc_testing',
    'trFunc_testing_params_policyFunc',
    'trFunc_testing_params_policyFunc_params_basic_policyFunc',
    'trFunc_testing_params_policyFunc_params_basic_policyFunc_params_f_symptoms_nonCOVID',
    'trFunc_testing_params_inpFunc_testSpecifications',
    'trFunc_testing_params_trFunc_testCapacity',
    'trFunc_quarantine',
    'dydt_Complete'
]


paramTypes["ensemble"] = [
    'realStartDate',
    
    # Travel infections
    'trFunc_travelInfectionRate_ageAdjusted_params_travelMaxTime',
       'trFunc_travelInfectionRate_ageAdjusted_params_travelBaseRate',
       'trFunc_travelInfectionRate_ageAdjusted_params_travelDecline_mean',
       'trFunc_travelInfectionRate_ageAdjusted_params_travelDecline_slope',
       'trFunc_travelInfectionRate_ageAdjusted_params_travelInfection_peak',
       'trFunc_travelInfectionRate_ageAdjusted_params_travelInfection_maxloc',
       'trFunc_travelInfectionRate_ageAdjusted_params_travelInfection_shape',
    
    # new infections
    'trFunc_newInfections_params_ageSocialMixingBaseline',
       'trFunc_newInfections_params_ageSocialMixingDistancing',
       'trFunc_newInfections_params_withinHospitalSocialMixing',
       'trFunc_newInfections_params_transmissionInfectionStage',
    
    # disease progression
    'trFunc_diseaseProgression_params_nonsymptomatic_ratio',
    'trFunc_diseaseProgression_params_infect_to_symptoms',
    'trFunc_diseaseProgression_params_symptom_to_recovery',
    'trFunc_diseaseProgression_params_symptom_to_hospitalisation',
    'trFunc_diseaseProgression_params_hospitalisation_to_recovery',
    'trFunc_diseaseProgression_params_IgG_formation',
    'trFunc_diseaseProgression_params_caseFatalityRatioHospital_given_COVID_by_age',
    'trFunc_diseaseProgression_params_ageRelativeRecoverySpeed',
    'trFunc_diseaseProgression_params_nonsymp_to_recovery',
    'trFunc_diseaseProgression_params_inverse_IS1_IS2',
    
    # Hospitalisation
    'trFunc_HospitalAdmission_params_ageHospitalisationRateBaseline',
   'trFunc_HospitalAdmission_params_infToHospitalExtra',
   'trFunc_HospitalAdmission_params_ageRelativeExtraAdmissionRiskToCovid',
   'trFunc_HospitalDischarge_params_ageHospitalisationRecoveryRateBaseline',
   'trFunc_HospitalDischarge_params_dischargeDueToCovidRateMultiplier',
    
    # PCR testing? 
    'trFunc_testing_params_inpFunc_testSpecifications_params_PCR_FNR_I1_to_R2',
    'trFunc_testing_params_inpFunc_testSpecifications_params_PCR_FPR',
    
    # Symptoms
    'trFunc_testing_params_policyFunc_params_basic_policyFunc_params_f_symptoms_nonCOVID_params_symptomsIliRCGP',
    'trFunc_testing_params_policyFunc_params_basic_policyFunc_params_f_symptoms_nonCOVID_params_symptomsRespInHospitalFAEs'
]


paramTypes["policy"] = [    
    # Timings
    'tStopSocialDistancing',
    'tStartImmunityPassports', 'tStopImmunityPassports',
    'tStartQuarantineCaseIsolation', 'tStopQuarantineCaseIsolation',
    
    # Quarantine
    'trFunc_quarantine_params_nDaysInHomeIsolation',
    'trFunc_newInfections_params_ageSocialMixingIsolation',
    'trFunc_quarantine_params_timeToIsolation',
    'trFunc_quarantine_params_symptomHospitalisedRate_ageAdjusted',
    'trFunc_quarantine_params_symptomaticHealthStates',
    
    # Testing
    'trFunc_testing_params_trFunc_testCapacity_params_testCapacity_pcr_phe_total',
    'trFunc_testing_params_trFunc_testCapacity_params_testCapacity_pcr_phe_inflexday',
    'trFunc_testing_params_trFunc_testCapacity_params_testCapacity_pcr_phe_inflexslope',
    'trFunc_testing_params_trFunc_testCapacity_params_testCapacity_pcr_country_total',
    'trFunc_testing_params_trFunc_testCapacity_params_testCapacity_pcr_country_inflexday',
    'trFunc_testing_params_trFunc_testCapacity_params_testCapacity_pcr_country_inflexslope',
    'trFunc_testing_params_trFunc_testCapacity_params_testCapacity_antibody_country_firstday',
    'trFunc_testing_params_trFunc_testCapacity_params_testCapacity_antibody_country_total',
    'trFunc_testing_params_trFunc_testCapacity_params_testCapacity_antibody_country_inflexday',
    'trFunc_testing_params_trFunc_testCapacity_params_testCapacity_antibody_country_inflexslope',
    'trFunc_testing_params_trFunc_testCapacity_params_testCapacity_antigenratio_country',
    
    'trFunc_testing_params_policyFunc_params_retesting_antigen_immunepos_ratio',
    'trFunc_testing_params_policyFunc_params_retesting_antibody_immunepos_ratio',
    
    'trFunc_testing_params_policyFunc_params_return_testsAvailable_remaining',
    
    'trFunc_testing_params_policyFunc_params_basic_policyFunc_params_antibody_testing_policy',
    'trFunc_testing_params_policyFunc_params_basic_policyFunc_params_distributeRemainingToRandom',
    'trFunc_testing_params_policyFunc_params_basic_policyFunc_params_distributeRemainingToRandom',
    'trFunc_testing_params_policyFunc_params_basic_policyFunc_params_return_testsAvailable_remaining',
    
    
    
    
    
    
    
    # Test specs
    'trFunc_testing_params_inpFunc_testSpecifications_params_antigen_FNR_I1_to_R2',
    'trFunc_testing_params_inpFunc_testSpecifications_params_antigen_FPR',
    'trFunc_testing_params_inpFunc_testSpecifications_params_antibody_FNR_I1_to_R2',
    'trFunc_testing_params_inpFunc_testSpecifications_params_antibody_FPR_S_to_I4'
]

# Check if we defined all params and nothing extra
print(set(paramTable_default.columns) - set([b for a in paramTypes.values() for b in a]))
print(set([b for a in paramTypes.values() for b in a]) - set(paramTable_default.columns))


# In[32]:


# # Save paramTypes to use in other notebooks
# with open('paramTypes.cpkl', 'wb') as fh:
#     cloudpickle.dump(paramTypes, fh)


# <a name="defineEnsemblePriors"></a>
# # Define sensible priors and normalised distances in parameter space for Bayesian exploration

# In[33]:


ensembleParamPriors = OrderedDict()
for paramName in paramTypes["ensemble"]:
    curParam = paramTable_default[paramName].loc[0]
    ensembleParamPriors[paramName] = OrderedDict(
        type = type(curParam),
        size =  curParam.shape if isinstance(curParam, np.ndarray) else (1,),
        defaultVal = curParam
    )
    
    
    
# Define "sampleFunc" sampling functions for all non-standard things,
# then all the rest just define as underdispersed gamma distributions with mean matching the default value

# Also define "distFunc", which helps us define distance from other samples 
# (makes sure the individual dimensions are on similar distance scales for the GP regression, so we "zscore" each)

# Helper func for zscoreing scaled beta dist
def getBetaStd(m,a,b):
    return m * np.sqrt( (a*b)/((a+b)**2 * (a+b+1)))



# Staring day of simulation
# -------------------------

# Just pick uniformly randomly from a range
ensembleParamPriors["realStartDate"]["sampleFunc"] = (
    lambda : random.sample(list(pd.date_range('2020-01-30','2020-03-05', freq='D')),1)[0]
)
ensembleParamPriors["realStartDate"]["distFunc"] = (
    lambda x,y: np.abs((x-y).days)/(np.sqrt(1./12.*36**2))
)


# SOCIAL MIXING
# -------------

# For baseline social mixing we can just take the given values, should be reasonably well established
ensembleParamPriors['trFunc_newInfections_params_ageSocialMixingBaseline']["sampleFunc"] = (
    lambda d=ensembleParamPriors['trFunc_newInfections_params_ageSocialMixingBaseline']["defaultVal"]: d
)
ensembleParamPriors['trFunc_newInfections_params_ageSocialMixingBaseline']["distFunc"] = lambda x,y: 0.

# For social distancing things are a little less clear. 
# We'll assume that the general mixing ratio changes amongst age groups caused by social distancing are correct,
# And we just multiply the overall level of social interactions

ensembleParamPriors['trFunc_newInfections_params_ageSocialMixingDistancing']["sampleFunc"] = (
    lambda d = ensembleParamPriors['trFunc_newInfections_params_ageSocialMixingDistancing']["defaultVal"]: (
        d
        *
        (np.random.beta(2,3) * 2) # Mostly smaller than 1 values, but can be larger, mean is 0.8 like this
    )
)
ensembleParamPriors['trFunc_newInfections_params_ageSocialMixingDistancing']["distFunc"] = (
    # Distances are all caused by the beta prior, so let's just figure out the distance in "beta distribution space",
    # and normalise the variance to 1
    lambda x,y,d=ensembleParamPriors['trFunc_newInfections_params_ageSocialMixingDistancing']["defaultVal"]: (
        # Get abs(first beta sample - second beta sample) and divide by expected std sqrt((ab/((a+b)^2*(a+b+1)))**2) = 0.4 
        np.abs(
            np.mean(x/d)
            -
            np.mean(y/d)
        ) 
        / getBetaStd(2,2,3)
    )
)


# Infectiusness
# --------------

# This the one we're most uncertain about, define a broad prior. 
# We do expect E and IN to be not very infectious (non-symptomatic cases)
# Whereas we do expect IS1 and IS2 to be much more infectious (symptomatic cases)

ensembleParamPriors['trFunc_newInfections_params_transmissionInfectionStage']["sampleFunc"] = (
    lambda : (
        np.stack([
            # Very low E state infectiousness with max rate 0.2
            0.2 * np.random.beta(1,5),
            # Low IN state infectiousness with max rate 0.5
            0.5 * np.random.beta(1,3),
            # High IS1 state infectiousness with max rate 2. (average 0.8)
            2* np.random.beta(2,3),
            
            # High IS2 state infectiousness with max rate 1.6
            1.6* np.random.beta(2,3)
        ])
    )
)

# zscore all dims independently then average the 4 distances
ensembleParamPriors['trFunc_newInfections_params_transmissionInfectionStage']["distFunc"] = (
    # Distances are all caused by the beta prior, so let's just figure out the distance in "beta distribution space",
    # and normalise the variance to 1
    lambda x,y: np.mean(
        # Get abs(first beta sample - second beta sample) and divide by expected std sqrt((ab/((a+b)^2*(a+b+1)))**2) = 0.4 
        np.abs(x - y) 
        / np.stack([getBetaStd(0.2,1,5), getBetaStd(0.5,2,3), getBetaStd(2,2,3), getBetaStd(1.6,2,3)])
    )
)





# Disease progression
# -------------------

# This one's based on the CHESS dataset exactly, so we only allow for little (but independent) variation around the computed values
ensembleParamPriors['trFunc_diseaseProgression_params_caseFatalityRatioHospital_given_COVID_by_age']["sampleFunc"] = (
    lambda d=ensembleParamPriors['trFunc_diseaseProgression_params_caseFatalityRatioHospital_given_COVID_by_age']["defaultVal"] : (
        d    
        *
        np.stack([ # make for each parameter independent beta numbers in range 0.75-1.25 to multiply
            0.5*np.random.beta(5,5)+0.75
            for _ in range(len(d))
        ])
    )
)
ensembleParamPriors['trFunc_diseaseProgression_params_caseFatalityRatioHospital_given_COVID_by_age']["distFunc"] = (
    lambda x,y,d=ensembleParamPriors['trFunc_diseaseProgression_params_caseFatalityRatioHospital_given_COVID_by_age']["defaultVal"]: np.mean(
        np.abs(
            x/d
            -
            y/d
        )
        / getBetaStd(0.5, 5, 5)
    ) # LET's do importance weigthing outside! * 0.25 # Because this is a strong assumption based on data, and with little variation in samples, I want the distances in sample space to matter less!
)


# This one is simply an assumption that younger people recover more / faster, let's just sample it as randn
ensembleParamPriors['trFunc_diseaseProgression_params_ageRelativeRecoverySpeed']["sampleFunc"] = (
    lambda d = ensembleParamPriors['trFunc_diseaseProgression_params_ageRelativeRecoverySpeed']["defaultVal"]: np.clip(
        0.5 * np.sort(
                np.random.randn(*d.shape)
            )[::-1],
        -0.99, np.inf
    )
)
ensembleParamPriors['trFunc_diseaseProgression_params_ageRelativeRecoverySpeed']["distFunc"] = (
    # we need to "undo" the sorting operation that resulted in x and y, thus take all pairwise distances, which 
    # should simply come from normal distribution
    lambda x,y: 0. if (x == y).all() else (np.mean( 
        np.abs(x[:,np.newaxis] - y[np.newaxis,:])
    ) / 0.5) # to zscore
)


# HOSPITALISATION
# ----------------


# These next two comes fully from NHS HES data, don't change it for now!
ensembleParamPriors['trFunc_HospitalAdmission_params_ageHospitalisationRateBaseline']["sampleFunc"] = (
    lambda d=ensembleParamPriors['trFunc_HospitalAdmission_params_ageHospitalisationRateBaseline']["defaultVal"]: d
)
ensembleParamPriors['trFunc_HospitalAdmission_params_ageHospitalisationRateBaseline']["distFunc"] = (
    lambda x,y: 0.
)
ensembleParamPriors['trFunc_HospitalDischarge_params_ageHospitalisationRecoveryRateBaseline']["sampleFunc"] =(
    lambda d=ensembleParamPriors['trFunc_HospitalDischarge_params_ageHospitalisationRecoveryRateBaseline']["defaultVal"]: d
)
ensembleParamPriors['trFunc_HospitalDischarge_params_ageHospitalisationRecoveryRateBaseline']["distFunc"] =(
    lambda x,y: 0.
)


# These ones are very important, and largely unknown, so let's define broad priors and explore them!

# This is extra rate of being hospitalised because of being in infection states E,IN, IS1, IS2, BEFORE AGE ADJUSTMENT
ensembleParamPriors['trFunc_HospitalAdmission_params_infToHospitalExtra']["sampleFunc"] = (
    lambda : (
        np.stack([
            # Very low E state with max rate 0.01 (1% chance)
            0.01 * np.random.beta(1,7),
            # Very low IN state with max rate 0.02 
            0.02 * np.random.beta(1,6),
            # Slighty higher very broad IS1 state infectiousness with max rate 0.1 (average 5%)
            0.1* np.random.beta(1.5,1.5),
            
            # Slighty higher very broad IS2 state infectiousness with max rate 0.1 (average 5%)
            0.1* np.random.beta(1.5,1.5),
        ])
    )
)
ensembleParamPriors['trFunc_HospitalAdmission_params_infToHospitalExtra']["distFunc"] = (
    # Distances are all caused by the beta prior, so let's just figure out the distance in "beta distribution space",
    # and normalise the variance to 1
    lambda x,y: np.mean(
        # Get abs(first beta sample - second beta sample) and divide by expected std sqrt((ab/((a+b)^2*(a+b+1)))**2) = 0.4 
        np.abs(x - y) 
        / np.stack([getBetaStd(0.01,1,7), getBetaStd(0.02,1,6), getBetaStd(0.1,1.5,1.5), getBetaStd(0.1,1.5,1.5)])
    )
)


# This one adjusts the above extra risks to differ amongst different age groups. 
# The actual observed admissions are based on data, but we still want to vary them slightly and independetly
# Instead of multiplication, here we'll add normally distributed rates
ensembleParamPriors['trFunc_HospitalAdmission_params_ageRelativeExtraAdmissionRiskToCovid']["sampleFunc"] = (
    lambda d = ensembleParamPriors['trFunc_HospitalAdmission_params_ageRelativeExtraAdmissionRiskToCovid']["defaultVal"]: (
        np.clip( d +  0.2 * np.random.randn(*d.shape), -0.99, np.inf)            
    )
)

ensembleParamPriors['trFunc_HospitalAdmission_params_ageRelativeExtraAdmissionRiskToCovid']["distFunc"] = (
    # Simple average distance of already-zscored (to 0.5 std) variables x-y
    lambda x,y: np.mean( 
        np.abs(x - y)
    ) / 0.5 # to zscore
)


# TESTING
# -------

# Set some broad priors on overall PCR efficiency
ensembleParamPriors['trFunc_testing_params_inpFunc_testSpecifications_params_PCR_FNR_I1_to_R2']["sampleFunc"] = (
    lambda : (
        np.stack([
            # E between 0.4-1.
            0.6 * np.random.beta(1.5,1.5) + 0.4,
            # IN between 0.2-0.6
            0.4 * np.random.beta(1.5,1.5) + 0.2,
            # IS1 between 0.01-0.35
            0.34 * np.random.beta(1.5,1.5) + 0.01,
            # IS2 between 0.05-0.55
            0.5 * np.random.beta(1.5,1.5) + 0.05,
            # R1 between 0.3-0.7
            0.4 * np.random.beta(1.5,1.5) + 0.3,
            # R1 between 0.4-1.
            0.6 * np.random.beta(1.5,1.5) + 0.4
        ])
    )
)
ensembleParamPriors['trFunc_testing_params_inpFunc_testSpecifications_params_PCR_FNR_I1_to_R2']["distFunc"] = (
    # Distances are all caused by the beta prior, so let's just figure out the distance in "beta distribution space",
    # and normalise the variance to 1
    lambda x,y: np.mean(
        # Get abs(first beta sample - second beta sample) and divide by expected std sqrt((ab/((a+b)^2*(a+b+1)))**2) = 0.4 
        np.abs(x - y) 
        / np.stack(
            [getBetaStd(0.6,1.5,1.5), getBetaStd(0.4,1.5,1.5), getBetaStd(0.34,1.5,1.5), getBetaStd(0.5,1.5,1.5), getBetaStd(0.4,1.5,1.5), getBetaStd(0.6,1.5,1.5)])
    )
)



# ALL OTHER SINGLE SCALAR NUMERIC PARAMETER VALUES (largely "days")
# --------------------------------------------------------
# We just define them as multiplied by beta distributed numbers between 0.5-1.5 to adjust the rates
for key in ensembleParamPriors.keys():
    if "sampleFunc" not in ensembleParamPriors[key]:
            ensembleParamPriors[key]["sampleFunc"] = (
                lambda d=copy.deepcopy(ensembleParamPriors[key]["defaultVal"]), t=copy.deepcopy(ensembleParamPriors[key]["type"]): np.cast[t](
                    (1. * np.random.beta(3.,3.) + 0.5)
                    *
                    d
                )
            )
            ensembleParamPriors[key]["distFunc"] = (
                lambda x,y,d=copy.deepcopy(ensembleParamPriors[key]["defaultVal"]): (
                        # Get abs(first beta sample - second beta sample) and divide by expected std sqrt((ab/((a+b)^2*(a+b+1)))**2) = 0.4 
                        np.abs(
                            (x / d)
                            - 
                            (y / d)
                        ) 
                        / getBetaStd(1.,3.,3.)
                    )
            )

        
# Naive pairwise dist implementation for arbitary objects of the same type
def getPairwiseDistsSingleDim(distFunc, listOfObjects, listOfObjectsOther=None):#, symmetricDist=True): # we assume symmetric distance
    if listOfObjectsOther is None:
        listOfObjectsOther = listOfObjects 
        squareOutput = True
    else:
        squareOutput = False
    
    out = np.zeros((len(listOfObjects), len(listOfObjectsOther)))
    
    for i in range(len(listOfObjects)):
        j_start = i+1 if squareOutput else 0
        for j in range(j_start, len(listOfObjectsOther)):
            out[i,j] = distFunc(listOfObjects[i], listOfObjectsOther[j])
            
    if squareOutput:
        out = out + out.T
                    
    return out
    
    
# # Make sure that all simulation average distances (except the deterministic samplers) are on the same scale: 
# print("Average distances between samples for 100 samples, should be ~1.1 or 0. for static things")
# print("-----------------------------------------------------------------------------------------")
# for key in ensembleParamPriors:
#     tmp = getPairwiseDistsSingleDim(
#         ensembleParamPriors[key]["distFunc"],
#         [ensembleParamPriors[key]["sampleFunc"]() for i in range(100)]
#     )

#     print(
#         np.mean(tmp[np.triu_indices_from(tmp, k=1)]), key # this should be around 1.1 for every variable, the average distance of z-scored floats
#     )


# In[34]:


def getEnsembleParamSample(num_samples = 1, ensembleParamPriors=ensembleParamPriors):
    """
    Ensemble parameters do not have to live on a grid, just try and sample them from given priors
    """
    if num_samples == 1:
        newParamDict = OrderedDict()
        for key in ensembleParamPriors:
            newParamDict[key] = ensembleParamPriors[key]["sampleFunc"]()

        #return newParamDict
        return pd.DataFrame([newParamDict.values()], columns=newParamDict.keys())
    else:
        for i in range(num_samples):
            if i==0:
                df_out = getEnsembleParamSample(num_samples = 1, ensembleParamPriors=ensembleParamPriors)
            else:
                df_out = df_out.append(getEnsembleParamSample(num_samples = 1, ensembleParamPriors=ensembleParamPriors))
        
        df_out = df_out.reset_index(drop=True)
        return df_out
    


# In[35]:


def getEnsembleParamSampleDistance(df1, df2=None, weighting = None, ensembleParamPriors=ensembleParamPriors):
    """
    Given two sets of ensemble parameter samples in pandas dataframes (as output by getEnsembleParamSample), 
    # returns the pairwise distance between all
    This relies on having individual distance metrics in each parameter type `axis`
    """
    if df2 is not None:
        out = np.zeros((len(df1), len(df2)))
    else:
        df2 = df1
        out = np.zeros((len(df1), len(df2)))

    # If no pre-defined weighting of columns' distances (default is 1), then use equal weigthing
    weighting = OrderedDict() if weighting is None else weighting
        
    # Go through each key, 
    # get the within-axis pairwise distances
    # add their squares to the out matrix (we'll take the sqrt of the mean afterwards to get overall eucl distance)
    for key in ensembleParamPriors:
        cur_weight = weighting[key] if key in weighting else 1.
        out += (
            cur_weight*
            getPairwiseDistsSingleDim(
                distFunc = ensembleParamPriors[key]["distFunc"],
                listOfObjects = list(df1[key]),
                listOfObjectsOther = list(df2[key])
            )
        )**2
        
      

    out = np.sqrt(out)
        
    return out
            
                
        
    
        


# In[36]:


# For all target likelihoods we define initial mean and variance (largely based on earlier random run results, see histogtam in plotEnsembles!)
def getGPR_prediction(
    df_sor, 
    df_new, 
    dist_func = getEnsembleParamSampleDistance,
    target_likelihoods = OrderedDict( # descibes the column name in df_sor, and the mean function and output variance of the kernel for that likelihood
        likelihood_0 = OrderedDict(mean=0., var=400.**2),
        likelihood_1 = OrderedDict(mean=0., var=600.**2),
        likelihood_2 = OrderedDict(mean=0., var=500.**2),
        likelihood_3 = OrderedDict(mean=0., var=1500.**2),
        likelihood_total = OrderedDict(mean=0., var=5000.**2)
#         likelihood_0 = OrderedDict(mean=-200., var=100.**2),
#         likelihood_1 = OrderedDict(mean=-500., var=200.**2),
#         likelihood_2 = OrderedDict(mean=-300., var=100.**2),
#         likelihood_3 = OrderedDict(mean=-1700., var=500.**2),
#         likelihood_total = OrderedDict(mean=-3000., var=1000.**2)
    ), 
    kernel_dist_scale2 = 10., # (average dist^2 is 48, so this sets its to exp-2=0.13 on average)
    Ksor_inv=None, # pass the precomputed inverse kernel matrix on df_sor if it hasn't changed
    return_Ksor_only = False
    ):
    
    # Compute the inverse kernel matrix if doesn't exist yet
    if Ksor_inv is None:
        return_Ksor_inv = True
        Ksor_inv = np.exp(-dist_func(df_sor)**2/(2.*kernel_dist_scale2))
        Ksor_inv = np.linalg.inv(Ksor_inv + 1e-12*np.eye(*Ksor_inv.shape))
        if return_Ksor_only:
            return Ksor_inv
    else:
        return_Ksor_inv = False
        
    # Set the new only kernel (independent points)
    Knew = np.eye(len(df_new))
    
    # Comute the cross-kernel
    Kcross = np.exp(-dist_func(df_new, df_sor)**2/(2.*kernel_dist_scale2))
    
    # for each target likelihood, add the predicted likelihoods as columns to df_new
    
    for lik_col in target_likelihoods.keys():
        # Fill in the predicted mean
        df_new["GPR_pred_mean_"+lik_col] = (
            target_likelihoods[lik_col]["mean"] + # mean function
            (
                np.matmul(
                    Kcross,
                    np.matmul(
                        Ksor_inv,
                        df_sor[lik_col].values - target_likelihoods[lik_col]["mean"] # observed - mean function
                    )
                )
            )
        )
        
        # Fill in the predicted standard deviation (reduced compared to the baseline given in target_likelihoods[var])
        df_new["GPR_pred_std_"+lik_col] = np.sqrt(
            target_likelihoods[lik_col]["var"] * # output variance
            np.diag(
                Knew
                -
                np.matmul(
                    Kcross,
                    np.matmul(
                        Ksor_inv,
                        Kcross.T
                    )
                )
            )
        )

    if return_Ksor_inv:
        return df_new, Ksor_inv
    else:
        return df_new
    


# In[38]:


# # Uncomment below to run example sampling + GP regression

# # Sample a few param sets as an example
# tmp = getEnsembleParamSample(10)

# # Test the GP regression from above
# for i in range(4):
#     tmp["likelihood_"+str(i)] = -500+100*np.random.randn(10,)
    
# tmp["likelihood_total"] = tmp.loc[:,list(tmp.columns.str.startswith("likelihood_"))].sum(1)
    
# for i in range(3):
#     if i == 0:
#         tmp_new =  getEnsembleParamSample()
#     else:
#         tmp_new = tmp_new.append(getEnsembleParamSample())
        
# tmp_new = tmp_new.reset_index(drop=True)

# out1, Ksor_inv = getGPR_prediction(
#     tmp, tmp_new
# )

# out2 = getGPR_prediction(
#     tmp, tmp_new, Ksor_inv= Ksor_inv 
# )

# out2


# # Evaluate simulation likelihood given data

# ## Load datasets

# ### NHS England deaths dataset

# In[40]:


# NHS daily deaths report (about 24 hours behind)
# TODO - might need to manually update link and column numbers (maybe not consistent across days, cannot yet automate)
# NOTE - NHS started deleting their old files, and now only the latest seems to be available...
df_UK_NHS_daily_COVID_deaths = pd.read_excel(
    "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2020/04/COVID-19-total-announced-deaths-20-April-2020.xlsx",
    sheet_name = "COVID19 total deaths by age",
    index_col=0,
    usecols = "B,E:AX",
    skip_rows = range(17),
    nrows = 22
).iloc[14:].transpose().set_index("Age group").rename_axis(index = "Date", columns = "AgeGroup")

df_UK_NHS_daily_COVID_deaths.index = pd.to_datetime(df_UK_NHS_daily_COVID_deaths.index, format="%Y-%m-%d")

df_UK_NHS_daily_COVID_deaths = df_UK_NHS_daily_COVID_deaths.drop(df_UK_NHS_daily_COVID_deaths.columns[:2], axis=1)

df_UK_NHS_daily_COVID_deaths

# Ignore very recent unreliable data points
df_UK_NHS_daily_COVID_deaths = df_UK_NHS_daily_COVID_deaths.loc[
            df_UK_NHS_daily_COVID_deaths.index <= CONST_DATA_CUTOFF_DATE]

df_UK_NHS_daily_COVID_deaths


# ### NHS England CHESS - COVID hospitalisations - dataset

# In[41]:


# Load the aggregate data (ask @SebastianVollmer for aggregation details and or data access!)
df_CHESS = pd.read_csv("/mnt/efs/data/CHESS_Aggregate20200417.csv").drop(0)

# Clean the dates and make them index
# The "1899-12-30" is simply total, ignore that.
# The 2020-09-03, 2020-10-03, 2020-11-03, 2020-12-03 are parsed wrong and are march 09-12 supposedly.
# The data collection is only officially started across england on 09 March, the February dates seem empty, delete.
# Rest are ok

df_CHESS.index = pd.to_datetime(df_CHESS["DateOfAdmission"].values,format="%d-%m-%Y")

# Ignore too old and too recent data points
df_CHESS = df_CHESS.sort_index().drop("DateOfAdmission", axis=1).query('20200309 <= index <= ' + CONST_DATA_CUTOFF_DATE)

df_CHESS


# In[42]:


df_CHESS.columns


# In[43]:


# Get the hospitalised people who tested positive for COVID, using cumsum (TODO: for now assuming they're all still in hospital)
df_CHESS_newCOVID = df_CHESS.loc[:,df_CHESS.columns.str.startswith("AllAdmittedPatientsWithNewLabConfirmedCOVID19")]

df_CHESS_newCOVID 


# In[ ]:





# ## Define likelihoods

# In[44]:


# Load an example simulation:
simExample = np.load("/mnt/efs/results/run_20200408T195337/outTensor_20200408T195337_slr7ep10hy0q9iyr3k36.npy")
simExample_newOnly = np.load("/mnt/efs/results/run_20200408T195337/outTensor_20200408T195337_slr7ep10hy0q9iyr3k36_newOnly.npy")


# In[45]:


# We expect 
def joinDataAndSimCurves(
    df_curves, # a pandas dataframe with dates as index, and each column is a curve
    simCurves, # curves x time np array
    simStartDate, # curves start dates
    simCurvesNames = None,
    fulljoin = False # if true, then one keeps all dates in the simulation, instead of just the ones in the date 
    ):
    
    out_df = copy.deepcopy(df_curves)
    
    simCurveIndex = pd.date_range(start=simStartDate, freq='D', periods=simCurves.shape[1])
    
    if simCurvesNames is None:
        simCurvesNames = ["simCurve_{}".format(i) for i in range(simCurves.shape[0])]
    
    join_type = "outer" if fulljoin else "left"
    
    for i, curve in enumerate(simCurves):
        out_df = out_df.join(
            pd.DataFrame(
                index = simCurveIndex,
                data = simCurves[i],
                columns=[simCurvesNames[i]]
            ),
            how = join_type
        )
    
    return out_df


# In[46]:


def getNegBinomParams(mu, alpha):
    """ 
    From https://stats.stackexchange.com/questions/260580/negative-binomial-distribution-with-python-scipy-stats
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    Parameters
    ----------
    mu : float 
       Mean of NB distribution.
    alpha : float
       Overdispersion parameter used for variance calculation.

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    var = mu + alpha * mu ** 2
    p = (var - mu) / var
    r = mu ** 2 / (var - mu)
    return r, p

def NegBinom_logpmf(a, m, x):
    binom_vec = np.array([binom(x1 + a - 1, x1) for x1 in x])
    logpmf = np.log(binom_vec) + a * np.log(a / (m + a))  + x * np.log((m / (m + a)))
    return logpmf


# ### Deaths in hospitals

# In[47]:


def likFunc_deaths(
    sim, # use newOnly for deaths by day needed here
    simStartDate, 
    df,
    sumAges = True,
    outputDataframe = False, # If true, outputs the data-curves and simulated curves instead of likelihood, for plotting
    fulljoin = False # if true, then one keeps all dates in the simulation, instead of just the ones in the date
    ):

        
    # Get deaths by day in simulation in hospitals for people with positive tests
    deaths_Sim_byAge = np.sum(sim[:,-1,2,:,:],axis=(1))
    
    
    if sumAges:
        deaths_Sim = np.sum(deaths_Sim_byAge,axis=0, keepdims=True) 
        deaths_data = pd.DataFrame(df.sum(1))
    else:
        # Change the grouping of ages to be same as dataset
        deaths_Sim_byAge_regroup = regroup_by_age(
            deaths_Sim_byAge, 
            fromAgeSplits = np.arange(10,80+1,10), 
            toAgeSplits = np.arange(20,80+1,20)
        )
        
        deaths_Sim = deaths_Sim_byAge_regroup
        deaths_data = df
        
    
    
    # Join the two dataframes to align in time
    df_full = joinDataAndSimCurves(
        df_curves = deaths_data, # a pandas dataframe with dates as index, and each column is a curve
        simCurves = deaths_Sim, # curves x time np array
        simStartDate = simStartDate, # curves start dates
        fulljoin = fulljoin
    )
    
    # If true, outputs the data-curves and simulated curves instead of likelihood, for plotting
    if outputDataframe:
        return df_full
    
    # We assume the order of columns in data and in simCurves are the same!
    #return df_full
    return np.nansum(
        NegBinom_logpmf(2., 
                        # Select all simCurve columns and reshape to a single vector
                        m = 1e-8+np.reshape(df_full.loc[:,df_full.columns.str.startswith("simCurve_")==True].values,-1), 
                        # Select all data columns and reshape to a single vector
                        x = np.reshape(df_full.loc[:,(df_full.columns.str.startswith("simCurve_")==True)==False].values,-1)
                       )
    )
 


# In[48]:


likFunc_deaths(
    sim = simExample_newOnly, 
    simStartDate = '2020-02-12',
    df = copy.deepcopy(df_UK_NHS_daily_COVID_deaths),
    sumAges=True
)


# ### Test outcomes in hospitals

# In[49]:


def likFunc_newHospPositive(
    sim, # Here we'll make sure to pass the "_newOnly" tensor!
    simStartDate, 
    df,
    sumAges = True,
    outputDataframe = False, # If true, outputs the data-curves and simulated curves instead of likelihood, for plotting
    fulljoin = False # if true, then one keeps all dates in the simulation, instead of just the ones in the date
    ):
    """
    Get the number of hospitalised people testing positive each day.
    This fits well with "policyFunc_testing_symptomaticOnly" being active, which prioratises testing hospitalised people
    As per 09 April this is a very reasonable assumption
    """
    
    # Calculate the simulation curves of hospitalised people getting positive tests each day
    # TODO - run the simulation with actual test numbers each day, would make fits a LOT better.
    # Take into account the number of positive tested people who leave the hospital and add that as well 
    # (as they were also replaced by new people testing positive in the hospital!)
    
    # Change in hospital and testing positive
    newHospPositives_sim = np.sum(sim[:,:,2,1,:], axis=(1,))
    
    if sumAges:
        hospPos_Sim = np.sum(newHospPositives_sim,axis=0, keepdims=True) 
        hospPos_data = pd.DataFrame(df.sum(1))
    else:
        
        # Change the grouping of ages to be same as dataset
        
        hospPos_Sim = regroup_by_age(
            newHospPositives_sim, 
            fromAgeSplits = np.arange(10,80+1,10), 
            toAgeSplits = np.concatenate([np.array([1,5,15,25]),np.arange(45,85+1,10)])
        )
        
        hospPos_data = df
        
        
    # Join the two dataframes to align in time
    df_full = joinDataAndSimCurves(
        df_curves = hospPos_data, # a pandas dataframe with dates as index, and each column is a curve
        simCurves = hospPos_Sim, # curves x time np array
        simStartDate = simStartDate, # curves start dates
        fulljoin = fulljoin
    )
    
    # If true, outputs the data-curves and simulated curves instead of likelihood, for plotting
    if outputDataframe:
        return df_full
    
    
    # We assume the order of columns in data and in simCurves are the same!
    #return df_full
    return np.nansum(
        NegBinom_logpmf(2., 
                        # Select all simCurve columns and reshape to a single vector
                        m = 1e-8+np.reshape(df_full.loc[:,df_full.columns.str.startswith("simCurve_")==True].values,-1), 
                        # Select all data columns and reshape to a single vector
                        x = np.reshape(df_full.loc[:,(df_full.columns.str.startswith("simCurve_")==True)==False].values,-1)
                       )
    )
    


# In[50]:


likFunc_newHospPositive(
    sim = simExample_newOnly, 
    simStartDate = '2020-02-22',
    df = copy.deepcopy(
        df_CHESS_newCOVID
    ),
    sumAges=True
)


# In[51]:


# Get the likelihoods
def getLikelihoodsWithStartDates(
    sims,
    likFuncs,
    simIndices,
    dataTables,
    startDates
    ):
    
    out = np.zeros((len(likFuncs), len(startDates)))
    
    for ind, likFunc in enumerate(likFuncs):
        out[ind] = np.array([
            likFunc(
                sim = sims[simIndices[ind]], 
                simStartDate = cur_startDate,
                df = copy.deepcopy(dataTables[ind])
            )
            for cur_startDate in startDates
        ])
                
    return out


# # Parallel execution with dask

# In[52]:


client = Client("127.0.0.1:8786")


# In[ ]:


# Set up where to save and save default parameters

timeOfRunning = datetime.now().strftime("%Y%m%dT%H%M%S")

saveDir = "/mnt/efs/results/run_" + timeOfRunning + "/"
os.makedirs(saveDir, exist_ok=True)
os.chmod(saveDir, 0o777) # enable workers to write the files

# Save the default parameter dictionary that we'll merge with new inputs
paramDict_default = build_paramDict(dydt_Complete)
paramDict_default["dydt_Complete"] = dydt_Complete
paramDict_default["INIT_stateTensor_init"] = stateTensor_init
with open(saveDir+'paramDict_default.cpkl', 'wb') as fh:
    cloudpickle.dump(paramDict_default, fh)
    
with open(saveDir+'ensembleParamPriors.cpkl', 'wb') as fh:
    cloudpickle.dump(ensembleParamPriors, fh)

with open(saveDir+'getGPR_prediction_func.cpkl', 'wb') as fh:
    cloudpickle.dump(getGPR_prediction, fh)

    


# ## Run until infinity

# In[53]:


# Run parallel for each parameter setting and save to out_fname
def runAll(newParams_row, stateTensor_init=stateTensor_init, defaultDict=paramDict_default, timeOfRunning=timeOfRunning):
    # Run model 
    # Make sure the newOnly stuff is saved as well
    curDict = copy.deepcopy(defaultDict)
    curDict["debugReturnNewPerDay"] = True
    
    out = solveSystem(stateTensor_init, 
                total_days = 80, # 80 days is enough to incorporate all data!
                **paramTable_toDict(
                            # keep only relevant columns in newParams_row for this
                           newParams_row[list(set(newParams_row.columns) & set(paramDict_toTable(defaultDict).columns))].reset_index(drop=True),
                           defaultDict=copy.deepcopy(curDict)
                    )
               )
    # The out is now both the states and the cumsum
    out_newOnly = np.diff(np.concatenate([np.expand_dims(copy.deepcopy(out[0][:,:,:,:,0]),axis=4), copy.deepcopy(out[1])], axis=-1), axis=-1)
    out = out[0]
    
    
    # Compute likelihoods
    out_liks = getLikelihoodsWithStartDates(
        sims= [out, out_newOnly], 
        likFuncs = [
            lambda sim, simStartDate, df: likFunc_deaths( sim, simStartDate, df, sumAges=True),
            lambda sim, simStartDate, df: likFunc_deaths( sim, simStartDate, df, sumAges=False),
            lambda sim, simStartDate, df: likFunc_newHospPositive( sim, simStartDate, df, sumAges=True),
            lambda sim, simStartDate, df: likFunc_newHospPositive( sim, simStartDate, df, sumAges=False)
        ],
        simIndices = [
            1, 1, 1, 1
        ],
        dataTables = [
            df_UK_NHS_daily_COVID_deaths,
            df_UK_NHS_daily_COVID_deaths,
            df_CHESS_newCOVID,
            df_CHESS_newCOVID
        ],
        startDates = newParams_row['realStartDate']
    )
    
    for i in range(out_liks.shape[0]):
        newParams_row["likelihood_" + str(i)] = out_liks[i,0]
        
    newParams_row["likelihood_total"] = np.sum(out_liks)
    
    newParams_row["out_fname"] = "outTensor_" + timeOfRunning + "_" + ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(20))+".npy"
    
    
    return out, out_newOnly, newParams_row


# In[54]:


# This function returns numSelected parameter sets that we wish to evaluate, plus the remaining ones
def gpSelectionFunction(proposedNewParams, numSelected, total_lik_threshold = -5000):
    # Select not terrible ones, not necassirily best ones
    ind_GoodCandidates = list(proposedNewParams.index[proposedNewParams.GPR_pred_mean_likelihood_total > total_lik_threshold])
    if len(ind_GoodCandidates) > numSelected:
        ind_GoodCandidates = random.sample(ind_GoodCandidates, numSelected)
    
    return (
        # Good candidates
        proposedNewParams.loc[ind_GoodCandidates],
        # Remaining
        proposedNewParams.loc[list(set(proposedNewParams.index)-set(ind_GoodCandidates))],
    )


# In[ ]:


gpNewProposalIter = 192
gpNewProposalJobs = 100
gpSOD = 180
gpNewProposalMinibatchSize = 100
maxTotalEnsemble = 200000


curIndex = 0

cur_gp_ind = 0
cur_gp_batch = 0
seqs_added = 0

gp_futures = [] 

# # Submit an initial bunch of futures to serve as anchor points for GP regression
futures = []
for index in range(gpNewProposalIter+1):
    tmp_params_row = getEnsembleParamSample(ensembleParamPriors=ensembleParamPriors)
    fut = client.submit(runAll, tmp_params_row)
    futures.append(fut)


seq = as_completed(futures)
    
# Do the processing, and keep submitting new jobs on the fly
for future in seq:
    if future.status == "finished":
        out, out_newOnly, newParams_row = future.result()
            
        # Save all the files
        np.save(file = saveDir + newParams_row.at[0, "out_fname"],
                        arr= out
                    )

        np.save(file = saveDir + newParams_row.at[0, "out_fname"][:-4]+"_newOnly.npy",
                        arr= out_newOnly
                    )
        
        newParams_row.index = [curIndex]
        
        if curIndex == 0:
            paramTable_new = newParams_row
            
        else:
            paramTable_new = paramTable_new.append(newParams_row)
            
        curIndex += 1
        
        if (curIndex % 100)==0:
            # Save paramTable
            paramTable_new.to_hdf(saveDir + "paramTable_part{}".format(0), key="paramTable")
        
        # If no GP job is running, and we dont have a huge backlog submit some
        if (not gp_futures) and (curIndex > gpNewProposalIter*0.8*(cur_gp_batch+1)) and (seqs_added-curIndex<2000):
            cur_gp_batch += 1

            # Submit some GP jobs
            
            # Calculate the current inverse
            if curIndex == gpNewProposalIter: # very first time we do it blocking
                paramTable_new, Ksor_inv = getGPR_prediction(paramTable_new, paramTable_new)
            else:
                # Compute it on a random subset, this should block only for few seconds
                # Take best 100 + random rest
                curSOD_inds = (
                    list(paramTable_new.sort_values("likelihood_total").tail(100).index) + 
                    random.sample(list(paramTable_new.index), gpSOD-100)
                )
                Ksor_inv_SOD = getGPR_prediction(
                    copy.deepcopy(paramTable_new.loc[curSOD_inds]), 
                    copy.deepcopy(paramTable_new.loc[curSOD_inds]),
                    return_Ksor_only=True
                )
                
                # If we use Subset of Regressions instead of subset of Data, compute the projections as well!
                # paramTable_new.loc[list(set(paramTable_new.index)-set(curSOR_inds))]

            # Submit the GP jobs (this takes a while, would be good to have some more jobs before)
            for i_gp in range(gpNewProposalJobs):
                fut = client.submit(getGPR_prediction, 
                              copy.deepcopy(paramTable_new.loc[curSOD_inds]), 
                              getEnsembleParamSample(gpNewProposalMinibatchSize, ensembleParamPriors=ensembleParamPriors), 
                              Ksor_inv=Ksor_inv_SOD)
                gp_futures.append(fut)
                
                
        # Check every time if all GP jobs finished, but otherwise don't wait for them!
        if gp_futures and all([gp_fut.status == "finished" for gp_fut in gp_futures]):
                
            # This is a blocking step to get new proposals
            for gp_future in as_completed(gp_futures):
                if gp_future.status == "finished":
                    outNewParams = gp_future.result()

                    if cur_gp_ind == 0:
                        proposedNewParams = outNewParams
                    else:
                        proposedNewParams = proposedNewParams.append(outNewParams)
                        
                    cur_gp_ind+=1

                client.cancel(gp_future)
            
            gp_futures = []
                
            proposedNewParams = proposedNewParams.reset_index(drop=True)
            
            if len(proposedNewParams)>100000:
                proposedNewParams = proposedNewParams.loc[0:100000]
                
            # Submit new jobs based on the proposed params, keep the rest for use later
            submitNewParams, proposedNewParams = gpSelectionFunction(
                proposedNewParams, 
                numSelected=gpNewProposalIter,
                total_lik_threshold=np.quantile(paramTable_new.loc[:,"likelihood_total"],0.75)
            )
            
            print("Submitting {}/{} new jobs based on GP results".format(len(submitNewParams), gpNewProposalIter))
            
            submitNewParams = submitNewParams.reset_index(drop=True)
            for i in range(len(submitNewParams)):
                tmp = copy.deepcopy(submitNewParams.loc[i:i])
                tmp = tmp.reset_index(drop=True)
                new_future = client.submit(runAll, tmp)
                seq.add(new_future)
                seqs_added += 1

    client.cancel(future)
    
    # Check if we're running out of jobs, add some random ones
    if curIndex - seqs_added > 50:
        print("adding random jobs while waiting for GP to propose new ones...")
        for i11 in range(30):
            tmp_params_row = getEnsembleParamSample(ensembleParamPriors=ensembleParamPriors)
            new_future = client.submit(runAll, tmp_params_row)
            seq.add(new_future)
            seqs_added += 1
            
    if curIndex > maxTotalEnsemble:
        client.cancel(futures)
        client.cancel(gp_futures)
        break
    


# In[ ]:


client.cancel(futures)


# In[ ]:


client.cancel(gp_futures)


# In[ ]:


client.close()

