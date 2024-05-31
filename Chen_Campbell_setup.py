# Setup Code for Chen and Campbell's Whole Brain version

# Original Chen and Capmbell model from: Chen, L., Campbell, S.A. Exact mean-field models
# for spiking neural networks with adaptation. J Comput Neurosci 50, 445–469 (2022).
# https://doi.org/10.1007/s10827-022-00825-9

# Code By Adrià Puig

import numpy as np

import WholeBrain.Models.Chen_Campbell_Whole_Brain_version as CN
import WholeBrain.Integrators.HeunStochastic as integrator
integrator.neuronalModel = CN
integrator.verbose = False

import matplotlib.pyplot as plt
import scipy.io as sio

import WholeBrain.Utils.BOLD.BOLDHemModel_Stephan2008 as Stephan2008
import WholeBrain.Utils.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2008
simulateBOLD.warmUp = True

import WholeBrain.Optimizers.ParmSweep as optim1D
optim1D.simulateBOLD = simulateBOLD
optim1D.integrator = integrator

# ------------------------------------------------
# Chen and Campbell's model works in seconds, not in milliseconds, so
# all integration parameters should be adjusted accordingly...
# ------------------------------------------------
simulateBOLD.dt = 0.001
simulateBOLD.Tmax = 176.  # This is the length, in seconds
# simulateBOLD.dtt = 1.  # We are NOT using milliseconds
# simulateBOLD.t_min = 10 * simulateBOLD.TR
# simulateBOLD.recomputeTmaxneuronal() <- do not update Tmaxneuronal this way!
simulateBOLD.warmUpFactor = 0.1
# simulateBOLD.Toffset = 30.
simulateBOLD.Tmaxneuronal = int(simulateBOLD.Tmax * simulateBOLD.TR)
integrator.ds = 0.001  # record every TR seconds

import WholeBrain.Observables.BOLDFilters as BOLDFilters
BOLDFilters.TR = 2.
BOLDFilters.flp = 0.02
BOLDFilters.fhi = 0.1
import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.swFCD as swFCD

from Chen_Campbell_load_data import *

def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    print("\n\nRecompiling signatures!!!")
    CN.recompileSignatures()
    integrator.recompileSignatures()

def transformEmpiricalSubjects(simulations, NumSubjects):
    transformed = {}
    for s in range(NumSubjects):
        transformed[s] = simulations[0, s]  # LR_version_symm(simulations[0, s])
    return transformed

print(f"Loading {SC_path}/")
sc80 = sio.loadmat(SC_path)['SC_dbs80FULL']
C = sc80/np.max(sc80)*0.2  # Normalization...

#fMRI data
task = 'EMOTION'

# Creem una nova matriu per emmagatzemar les dades de la tasca 'EMOTION' per als 20 subjectes
simulations = np.empty((1, 20), dtype=object)

# Emplenem la matriu amb les dades dels 20 subjectes per a la tasca 'EMOTION'
for i, subject_data in enumerate(timeseries[task].values()):
    simulations[0, i] = subject_data

for i in range(simulations.shape[1]):
    simulations[0, i] = simulations[0, i][:, :176]

(N, Tmax) = simulations[0, 0].shape  # N = number of areas; Tmax = total time

print(f'timeseries is {simulations.shape} and each entry has N={N} regions and Tmax={Tmax}')

NumSubjects = 20  # Number of Subjects in empirical fMRI dataset, originally 20...
print(f"Simulating {NumSubjects} subjects!")

recompileSignatures()

tc_transf_REST = transformEmpiricalSubjects(simulations, NumSubjects)
