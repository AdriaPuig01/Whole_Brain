# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Implementation of the Heun Integrator
# Based on the code from TVB:
#     It is a simple example of a predictor-corrector method. It is also known as
#     modified trapezoidal method, which uses the Euler method as its predictor.
#     And it is also a implicit integration scheme.
#
#     [1] Kloeden and Platen, Springer 1995, *Numerical solution of stochastic
#         differential equations.
#
#     From [1]:
#         .. math::
#             X_{n+1} &= X_n + dt (dX(t_n, X_n) +
#                                  dX(t_{n+1}, \tilde{X}_{n+1})) / 2 \\
#             \tilde{X}_{n+1} &= X_n + dt dX(t_n, X_n)
#         cf. Equation 1.11, page 283.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
import numpy as np
from WholeBrain.Utils import numTricks as iC
import WholeBrain.Stimuli.constant as cons
from numba import jit

print("Going to use the Heun Integrator...")

neuronalModel = None  # To be able to choose the model externally...
verbose = True


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    # # initBookkeeping.recompile()
    # neuronalModel.recompileSignatures()
    # recordBookkeeping.recompile()
    # integrationStep.recompile()
    pass

# Functions to convert the stimulus from a function to an array
# --------------------------------------------------------------------------
stimuli = cons  # To add some stimuli, if needed...
allStimuli = None
def initStimuli(dt, Tmaxneuronal):
    global allStimuli
    tValues = np.arange(0, Tmaxneuronal, dt)
    if stimuli is not None:
        allStimuli = np.array(list(map(stimuli.stimulus, tValues)))
    else:
        allStimuli = np.zeros(len(tValues))


# bookkeeping vars & methods -> Just forward them to the neuronal model we are using...
# ==========================================================================
# ==========================================================================
# ==========================================================================
# Bookkeeping variables of interest...
# --------------------------------------------------------------------------
ds = 1  # downsampling stepsize
# # @jit(nopython=True)
def initBookkeeping(N, tmax):
    # global curr_xn, curr_rn, nn
    # global curr_obsVars
    # curr_xn = np.zeros((int(tmax), N))
    # curr_rn = np.zeros((int(tmax), N))
    obsVars = neuronalModel.numObsVars()
    timeElements = int(tmax/ds) + 1  # the last +1 because of isClose roundings...
    return np.zeros((timeElements, obsVars, N))


@jit(nopython=True)
def recordBookkeeping(t, obsVars, curr_obsVars):
    # global curr_obsVars
    if iC.isInt(t/ds):
        nn = int(np.round(t/ds))  # it is an int-ish...
        curr_obsVars[nn,:,:] = obsVars[:,:]
    return curr_obsVars


# ==========================================================================
# ==========================================================================
# ==========================================================================
# Heun Integration
# --------------------------------------------------------------------------
# sigma = 0.01
clamping = True
@jit(nopython=True)
def integrationStep(simVars, dt, stimulus, G, Wjump_2):  #, curr_obsVars, doBookkeeping):
    def doClamping(simVariables):
        if clamping:
            simVariables = np.where(simVariables > 100., 100., simVariables)  # clamp values to 0..1
            simVariables = np.where(simVariables < 0., 0., simVariables)
        return simVariables

    dvars_obsVars = neuronalModel.dfun(simVars, stimulus, G, Wjump_2)
    dvars = dvars_obsVars[0]; obsVars = dvars_obsVars[1]  # cannot use unpacking in numba...

    inter = simVars + dt * dvars
    inter = doClamping(inter)

    dvars_obsVars = neuronalModel.dfun(inter, stimulus, G, Wjump_2)
    dvars2 = dvars_obsVars[0]; obsVars = dvars_obsVars[1]  # cannot use unpacking in numba...
    dX = (dvars + dvars2) * dt / 2.0

    simVars = simVars + dX
    simVars = doClamping(simVars)

    return simVars, obsVars


# # @jit(nopython=True)
def integrationLoop(dt, Tmaxneuronal, simVars, G, Wjump_2, doBookkeeping, curr_obsVars):
    # Variables:
    # dt = integration time step in milliseconds
    # Tmaxneuronal = total time to integrate in milliseconds
    for t in np.arange(0, Tmaxneuronal, dt):
        stimulus = allStimuli[int(t / dt)]
        simVars_obsVars = integrationStep(simVars, dt, stimulus, G, Wjump_2)
        simVars = simVars_obsVars[0]; obsVars = simVars_obsVars[1]  # cannot use unpacking in numba...
        if doBookkeeping:
            curr_obsVars = recordBookkeeping(t, obsVars, curr_obsVars)
    return simVars, curr_obsVars


# # @jit(nopython=True)
def integrate(dt, Tmaxneuronal, simVars, G, Wjump_2, doBookkeeping = True):
    # numSimVars = simVars.shape[0]
    N = simVars.shape[1]  # N = neuronalModel.SC.shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    curr_obsVars = initBookkeeping(N, Tmaxneuronal)
    return integrationLoop(dt, Tmaxneuronal, simVars, G, Wjump_2, doBookkeeping, curr_obsVars)


# ==========================================================================
# ==========================================================================
# ==========================================================================
def simulate(dt, Tmaxneuronal, G, Wjump_2):
    if verbose:
        print("Simulating...", flush=True)
    N = neuronalModel.getParm('N').shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    simVars = neuronalModel.initSim(N)
    initStimuli(dt, Tmaxneuronal)
    simVars, obsVars = integrate(dt, Tmaxneuronal, simVars, G, Wjump_2)
    return obsVars


def warmUpAndSimulate(dt, Tmaxneuronal, TWarmUp=10000):
    N = neuronalModel.getParm('SC').shape[0]  # size(C,1) #N = CFile["Order"].shape[1]
    simVars = neuronalModel.initSim(N)
    if verbose:
        print("Warming Up...", end=" ", flush=True)
    initStimuli(dt, TWarmUp)
    simVars, obsVars = integrate(dt, TWarmUp, simVars, doBookkeeping=False)
    if verbose:
        print("and simulating!!!", flush=True)
    initStimuli(dt, Tmaxneuronal)
    simVars, obsVars = integrate(dt, Tmaxneuronal, simVars, doBookkeeping=True)
    return obsVars


# ======================================================================
# Debug/test code
# To use it, comment the @jit(nopython=True) line at integrationStep
# Otherwise you'll get weird numba errors
# ======================================================================
if __name__ == '__main__':
    import math
    import matplotlib.pyplot as plt

    class dummyNeuronalModel:
        def __init__(self):
            pass
        # we will use the differential equation y'(t) = y(t).
        def dfun(self, simVars, p):
            y = simVars
            return y, y
        def recompileSignatures(self):
            pass
        def numObsVars(self):
            return 1

    neuronalModel = dummyNeuronalModel()

    # The analytic solution is y = e^t.
    def asol(t):
        return math.exp(t)

    clamping = False
    dt = 0.5
    ds = dt
    Tmax = 5.0
    y0 = np.array([[1.0]])

    initStimuli(dt, Tmax)

    simVars, obsVars = integrate(dt, Tmax, y0)

    t = np.arange(0.0, Tmax, dt)
    yasol = np.vectorize(asol)
    plt.plot(t,obsVars.flatten()[:-1],'r-',label="Heun's")
    plt.plot(t,yasol(t),'b-', label='analytical')
    plt.legend()
    plt.show()

# ==========================================================================
# ==========================================================================
# ========================================================================== --EOF
