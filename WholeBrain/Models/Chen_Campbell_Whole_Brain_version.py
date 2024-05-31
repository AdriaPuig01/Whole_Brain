# ==========================================================================
# ==========================================================================
# ==========================================================================
# Chen and Campbell's population model adapted to
# Whole-Brain level by Adrià Puig
#
# Original Chen and Capmbell model from: Chen, L., Campbell, S.A. Exact mean-field models
# for spiking neural networks with adaptation. J Comput Neurosci 50, 445–469 (2022).
# https://doi.org/10.1007/s10827-022-00825-9
#
# Code by Adrià Puig
#
# ==========================================================================
import numpy as np
from numba import jit
from scipy.integrate import odeint

print("Going to use model Chen and Campbell...")

def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    dfun.recompile()

# ==========================================================================
# ==========================================================================
# ==========================================================================
# Model dimensional parameters
# --------------------------------------------------------------------------
beta = -1            # Resonator/Integrator Variable
TW = 200              # time constant of the adaptation eq.
VR = -82.66              # resting membrane potential
C = 250               # Capacitance
k1 = 10              # spike width factor
VT = -42.34          # threshold
Er = 0                # Reversal Potential
Gsyn = 200            # maximal conductance
Iext = 0              # extra current
Sjump = 0.8
Tsyn = 4              # synaptic time constant
Vpeak = 30
Vreset = -55
Wjump_1 = 800
Wjump_2 = 200
k = 0.8

# Scaling relations from dimensional to dimensionless parameters
a_exc = C/(TW*k1*abs(VR))
a_inh = C/(TW*k1*abs(VR))
b = beta/(k1*abs(VR))
alpha = 1 + VT/abs(VR)
gsyn = Gsyn/(k1*abs(VR))
er = 1+Er/abs(VR)
Ie = Iext/(k1*VR*VR)
tsyn = Tsyn*k1*abs(VR)/C
sjump = Sjump*C/(k1*abs(VR))
vpeak = 1 + Vpeak/abs(VR)
vreset = 1 + Vreset/abs(VR)
wjump_exc = Wjump_1/(k1*VR*VR)
wjump_inh = Wjump_2/(k1*VR*VR)

# Parameters Lorentzian bath
mu = 120
hw = 20

# --------------------------------------------------------------------------

# Simulation variables
# @jit(nopython=True)
def initSim(N):
    r_exc = np.zeros(N)
    v_exc = np.zeros(N)
    w_exc = np.zeros(N)
    s_exc = np.zeros(N)

    r_inh = np.zeros(N)
    v_inh = np.zeros(N)
    w_inh = np.zeros(N)
    s_inh = np.zeros(N)

    return np.stack((r_exc, v_exc, w_exc, s_exc, r_inh, v_inh, w_inh, s_inh))

# --------------------------------------------------------------------------

# @jit(nopython=True)
def numObsVars():  # Returns the number of observation vars used
    return 6

# --------------------------------------------------------------------------
# Set the parameters for this model
def setParms(modelParms):
    global G, J, SC, N
    if 'we' in modelParms or 'G' in modelParms:
        G = modelParms['we']
    if 'J' in modelParms:
        J = modelParms['J']
    if 'SC' in modelParms:
        SC = modelParms['SC']
    if 'N' in modelParms:
        N = modelParms['N']

def getParm(parmList):
    if 'we' in parmList or 'G' in parmList:
        return G
    if 'J' in parmList:
        return J
    if 'SC' in parmList:
        return SC
    if 'N' in parmList:
        return N
    return None

# ----------------- Whole-Brain version of Chen and Campbell's model ----------------------
@jit(nopython=True)
def dfun(simVars, I):

    [r_exc, v_exc, w_exc, s_exc, r_inh, v_inh, w_inh, s_inh] = simVars

    I_exc = (k * gsyn * s_exc - J*(1-k) * gsyn * s_inh + gsyn * G * (SC @ s_exc)) * (er - v_exc)
    I_inh = (k * gsyn * s_exc - (1-k) * gsyn * s_inh) * (er - v_inh)

    rm_exc = hw / np.pi + 2 * r_exc * v_exc - r_exc * (gsyn * s_exc + alpha)
    vm_exc = v_exc ** 2 - alpha * v_exc + gsyn * s_exc * (er - v_exc) - np.pi ** 2 * r_exc ** 2 - w_exc + mu + I_exc
    wm_exc = a_exc * (b * v_exc - w_exc) + wjump_exc * r_exc
    sm_exc = -s_exc / tsyn + sjump * r_exc

    rm_inh = hw / np.pi + 2 * r_inh * v_inh - r_inh * (gsyn * s_inh + alpha)
    vm_inh = v_inh ** 2 - alpha * v_inh + gsyn * s_inh * (er - v_inh) - np.pi ** 2 * r_inh ** 2 - w_inh + mu + I_inh
    wm_inh = a_inh * (b * v_inh - w_inh) + wjump_inh * r_inh
    sm_inh = -s_inh / tsyn + sjump * r_inh

    return np.stack((rm_exc, vm_exc, wm_exc, sm_exc, rm_inh, vm_inh, wm_inh, sm_inh)), np.stack((r_exc, v_exc, w_exc, r_inh, v_inh, w_inh))

# ==========================================================================
# ==========================================================================
# ==========================================================================
