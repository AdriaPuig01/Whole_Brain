# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
# Original Chen and Capmbell model from: Chen, L., Campbell, S.A. Exact mean-field models
# for spiking neural networks with adaptation. J Comput Neurosci 50, 445–469 (2022).
# https://doi.org/10.1007/s10827-022-00825-9
# --------------------------------------------------------------------------
#
#  Computes simulations with Chen and Campbell's model (CC):
#
#  Code by Adrià Puig
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from pathlib import Path

from Chen_Campbell_setup import *

filePath = save_path + 'Chen_Campbell_Jneuro_EMOTION.mat'
if not Path(filePath).is_file():
    import fitting_fgain_CC as fitting
    fitting.fitting_ModelParms(tc_transf_REST, '_EMOTION')

print('Loading {}'.format(filePath))
fNeuro = sio.loadmat(filePath)
Js = fNeuro['J'].flatten()
fitting_PLA = fNeuro['fitting_PLA'].flatten()
FCDfitt_PLA = fNeuro['FCDfitt_PLA'].flatten()

maxFC = Js[np.argmax(fitting_PLA)]
minFCD = Js[np.argmin(FCDfitt_PLA)]
print("\n\n#####################################################################################################")
print(f"# Max FC({maxFC}) = {np.max(fitting_PLA)}             ** Min FCD({minFCD}) = {np.min(FCDfitt_PLA)} **")
print("#####################################################################################################\n\n")

plt.rcParams.update({'font.size': 15})
plotFCDpla, = plt.plot(Js, FCDfitt_PLA)
plotFCDpla.set_label("FCD RELATIONAL")
plotFCpla, = plt.plot(Js, fitting_PLA)
plotFCpla.set_label("FC RELATIONAL")
plt.title("Whole-brain fitting")
plt.ylabel("Fitting")
plt.xlabel("Global Coupling (J)")
plt.legend()
plt.show()
