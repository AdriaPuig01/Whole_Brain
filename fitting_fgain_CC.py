# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
# Original Chen and Capmbell model from: Chen, L., Campbell, S.A. Exact mean-field models
# for spiking neural networks with adaptation. J Comput Neurosci 50, 445–469 (2022).
# https://doi.org/10.1007/s10827-022-00825-9
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN
#
#  Code By Adrià Puig
# ==========================================================================
# ==========================================================================

# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
from Chen_Campbell_setup import *
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


# ==========================================================================
# ==========================================================================
# ==========================================================================
# IMPORTANT: This function was created to reproduce Chen and Campbell's code.
# Actually, this only performs the fitting which gives the value of J to use for further computations.
# For the plotting, see the respective file (fig_CC.py)
def fitting_ModelParms(tc_transf, suffix):
    # %%%%%%%%%%%%%%% Set General Model Parameters
    J_fileNames = save_path + "J_Balance_we{}.mat"

    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True)}

    wStart = 0.5
    step = 0.1  # 0.025
    wEnd = 1.5 + step
    Js = np.arange(wStart, wEnd, step)

    # Model Simulations
    # ------------------------------------------
    CN.setParms({'we': 700.2, 'SC': C, 'N': 80})
    modelParms = [{'J': J} for J in Js]

    # Now, optimize all J values: determine optimal J to work with
    print("\n\n###################################################################")
    print("# Compute Js")
    print("###################################################################\n")
    fitting = optim1D.distanceForAll_Parms(tc_transf, Js, modelParms, NumSimSubjects=NumSubjects,
                                           distanceSettings=distanceSettings,
                                           parmLabel='J',
                                           outFilePath=save_path, fileNameSuffix=suffix)

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    print("Optimal:\n", optimal)

    filePath = save_path + 'Chen_Campbell_Jneuro_EMOTION.mat'
    sio.savemat(filePath, #{'JI': JI})
                {'J': Js,
                 'fitting_PLA': fitting['FC'],  # fitting_PLA,
                 'FCDfitt_PLA': fitting['swFCD'],  # FCDfitt_PLA
                 })
    print(f"DONE!!! (file: {filePath})")

if __name__ == '__main__':
    fitting_ModelParms(tc_transf_REST, '_EMOTION')
