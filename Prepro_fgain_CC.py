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
# IMPORTANT: This function was created to reproduce Chen and Capmbell's code.
# Actually, this only performs the fitting which gives the value of we (we in the original
# code, G in the paper) to use for further computations.
# For the plotting, see the respective file (fig_CC.py)
def prepro_G_Optim():
    # %%%%%%%%%%%%%%% Set General Model Parameters
    J_fileNames = save_path + "J_Balance_we{}.mat"

    distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True)}

    wStart = 700
    step = 0.2
    wEnd = 701.8 + step
    WEs = np.arange(wStart, wEnd, step)

    # Model Simulations
    # ------------------------------------------
    CN.setParms({'J': 1., 'SC': C, 'N': 80})
    modelParms = [{'we': we} for we in WEs]

    # Now, optimize all we (G) values: determine optimal G to work with
    print("\n\n###################################################################")
    print("# Compute G_Optim")
    print("###################################################################\n")
    fitting = optim1D.distanceForAll_Parms(tc_transf_REST, WEs, modelParms, NumSimSubjects=NumSubjects,
                                           distanceSettings=distanceSettings,
                                           parmLabel='we',
                                           outFilePath=save_path, fileNameSuffix='')

    optimal = {sd: distanceSettings[sd][0].findMinMax(fitting[sd]) for sd in distanceSettings}
    print("Optimal:\n", optimal)

    filePath = save_path + 'Chen_Campbell_fneuro.mat'
    sio.savemat(filePath,  # {'JI': JI})
                {'we': WEs,
                 'fitting_PLA': fitting['FC'],  # fitting_PLA,
                 'FCDfitt_PLA': fitting['swFCD'],  # FCDfitt_PLA
                 })
    print(f"DONE!!! (file: {filePath})")

if __name__ == '__main__':
    np.random.seed(0)
    prepro_G_Optim()
