# Whole_Brain

This GitHub repository contains the code for my final degree project: **Implementation of a Whole-Brain System Based on Chen and Campbell’s Model** by Adrià Puig.

**Original Article for Chen and Campbell's Population Model**:
Chen, L., & Campbell, S. A. (2022). Exact mean-field models for spiking neural networks with adaptation. *Journal of Computational Neuroscience, 50*(3), 445–469. [https://doi.org/10.1007/s10827-022-00825-9](https://doi.org/10.1007/s10827-022-00825-9)

The **WholeBrain** folder is entirely extracted from my tutor's GitHub repository, except for the Chen and Campbell's Whole Brain version file in the **Models** folder.

**My Tutor's Library**:
[https://github.com/dagush/WholeBrain](https://github.com/dagush/WholeBrain)

Chen_Campbell_load_data: Selects 20 random subjects out of all 1003. All file paths are defined here and must be modified before running the project to match your computer.

**Files Made by Me**:

- **fig_CC**: Main file to run, from where the Whole Brain fitting for both values \( we \) and \( J \) is plotted. At line 25, you can select which value to use. If using \( we \), call the `prepro_fgain` file, and if using \( J \), call the `fitting_fgain` file.

- **Prepro_fgain_CC.py**: Calculates observables swFCD and FC for every subject, modifying the value \( we \).

- **Fitting_fgain_CC.py**: Calculates observables swFCD and FC for every subject, modifying the value \( J \).

- **Chen_Campbell_setup**: Configures which model, integrator, and observable will be used. Also selects from which human activity the data is needed.

- **Models Folder:Chen_Campbell_Whole_Brain_version**: Contains the implementation of a Whole Brain version of Chen and Campbell's population model.
- 
