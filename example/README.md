This folder contains scripts that demonstrates how to use the cHIP model and reproduce our results and figures.

## Requirements
- qml: installation instructions available at https://www.qmlcode.org/installation.html
- xyz files of structures, required for $\Delta$-machine learning: can be downloaded from the Material Cloud at doi: 10.24435/materialscloud:2018.0014/v1 (B. Meyer, B. Sawatlon, S. Heinen, O. A. von Lilienfeld, and C. Corminboeuf, Chemical science 9, 7069 (2018).)
- update libnew path: libnew is available in this repository, it contained the script fo the HIP and cHIP models. Update the path at the begining of `example_hip.py`, `example_chip.py`, `create_delta_data.py`, `example_delta.py`, `chip_subsets.py`, `ml.py`(in the libnew folder).

## Main scripts 
Run these scripts in the order shown below to ensure that all the required files are generated properly.

**example_hip.py**: fits HIP parameters to the whole DB1, as per Section II.A. of the manuscript.

**example_chip.py**: calculates the average sigma_ij values for the ligands in DB1 and the cHIP predictions for all catalysts, as per Section II.B. of the manuscript.

**create_delta_data.py**: fit HIP and cHIP parameters to a growing training set size of the dataset, as per Section II.C. of the manuscript.

**example_delta.py**: trains a $\Delta$-ML model on the dataset, as per Section II.C. of the manuscript.

**chip_subsets.py**: generates the cHIP fit and learning curve data for all the Ligand1-Ligand2 subsets of DB1.

**catalyst_discovery.py**: generates new catalysts from metals and ligands in DB2 and predicts their performances. 

**plots.py**: generates the figures in the paper

## Supporting scripts
**krr.py**: provided by Danish Khan, https://github.com/dkhan42, used in example_delta.py to optimize the hyperparameters of the kernel ridge regression (KRR) model

**MBDF.py**: copy of the publicly available script at https://github.com/dkhan42/MBDF.git, pertains to Ref. 27 in the paper (D. Khan, S. Heinen, and O. A. von Lilienfeld, The Journal of Chemical Physics 159 (2023)). Used to generate many-body distribution functionals (MBDF) representations for the molecules in example_delta.py.

## Files:

**mbdf_initial**: file containing all DB1 compounds and their relative binding energies from Ref. 20 (10.24435/materialscloud:2018.0014/v1)
    - Name: Metal_Ligand1_Ligand2
    - idx and Value columns as defined below

**fitted_hip.csv**: file containing all DB1 compounds and their properties and parameters with the following columns:
    - Record: Ligand1_Ligand2
    - Field: Metal
    - Value: relative binding energies in kcal/mol. 7054 obtained from external source Ref. 20 (doi: 10.24435/materialscloud:2018.0014/v1) and 18062 generated in this work with machine learning (more details in the manuscript Table 1)
    - idx: indices of the compound
    - Predval: relative binding energy predictions from HIP (doi: 10.24435/materialscloud:2018.0014/v1)
    - Rho: metal constant from HIP (doi: 10.24435/materialscloud:2018.0014/v1)
    - Sigma: ligand pair constant, in kcal/mol, from HIP (doi: 10.24435/materialscloud:2018.0014/v1)
    - x0: energy offset, in kcal/mol, from HIP (doi: 10.24435/materialscloud:2018.0014/v1)
    - Ligand1: first ligand in the compounds
    - Ligand2: second ligand in the compounds

**representations.csv**: file containing a one-hot encoding representation for each ligand1-ligand2 pair in DB1. It is for categorical regression if sigmas for records in test set were not fitted during training, which can happen when the training set is small. Columns:
    - Record as defined above
    - L_0 to L_90: columns for each ligand. A value of 1 is assigned if the ligand is present at least once in the pair and 0 otherwise.


## Folders
**catalyst_discovery**: contains files related to DB2 and the catalyst discovery section (section III.C. in the paper)

**ref_20**: contains results (perdictions and learning curve data) of the training of an MBDF-based KRR model, described in section II.C. of the paper

**subsets**: contains files similar to mbdf_initial.csv but separated by categories of ligand pairs

**test_chip**: contains files related to the testing of the proposed combination rule and its comparison with a simple summation of published Hammett parameters for systems that have experimental reference data, as describted in section II.B. of the paper
