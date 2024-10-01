import pandas as pd
import numpy as np
import sys
sys.path.append('/path/to/libnew') # change this to the path where the libnew folder (provided in this repo) is located
from core import calc_sigmas_average


'''This script calculates the average sigma_ij values for the ligands in the dataset, as per Section II.B. of the manuscript.
After running this script, you will be able to reproduce the parity plot in Fig 5 of the manuscript, by plotting Predval vs Value.
The sigma values are obtained from the HIP fit prior to this step. A separate script for generating HIP predictions is provided. 

Reads:
--------
mbdf_full_fit.csv : file containing the HIP fitting results

Calls:
--------
calc_sigmas_average: function that calculates the average sigma_ij values for the ligands in the dataset

Writes:
---------
sigmas_lstsq.csv: file containing the individual sigmas for each substituent
sum_sigmas_lstsq.csv: file containing the sum of sigmas (HIP and cHIP) for each pair of substituents
mbdf_fitted_chip.csv: file containing the cHIP predictions for the dataset

'''

df = pd.read_csv('fitted_hip.csv')

if 'Ligand1' not in df.columns or 'Ligand2' not in df.columns:
    df[['Ligand1', 'Ligand2']] = df['Record'].str.split('_', expand=True)

sigmas, sum_sigmas = calc_sigmas_average(df[['Ligand1', 'Ligand2']], df['Sigma']) #sigma is obtained from HIP
sigmas.to_csv('sigmas_lstsq.csv')
sum_sigmas.to_csv('sum_sigmas_lstsq.csv')


#map the sigmas to the dataset
df['Sigma_c'] = df['Ligand1'].map(sigmas.set_index('Compound')['Sigmas_calc']) + df['Ligand2'].map(sigmas.set_index('Compound')['Sigmas_calc'])
df['Predval_c'] = df['x0'] + df['Rho'] * df['Sigma_c']
mae = np.mean(np.abs(df['Predval_c'] - df['Value']))
print('MAE: ', mae)

df.to_csv('fitted_chip.csv')