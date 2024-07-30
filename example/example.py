import pandas as pd
import sys
sys.path.append('/path/to/libnew')
from core import calc_sigmas_average


df5 = pd.read_csv('mbdf_full_fit1.csv')
sigmas, sum_sigmas = calc_sigmas_average(df5[['Ligand1', 'Ligand2']], df5['Sigma_ij'], df5['Sigma'], TheilSen=False) #sigma_ij s obtained by dividing sigma of symmetrical compounds by 2, sigma is obtained from HIP
sigmas.to_csv('sigmas_lstsq.csv')
sum_sigmas.to_csv('sum_sigmas_lstsq.csv')

