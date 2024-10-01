import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from itertools import combinations, product


''' Using sigma, rho, x0 fit from the whole dataset, predicts Binding energies for new catalysts.
After running this script, you will be able to generate the volcano plot in Fig. 9 of the manuscript, by plotting the predicted binding energies of each reaction step against
the predicted binding energies from the oxidative addition step of the dataset.
The reference data was obtained from the dataset in Ref 64 of the paper, M. Busch, M. D. Wodrich, and C. Corminboeuf, ACS Catalysis 7, 5643 (2017).
The equivalence in the names of the records in the this paper and ours is as follows:
    S1: 71, S3: 2, S5: 68, R1: 9, R3: 4, R4: 81, R5: 82, R6: 83, R8: 84, R11: 41

Reads:
--------
The following files can be regenerated in the same way as described in example_chip.py. 
    fit_pred_A.csv: a csv file with the cHIP fit parameters for the oxidative addition step
    fit_pred_B.csv: a csv file with the cHIP fit parameters for the migratory insertion step
    fit_pred_C.csv: a csv file with the cHIP fit parameters for the reductive elimination step

Calls:
--------
    get_params: function that gets the sigma, rho, x0 parameters from the dataset
    get_predictions: function that predicts the binding energies for the new catalysts
    get_new_catalysts: function that generates the new catalysts by mixing and matching the catalysts in the dataset
    map_names: function that maps the names of the records to the names in the xyz files
    get_performance: function that calculates the performance of the predictions
    
Writes:
---------
    new_catalysts_A.csv: file containing the predictions for the oxidative addition step
    new_catalysts_B.csv: file containing the predictions for the migratory insertion step
    new_catalysts_C.csv: file containing the predictions for the reductive elimination step
'''


def get_params(df):

    df['Sigma_i'] = df['Sigma']/2
    df_sigmas = df[['Record', 'Sigma_i']].drop_duplicates()
    df_rho = df[['Field', 'Rho']].drop_duplicates()
    df_x0 = df[['Field', 'x0']].drop_duplicates()

    dict_sigmas = dict(zip(df_sigmas['Record'], df_sigmas['Sigma_i']))
    dict_rho = dict(zip(df_rho['Field'], df_rho['Rho']))
    dict_x0 = dict(zip(df_x0['Field'], df_x0['x0']))
    return dict_sigmas, dict_rho, dict_x0

def get_predictions(df, dict_sigmas, dict_rho, dict_x0, subscript=''):

    df['Sigma_i'] = df['Record'].map(dict_sigmas)
    df['Rho'] = df['Field'].map(dict_rho)
    df['x0'] = df['Field'].map(dict_x0)
    df['Predval_{}'.format(subscript)] = df['Sigma_i'] * df['Rho'] + df['x0']
    return df


def get_new_catalysts(df):
    '''mix and match catalysts'''
    pairs = list(combinations(df['Record'].unique(), 2))

    pair_df = pd.DataFrame(pairs, columns=['Ligand_1', 'Ligand_2'])
    pair_df = pair_df[pair_df['Ligand_1'] != pair_df['Ligand_2']] #laready contained in original dataset
    pair_df['Record'] = pair_df['Ligand_1'] + '_' + pair_df['Ligand_2']

    metals = df['Field'].unique()

    metal_records_combinations = list(product(metals, pair_df['Record']))

    mixnmatch_df = pd.DataFrame(metal_records_combinations, columns=['Field', 'Record'])
    mixnmatch_df['Ligand_1'] = mixnmatch_df['Record'].str.split('_').str[0]
    mixnmatch_df['Ligand_2'] = mixnmatch_df['Record'].str.split('_').str[1]
    mixnmatch_df['Name'] = mixnmatch_df['Field'] + '_' + mixnmatch_df['Record']

    return mixnmatch_df

def map_names(df, ref_col='Record', new_col='Name'):
    '''df: the dataframe with the new catalysts
    ref_col: the column with the names of the records
    new_col: the column with the new names for the records'''

    rec_dict = {'S1':'71', 'S3':'2', 'S5':'68', 'R1':'9', 'R3':'4', 'R4':'81', 'R5':'82', 'R6':'83', 'R8':'84', 'R11':'41'}
    #map the records to the names in the xyz files
    df[new_col] = df[ref_col].map(rec_dict)


    return df

def get_performance(df_pred,  df_true, rxn_step):
    '''the df true from Meyer's dataset'''
    df_pred = map_names(df_pred, 'Ligand_1', 'DB1_L1') 
    df_pred = map_names(df_pred, 'Ligand_2', 'DB1_L2') #matching the ligand names in DB1 and DB2
    df_pred['DB1_name'] = df_pred['Field'] + '_' + df_pred['DB1_L1'] + '_' + df_pred['DB1_L2']

    df_true = df_true[['Predval', 'Name']]
    print(df_pred[df_pred['DB1_name'] != np.nan])

    df_pred = df_pred.merge(df_true, left_on='DB1_name', right_on='Name')
    df_pred['Error'] = df_pred['Predval'] - df_pred['Predval_{}'.format(rxn_step)]
    print(df_pred)
    print('Mean absolute error: ', df_pred['Error'].abs().mean())
    print('std: ', df_pred['Error'].std())

    return df_pred

def get_new_catalysts_all():
    rxn_steps = ['A', 'B', 'C']


    for step in rxn_steps:
        print('\nStep ', step)
        df_load = pd.read_csv('catalyst_discovery/fit_pred_{}.csv'.format(step))
        df_step = df_load.copy()

        dicsigmas, dicrho, dicx0 = get_params(df_step)

        new_df = get_new_catalysts(df_step)
        new_df['Sigma_i'] = new_df['Ligand_1'].map(dicsigmas)
        new_df['Sigma_j'] = new_df['Ligand_2'].map(dicsigmas)
        new_df['Sigma_ij'] = new_df['Sigma_i'] + new_df['Sigma_j']
        new_df['Rho'] = new_df['Field'].map(dicrho)
        new_df['x0'] = new_df['Field'].map(dicx0)
        new_df['Predval_{}'.format(step)] = new_df['Sigma_ij'] * new_df['Rho'] + new_df['x0']
        
        new_df.to_csv('catalyst_discovery/new_catalysts_{}.csv'.format(step), index=False) 

if __name__ == '__main__':
    get_new_catalysts_all()
    
    '''Optional: Check the performance of the predictions, using the catalysts in common with DB1. Careful, DB1 has relative binding energies, while DB2 has relative finding free energies'''
    # db1 = pd.read_csv('mbdf_initial.csv')
    # db1['Name'] = db1['Field'] + '_' + db1['Record']
  
    # mixnmatch_pred = pd.read_csv('catalyst_discovery/new_catalysts_A.csv').copy()
    # mixnmatch_pred = get_performance(mixnmatch_pred, db1, 'A')
    # mixnmatch_pred.to_csv('catalyst_discovery/mixnmatch_pred_A.csv', index=False) #can only do it for A, because the other steps are not in DB1

    








