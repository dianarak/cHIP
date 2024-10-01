import pandas as pd
import numpy as np
import sys
sys.path.append('/path/to/libnew') # change this to the path where the libnew folder (provided in this repo) is located
import ml


'''This script fits HIP parameters to the whole dataset, as per Section II.A. of the manuscript.
After running this script, you will be able to reproduce the partity plot in Fig. S3 of the manuscript, by plotting Predval vs Value.

Reads:
--------
mbdf_initial.csv: a csv file with the following columns: Name, idx, Value
representations.csv: a csv file with a one-hot encoding of the records in the dataset, required for categorical regression if sigmas for records in test set were not fitted during training. 
                    Although irrelevant in this script where the whole dataset is used for fitting, it is still required as an input to the get_training_params function.

Calls:
--------
get_training_params: function that fits the HIP parameters to the dataset
eval_df: function that evaluates the HIP model on the dataset

Writes:
---------
mbdf_fitted_hip.csv: file containing the fitted HIP parameters for the whole dataset
'''

df = pd.read_csv('mbdf_initial.csv')

#split the name column into two columns: Record and Field
df['Field'] = df['Name'].str.split('_').str[0]
df['Record'] = df['Name'].str.split('_').str[1] + '_' + df['Name'].str.split('_').str[2]

#get the representations for categorical regression needs
data4sigmas = pd.read_csv('representations.csv')
data4sigmas = data4sigmas.set_index('Record')

listofTSsizes = [len(df)] #fitting on the whole dataset
all_preds = np.zeros((len(df)))
all_rhos = np.zeros((len(df)))
all_sigmas = np.zeros((len(df)))
all_x0 = np.zeros((len(df)))

for TSsize in listofTSsizes:
    print('N = ', TSsize)
        
    #evaluate the Hammett parameters
    train_params = ml.get_training_params(df, data4sigmas, onehot=True)
        
    #find the fitting error
    train_pred = ml.eval_df(df, train_params)

    if TSsize == listofTSsizes[-1]: 
        all_preds = (train_pred)
        all_rhos = (df['Field'].map(train_params.dicrho))
        all_sigmas = (df['Record'].map(train_params.dicsigmas))
        all_x0 = (df['Field'].map(train_params.dicx0))


df['Predval'] = all_preds
df['Rho'] = all_rhos
df['Sigma'] = all_sigmas
df['x0'] = all_x0
mae = np.mean(np.abs(df['Predval'] - df['Value']))
print('MAE: ', mae)

df.to_csv('fitted_hip.csv')

