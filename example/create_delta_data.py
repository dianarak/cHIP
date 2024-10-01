'''This script fit HIP and cHIP parameters to a growing training set size of the dataset, as per Section II.C. of the manuscript.
The results at each training set size are saved and used to train the $\Delta$-ML model in the example_delta.py script.
After running this script, you will be able to reproduce the $\Delta$-ML learning curve in Fig. 8 of the manuscript, by plotting the mean absolute error vs Training set size.
You will also be able to run the example_delta.py script.

Reads:
--------
    mbdf_initial.csv: a csv file with the following columns: Name, idx, Value
    representations.csv: a csv file with a one-hot encoding of the records in the dataset, required for categorical regression if sigmas for records in test set were not fitted during training,
                        which can happen when the training set is small.

Calls:
--------
From libnew, the following functions are called: (note that the path to libnew should be added to the sys.path in the script)
    ml.get_training_params: function that fits the HIP parameters to the dataset
    ml.eval_df: function that evaluates the HIP model on the dataset
    core.calc_sigmas_average: function that calculates the average sigma_ij values for the ligands in the dataset

Writes:
---------
    chip_maes.csv: file containing the mean absolute error for each training set size
    chip_allsizes_preds.csv: file containing the predictions for each training size, calculated using cHIP
    chip_allsizes_sigmas.csv: file containing the sigmas for each training size, calculated using cHIP
    hip_allsizes_sigmas.csv: file containing the sigmas for each training size, calculated using HIP
    hip_allsizes_preds.csv: file containing the predictions for each training size, calculated using HIP
    hip_allsizes_rhos.csv: file containing the rhos for each training size, calculated using HIP
    hip_allsizes_x0.csv: file containing the x0 for each training size, calculated using HIP

'''

import pandas as pd
import numpy as np 
import time
import random
pd.options.mode.chained_assignment = None


import sys
sys.path.append('/path/to/libnew') # change this to the path where the libnew folder (provided in this repo) is located
from core import calc_sigmas_average
import ml

#get the representations
data4sigmas = pd.read_csv('representations.csv')
data4sigmas = data4sigmas.set_index('Record')

all = pd.read_csv('mbdf_initial.csv')

#Field and Record are the two columns that contain the name of the molecule
all['Field'] = all['Name'].str.split('_').str[0]
all['Record'] = all['Name'].str.split('_').str[1] + '_' + all['Name'].str.split('_').str[2]
all['Ligand1'] = all['Name'].str.split('_').str[1]
all['Ligand2'] = all['Name'].str.split('_').str[2]


listofTSsizes = [625,1250,2500,5000,10000,20000]
nModels = 4


test_size = 5000
all_idx = np.array(all.index.values)
random.seed(667)
random.shuffle(all_idx)
train_indices = [] 
test_indices = []

# dividing the dataset into 5 folds, each fold is used once as a test set
nFolds = 5
for i in range(nFolds):
    test_indices.append(all_idx[int(i*test_size):int((i+1)*test_size)])
    train_indices.append(np.concatenate((all_idx[:int(i*test_size)], all_idx[int((i+1)*test_size):])))
test_indices[-1] = np.concatenate((test_indices[-1], all_idx[int(nFolds*test_size):]))

start_time = time.time()

all_preds = np.zeros((len(all), len(listofTSsizes)))
all_preds_lstsq = np.zeros((len(all), len(listofTSsizes)))
all_sigmas = np.zeros((len(all), len(listofTSsizes)))
all_sigmas_lstsq = np.zeros((len(all), len(listofTSsizes)))
all_rhos = np.zeros((len(all), len(listofTSsizes)))
all_x0 = np.zeros((len(all), len(listofTSsizes)))
all_vals = np.zeros((len(all), len(listofTSsizes)))
maes_allFolds = []
std_perFold = []

for i in range(nFolds):
    print('\nFold ', i+1)
    train_idx = train_indices[i]
    test_idx = test_indices[i]
    train  = all.iloc[train_idx]
    test = all.iloc[test_idx]

    maes_1fold = []
    std_1fold = []

    # learning curve data for each fold

    for sizeIdx, TSsize in enumerate(listofTSsizes):
        print('Trainsize = ', TSsize)
        maes_models = []


        for model_num, model in enumerate(range(nModels)):
            print('Model %d' % (model_num+1))
            newtraindf = train.sample(n=TSsize, random_state=model_num)
            
            #evaluate the Hammett parameters
            train_params = ml.get_training_params(newtraindf, data4sigmas, onehot=True)
                
            #predict on the test set
            test_pred = ml.eval_df(test, train_params)

            all_preds[test_idx, sizeIdx] += (test_pred/nModels)
            all_preds[test_idx, sizeIdx] += (test_pred/nModels)
            all_rhos[test_idx, sizeIdx] += (test['Field'].map(train_params.dicrho) / nModels)
            all_sigmas[test_idx, sizeIdx] += (test['Record'].map(train_params.dicsigmas) / nModels)
            all_x0[test_idx, sizeIdx] += (test['Field'].map(train_params.dicx0) / nModels)
            all_vals[test_idx, sizeIdx] = test['Value']

            sigmas, sum_sigmas = calc_sigmas_average(test[['Ligand1', 'Ligand2']], test['Record'].map(train_params.dicsigmas), TheilSen=False)
            all_sigmas_lstsq[test_idx, sizeIdx] += (test['Ligand1'].map(sigmas.set_index('Compound')['Sigmas_calc']) + test['Ligand2'].map(sigmas.set_index('Compound')['Sigmas_calc'])) / nModels
            all_preds_lstsq[test_idx, sizeIdx] += (all_sigmas_lstsq[test_idx, sizeIdx] * all_rhos[test_idx, sizeIdx] + all_x0[test_idx, sizeIdx])
                    
            mae = np.nanmean(np.abs((test['Ligand1'].map(sigmas.set_index('Compound')['Sigmas_calc']) + test['Ligand2'].map(sigmas.set_index('Compound')['Sigmas_calc'])) 
                         * test['Field'].map(train_params.dicrho) + test['Field'].map(train_params.dicx0) - test['Value']))
            maes_models.append(mae)
           

        #mean and standard deviation of the maes of all the models in the current training set size
        maes_1fold.append(np.mean(maes_models))
        std_1fold.append(np.nanstd(maes_models)/np.sqrt(nModels)) 
            


    maes_allFolds.append(maes_1fold)
    std_perFold.append(std_1fold)
    print('Fold', i+1, ':')
    for j in range(len(listofTSsizes)):
        print(listofTSsizes[j], maes_1fold[j])

  
# prediction performances for all the folds and all the training set sizes
maes4lc = np.mean(maes_allFolds, axis=0)
std4lc = np.sqrt(np.nanmean(std_perFold, axis=0)**2 + np.nanstd(maes_allFolds, axis=0)**2/nFolds)

# learning curve data
lcdf = pd.DataFrame({'Trainsize': listofTSsizes, 'MAE': maes4lc, 'Std': std4lc})
lcdf.to_csv('chip_maes.csv')

# save all preds in a csv file with the corresponding trainingsize
all_preds_df = pd.DataFrame(all_preds)
all_preds_df.columns = listofTSsizes
all_preds_df.to_csv('hip_allsizes_preds.csv')

all_preds_lstsq_df = pd.DataFrame(all_preds_lstsq)
all_preds_lstsq_df.columns = listofTSsizes
all_preds_lstsq_df.to_csv('chip_allsizes_preds.csv')

all_rhos_df = pd.DataFrame(all_rhos)
all_rhos_df.columns = listofTSsizes
all_rhos_df.to_csv('hip_allsizes_rhos.csv')

all_sigmas_df = pd.DataFrame(all_sigmas)
all_sigmas_df.columns = listofTSsizes
all_sigmas_df.to_csv('hip_allsizes_sigmas.csv')

all_sigmas_lstsq_df = pd.DataFrame(all_sigmas_lstsq)
all_sigmas_lstsq_df.columns = listofTSsizes
all_sigmas_lstsq_df.to_csv('chip_allsizes_sigmas.csv')

all_x0_df = pd.DataFrame(all_x0)
all_x0_df.columns = listofTSsizes
all_x0_df.to_csv('hip_allsizes_x0.csv')


print("--- %s seconds ---" % str(round((time.time() - start_time),3)))

