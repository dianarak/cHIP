'''This script train a $\Delta$-ML model on the dataset, as per Section II.C. of the manuscript.
Before running this script, you need to run the create_delta_data.py script to generate the predictions for each training set size without combining rules. 
After running this script, you will be able to reproduce the $\Delta$-ML learning curve in Fig. 8 of the manuscript, bu plotting the mean absolute error vs Training set size.

Reads:
--------
    mbdf_initial.csv: a csv file with the following columns: Name, idx, Value
    hip_allsizes_preds.csv: file containing the predictions for each training size without combining rules
    hip_allsizes_sigmas.csv: file containing the sigmas for each training size
    hip_allsizes_rhos.csv: file containing the rhos for each training size
    hip_allsizes_x0.csv: file containing the x0 for each training size

Calls:
--------
From libnew, the following functions is called: (note that the path to libnew should be added to the sys.path in the script)
    core.calc_sigmas_average: function that calculates the average sigma_ij values for the ligands in the dataset

From krr.py, the following function is called:
    GridSearchCV: function that performs a grid search for the hyperparameters of the $\Delta$-ML model

From MBDF.py, the following function is called:
    generate_mbdf: function that generates the MBDF representation of the molecules

Writes:
---------
    chip_delta_maes.csv: file containing the mean absolute error for each training set size
    chip_delta_pred.csv: file containing the predictions for the biggest training set size'''


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
pd.options.mode.chained_assignment = None
from copy import deepcopy
import scipy.linalg as linalg
from tqdm import tqdm
import random
from krr import GridSearchCV
from MBDF import generate_mbdf

import qml
from qml.kernels import laplacian_kernel


import sys
sys.path.append('/path/to/libnew') # change this to the path where the libnew folder (provided in this repo) is located
from core import calc_sigmas_average

def train():
    
    df = pd.read_csv('mbdf_initial.csv', usecols=['idx', 'Name']) #to enusre correct indexing
    preds_perSize_df = pd.read_csv('hip_allsizes_preds.csv') #Contains the predictions for each training size without combining rules
    sigmas_perSize_df = pd.read_csv('hip_allsizes_sigmas.csv') 
    rhos_perSize_df = pd.read_csv('hip_allsizes_rhos.csv') 
    x0_perSize_df = pd.read_csv('hip_allsizes_x0.csv') 

    X = []
    
    # load or generate the mbdf representation
    if os.path.exists('DB1_mbdf_rep.npy'):
        X = np.load('DB1_mbdf_rep.npy', allow_pickle=True)
    else: 
        df = df.set_index('idx')
        mols = []
        for i in tqdm(df.index):
            mol = qml.Compound()
            xyz_file = df.loc[i]['Name']
            mol.read_xyz("All2Lig/" + xyz_file + ".xyz")
            mols.append(mol)    

        charges= np.asarray([mol.nuclear_charges for mol in mols])
        coords= np.asarray([mol.coordinates for mol in mols])

        X = generate_mbdf(charges, coords, local=False, progress_bar=True)
        np.save('DB1_mbdf_rep.npy', X) 
    print(X.shape)


    listofTSsizes = [625,1250,2500,5000,10000,20000]

    nModels = 4
    
    test_size = 5000
    all_idx = np.array(list(range(len(X))))
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
    all_preds = np.zeros(len(X))
    maes_allFolds = []
    std_perFold = []
    

    for i in range(nFolds):
        print('Fold ', i+1)
        train_idx = train_indices[i]
        test_idx = test_indices[i]

        maes_1fold = []
        std_1fold = []
        
        # generating learning curve data for each fold
        for TSsize in listofTSsizes:
            print('\nTSsize: ', TSsize)

            # calculate the combining rule results for each size
            initial_df = df.copy()
            initial_df['Ligand1'] = initial_df['Name'].str.split('_').str[1]
            initial_df['Ligand2'] = initial_df['Name'].str.split('_').str[2]
            initial_df['Predval'] = preds_perSize_df['{}'.format(TSsize)] 
            initial_df['Sigma'] = sigmas_perSize_df['{}'.format(TSsize)]
            initial_df['Rho'] = rhos_perSize_df['{}'.format(TSsize)]
            initial_df['x0'] = x0_perSize_df['{}'.format(TSsize)]
            print('initial df mae: ', np.mean(np.abs(initial_df['Value'] - initial_df['Predval'])))

            sigmas, sum_sigmas = calc_sigmas_average(initial_df[['Ligand1', 'Ligand2']], initial_df['Sigma'], TheilSen=False)

            # map predval_new according to name
            initial_df['Sigma_c'] = initial_df['Ligand1'].map(sigmas.set_index('Compound')['Sigmas_calc']) + initial_df['Ligand2'].map(sigmas.set_index('Compound')['Sigmas_calc'])
            initial_df['Predval_c'] = initial_df['Sigma_c'] * initial_df['Rho'] + initial_df['x0']
            
            initial_df['Err_{}'.format(TSsize)] = initial_df['Predval_c'] - initial_df['Value']
            print('mae: ', np.mean(np.abs(initial_df['Err_{}'.format(TSsize)])))
            
            y = []
            for j in tqdm(df.index):
                y.append(initial_df['Err_{}'.format(str(TSsize))].loc[j])
            y = np.array(y)


            maes_models = []

            for n in range(nModels):
                print('Model %d' % (n+1))
                random.seed(n)
                random.shuffle(train_idx)
                sub_index = train_idx[:TSsize] 

                # Dealing with NaN and inf values, which are present in the loaded cHIP predictions when the training set size is too small
                nan_idx = np.argwhere(np.isnan(y[sub_index]))
                nan_idx_test = np.argwhere(np.isnan(y[test_idx]))
                inf_idx = np.argwhere(np.isinf(y[sub_index]))
                inf_idx_test = np.argwhere(np.isinf(y[test_idx]))                

                if len(nan_idx) > 0 or len(inf_idx) > 0:
                    print('nan or inf in y')
                    if len (inf_idx) > 0 or len(inf_idx_test) > 0:
                        to_delete = np.concatenate((inf_idx, nan_idx))
                        to_delete_test = np.concatenate((inf_idx_test, nan_idx_test))
                        sub_index = np.delete(sub_index, to_delete)
                        test_idx = np.delete(test_idx, to_delete_test)
                    else:
                        sub_index = np.delete(sub_index, nan_idx)
                        test_idx = np.delete(test_idx, nan_idx_test)
                if len (inf_idx) > 0 or len(inf_idx_test) > 0:
                    sub_index = np.delete(sub_index, inf_idx)
                    test_idx = np.delete(test_idx, inf_idx_test)           

                params = {
                        'lambda': [1e-5, 1e-10, 1e-15],
                        'length': [0.1*1.2**k for k in range(47,60)] 
                    }
                if TSsize < 5000:
                    params = {
                        'lambda': [1e-5],
                        'length': [15.625, 31.25, 62.5, 125, 187.5, 250] 
                    }

                best_params = GridSearchCV(X[sub_index],y[sub_index],params,kernel='laplacian', norm=2 ,cv=4)
                ll = best_params['lambda']
                sigma = best_params['length']
                print('Best lambda: ', ll)
                print('Best sigma: ', str(round(sigma,3)))

                K      = laplacian_kernel(X[sub_index], X[sub_index], sigma)
                K_test = laplacian_kernel(X[sub_index], X[test_idx],  sigma)

                C = deepcopy(K)
                C[np.diag_indices_from(C)] += ll

                L = np.linalg.cholesky(C)
                alpha = linalg.cho_solve((L,True),y[sub_index])

                y_pred  = np.dot(K_test.T, alpha)
                diff = y_pred - y[test_idx]
                mae  = np.nanmean(np.abs(diff))
                maes_models.append(mae)

                if TSsize == listofTSsizes[-1]:
                    all_preds[test_idx] += (y_pred/nModels)
            
            #mean and standard deviation of the maes of all the models in the current training set size
            maes_1fold.append(np.nanmean(maes_models))
            std_1fold.append(np.nanstd(maes_models)/np.sqrt(nModels)) #std from sampling
            print('MAE: ', np.nanmean(maes_models),', Std: ', np.nanstd(maes_models))


        maes_allFolds.append(maes_1fold)
        std_perFold.append(std_1fold) 

        print('Fold', i+1, ': \n')
        for k in range(len(listofTSsizes)):
            print(listofTSsizes[k], maes_1fold[k])
    
    # need to save the prediction performance for all the folds and all the training set sizes
    maes4lc = np.nanmean(maes_allFolds, axis=0) 
    std_combined = np.sqrt(np.nanmean(std_perFold, axis=0)**2 + np.nanstd(maes_allFolds, axis=0)**2/nFolds) 

    # save the learning curve data with the training set size
    lcdf = pd.DataFrame({'Trainsize': listofTSsizes, 'MAE': maes4lc, 'Std': std_combined})
    lcdf.to_csv('chip_delta_maes.csv')

    res_df = df.copy()
    res_df['Predval_d'] = all_preds
    res_df.to_csv('chip_delta_pred.csv')

    print("--- %s seconds ---" % str(round((time.time() - start_time),3)))


if __name__ == '__main__':
    train()