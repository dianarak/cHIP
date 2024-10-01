import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import KFold
pd.options.mode.chained_assignment = None

import sys
sys.path.append('/path/to/libnew') # change this to the path where the libnew folder (provided in this repo) is located
from core import calc_sigmas_average
import ml


'''This scripts generates the cHIP fit and learning curve data for all the Ligand1-Ligand2 subsets of DB1.'''


def get_fit():
    ''' This function generates the cHIP fit for all the Ligand1-Ligand2 subsets of the data.
    After running this script, you will be able to reproduce the Hammett plot in Fig. 6 of the manuscript, bu plotting reference energies and cHIP prediction lines against cHIP sigmas.

    Reads:
    --------
        representations.csv: a csv file with the one-hot encoding of the records in the dataset, required for categorical regression if sigmas for records in test set were not fitted during training,
                            which can happen when the training set is small.
        subsets: a folder containing the Ligand1-Ligand2 subsets of the data, each in a separate csv file. The files should be named as sub_Ligand1_Ligand2.csv

    Calls:
    --------
    From libnew, the following functions are called: (note that the path to libnew should be added to the sys.path in the script)
        ml.get_training_params: function that fits the cHIP parameters to the dataset
        ml.eval_df: function that evaluates the cHIP model on the dataset
        core.calc_sigmas_average: function that calculates the average sigma_ij values for the ligands in the dataset

    Writes:
    ---------
        chip_fit_maes.csv: file containing the mean absolute error for each Ligand1-Ligand2 subset
        chip_fit_all.csv: file merging the cHIP predictions for all the models fitted on all Ligand1-Ligand2 subsets
        hip_fit_maes.csv: file containing the mean absolute error for each Ligand1-Ligand2 subset using the HIP model
    '''
    data4sigmas = pd.read_csv('representations.csv')
    data4sigmas = data4sigmas.set_index('Record')

    filenames = []
    maes = []
    stds = []
    maes_lstsq = []
    stds_lstsq = []
    df_merge = pd.DataFrame()


    for file in os.listdir('subsets/'): 
        if file.startswith('sub'):
            filename = file.split('.csv')[0]
            filename = filename.split('_')[1]
            print('Ligand combination: ',filename)
            filenames.append(filename)
            df = pd.read_csv('subsets/' + file)        

            df1 = df.copy()
            df1 = df1.drop(['Group_L1', 'Group_L2', 'L1L2'], axis=1)
            df1['idx'] = df1.index   

                        
            #evaluate the Hammett parameters
            train_params = ml.get_training_params(df1, data4sigmas, onehot=True)
                
            #predict on the test set
            test_pred = ml.eval_df(df1, train_params)
            
            mae = np.nanmean(np.abs(df1['Value'].values - test_pred))
            std = np.nanstd(np.abs(df1['Value'].values - test_pred))
            maes.append(mae)
            stds.append(std)
            print('MAE: ', mae, '   STD: ', std)

            df1['Predval'] = test_pred
            df1['Rho'] = df1['Field'].map(train_params.dicrho)
            df1['Sigma'] = df1['Record'].map(train_params.dicsigmas)
            df1['x0'] = df1['Field'].map(train_params.dicx0)
            sigma, sum_sigma = calc_sigmas_average(df1[['Ligand1', 'Ligand2']], df1['Sigma'], TheilSen=False)
            df1['Sigma_lstsq'] = df1['Ligand1'].map(sigma.set_index('Compound')['Sigmas_calc']) + df1['Ligand2'].map(sigma.set_index('Compound')['Sigmas_calc'])
            df1['Predval_lstsq'] = df1['Sigma_lstsq'] * df1['Rho'] + df1['x0']
            
            df1.to_csv('subsets/pred_fit_{}.csv'.format(filename))
            df_merge = pd.concat([df_merge, df1], axis=0)

            mae_lstsq = np.nanmean(np.abs(df1['Value'].values - df1['Predval_lstsq'].values))
            std_lstsq = np.nanstd(np.abs(df1['Value'].values - df1['Predval_lstsq'].values))
            maes_lstsq.append(mae_lstsq)
            stds_lstsq.append(std_lstsq)
            print('MAE lstsq: ', mae_lstsq, '   STD lstsq: ', std_lstsq)

    #save maes and stds 
    lcdf = pd.DataFrame({'Filename': filenames, 'MAE': maes, 'Std': stds})
    lcdf.to_csv('subsets/hip_fit_maes.csv')
    lcdf_lstsq = pd.DataFrame({'Filename': filenames, 'MAE': maes_lstsq, 'Std': stds_lstsq})
    lcdf_lstsq.to_csv('subsets/chip_fit_maes.csv')

    df_merge.to_csv('subsets/chip_fit_all.csv')        




def get_learning_curve_data():
    ''' This function generates the learning curve data for the cHIP model trained on subsets of the data.
    After running this script, you will be able to reproduce the learning curve plots for the subsets in Fig. 8 of the manuscript, by plotting the mean absolute error against the training set size.
    
    Reads:
    --------
        representations.csv: a csv file with the one-hot encoding of the records in the dataset, required for categorical regression if sigmas for records in test set were not fitted during training,
                            which can happen when the training set is small.
        subsets: a folder containing the Ligand1-Ligand2 subsets of the data, each in a separate csv file. The files should be named as sub_Ligand1_Ligand2.csv
        
    Calls:
    --------
    From libnew, the following functions are called: (note that the path to libnew should be added to the sys.path in the script)
        ml.get_training_params: function that fits the cHIP parameters to the dataset
        ml.eval_df: function that evaluates the cHIP model on the dataset
        
    Writes:
    ---------
        chip_maes_{}.csv: file containing the mean absolute error for each training set size for each Ligand1-Ligand2 subset
        chip_preds_{}.csv: file containing the cHIP predictions for each training set size for each Ligand1-Ligand2 subset
        '''

    #get the representations
    data4sigmas = pd.read_csv('representations.csv')
    data4sigmas = data4sigmas.set_index('Record')

    for file in os.listdir('subsets/'): 
        if file.startswith('sub'):
            filename = file.split('.csv')[0]
            filename = filename.split('_')[1]
            print('Ligand combination: ',filename)
            df = pd.read_csv('subsets/' + file)        

            df1 = df.copy()
            df1 = df1.drop(['Group_L1', 'Group_L2', 'L1L2'], axis=1)

            df1['idx'] = df1.index
            df1['Predval_c'] = np.nan
            df1['Sigma_c'] = np.nan
            df1['Rho'] = np.nan
            df1['Sigma'] = np.nan
            df1['x0'] = np.nan

            listofTSsizes = [250, 500, 1000, 2000, 4000]


            #dividing the dataset into 5 folds, each fold is used once as a test set
            nFolds = 5
            nModels = 4

            test_size = int(len(df1)/nFolds)
            all_idx = np.array(list(range(len(df1))))
            random.seed(667)
            random.shuffle(all_idx)
            train_indices = [] 
            test_indices = []
            
            for i in range(nFolds):
                test_indices.append(all_idx[int(i*test_size):int((i+1)*test_size)])
                train_indices.append(np.concatenate((all_idx[:int(i*test_size)], all_idx[int((i+1)*test_size):])))
            test_indices[-1] = np.concatenate((test_indices[-1], all_idx[int(nFolds*test_size):]))

            
            all_preds = np.zeros(len(df1))
            all_rhos = np.zeros(len(df1))
            all_sigmas = np.zeros(len(df1))
            all_sigmas_lstsq = np.zeros(len(df1))
            all_preds_lstsq = np.zeros(len(df1))
            all_x0 = np.zeros(len(df1))
            maes_allFolds = []
            for i in range(nFolds):
                print('Fold ', i+1)
                train_idx = train_indices[i]
                test_idx = test_indices[i]
                train  = df1.iloc[train_idx]
                test = df1.iloc[test_idx]
                maes_1fold = []
                sublistofTSsizes = []
            
                for TSsize in listofTSsizes:
                    if (TSsize > len(train)):
                        TSsize = len(train)
                    sublistofTSsizes.append(TSsize)
                    maes_models = []


                    for model_num, model in enumerate(range(nModels)):
                        newtraindf = train.sample(n=TSsize, random_state=model_num+67)
                        
                        #evaluate the Hammett parameters
                        train_params = ml.get_training_params(newtraindf, data4sigmas, onehot=True)
                            
                        #predict on the test set
                        test_pred = ml.eval_df(test, train_params)
                        test['Predval'] = test_pred
                        test['Rho'] = test['Field'].map(train_params.dicrho)
                        test['Sigma'] = test['Record'].map(train_params.dicsigmas)
                        test['x0'] = test['Field'].map(train_params.dicx0)

                        sigma, sum_sigma = calc_sigmas_average(test[['Ligand1', 'Ligand2']], test['Sigma'], TheilSen=False)
                        test['Sigma_c'] = test['Ligand1'].map(sigma.set_index('Compound')['Sigmas_calc']) + test['Ligand2'].map(sigma.set_index('Compound')['Sigmas_calc'])
                        test['Predval_c'] = test['Sigma_c'] * test['Rho'] + test['x0']

                        mae = np.nanmean(np.abs(test['Value'].values - test['Predval'].values))
                        maes_models.append(mae)

                        if TSsize == listofTSsizes[-1] or TSsize == len(train):
                            all_preds[test_idx] += (test_pred/nModels)
                            all_rhos[test_idx] += (test['Field'].map(train_params.dicrho) / nModels)
                            all_sigmas[test_idx] += (test['Record'].map(train_params.dicsigmas) / nModels)
                            all_x0[test_idx] += (test['Field'].map(train_params.dicx0) / nModels)
                            all_sigmas_lstsq[test_idx] += (test['Ligand1'].map(sigma.set_index('Compound')['Sigmas_calc']) + test['Ligand2'].map(sigma.set_index('Compound')['Sigmas_calc'])) / nModels
                            all_preds_lstsq[test_idx] += (all_sigmas_lstsq[test_idx] * all_rhos[test_idx] + all_x0[test_idx])
                        
           

                    #mean and standard deviation of the maes of all the models in the current training set size
                    maes_1fold.append(np.mean(maes_models))

                    #if the length of df1 is reached, then break
                    if (TSsize == len(train)):
                        break


                maes_allFolds.append(maes_1fold)
                for j in range(len(sublistofTSsizes)):
                    print(sublistofTSsizes[j], maes_1fold[j])

        
            # prediction performances for all the folds and all the training set sizes
            maes4lc = np.mean(maes_allFolds, axis=0)
            std4lc = np.std(maes_allFolds, axis=0)
            lcdf = pd.DataFrame({'Trainsize': sublistofTSsizes, 'MAE': maes4lc, 'Std': std4lc})
            lcdf.to_csv('subsets/chip_maes_{}.csv'.format(filename))

            df['Predval'] = all_preds
            df['Rho'] = all_rhos
            df['Sigma'] = all_sigmas
            df['x0'] = all_x0
            df['Sigma_c'] = all_sigmas_lstsq
            df['Predval_c'] = all_preds
            df.to_csv('subsets/chip_preds_{}.csv'.format(filename))


if __name__ == '__main__':
    # get_fit()
    get_learning_curve_data()