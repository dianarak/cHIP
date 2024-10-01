import numpy as np
import pandas as pd
import scipy
from scipy.linalg import cho_solve
import random
import qml
from qml.kernels import laplacian_kernel
from qml import fchl
from qml.representations import generate_fchl_acsf


import sys
sys.path.append('/path/to/libnew') # change this to the path where the libnew folder (provided in this repo) is located
import core

import warnings


def reshape2ml(data):
    ''' This function reshapes the rectangular dataframe (Records x Fields) into a list like that is
        ready to be used by the other ham.ml tools
        
        Parameters
        ----------
        data: pd.DataFrame
            rectangular df (Records x Fields)
            
        Returns
        ----------
        datalisted: pd.DataFrame
            reshaped list-like df, with one roe for each Record+Field
    
    
    '''
    
    #make a copy to be be sure that the original dataframe is not affected in any way
    dataidx = data.copy(deep=True) # we call it dataidx becase it contains a column for the idx counter
    dataidx['idx'] = dataidx.index
    datalisted = dataidx.melt(id_vars='idx')
    datalisted.columns = 'Record Field Value'.split(' ')

    datalisted = datalisted.dropna().reset_index(drop=True) #drop nan lines and reset index
    datalisted['idx'] = datalisted.index

    del dataidx #free up some memory
    
    return(datalisted)



def traintest(datalisted, TSsize, data_per_field=4, Record='Record', Field='Field'):
    ''' This functions selects the entries on the trainig set such that it is always possible to calculate the rhos
        of the system. To do so, it makes sure that at least one overlap cell for each column/row has at least
        5 entries
    
        Parameters
        ----------
        datalisted: Pandas DataFrame
            One row for each Record+Field with the corresponding Value and index
         
        TSsize: int
            size of the training set
            
        data_per_field: int
            define how many datapoints are taken for each field
            ATTENTION: to avoid problems with the creation of the training dataset, make sure that
                        TSsize > #field * (data_per_field * 2)
        
        Record, Field = 'Record', 'Field'   str
			name of Record and Field column
        
        Returns
        -----------
        traindf: pandas dataframe
            df of Record+Field and Values for the TRAINING set
        
        testdf: pandas dataframe
            df of Record+Field and Values for the TEST set
        '''
    #sncopy['idx'] = sncopy.index
    dataidx = datalisted.pivot_table(index=Record, columns = Field, values='idx')
    
    numfields = len(dataidx.columns)
    
    if ( numfields * data_per_field * 2 > TSsize):
        warnings.warn('Not enought training points to sample each field as requested, this WILL give problems. Be sure that TSsize > #field * (data_per_field * 2)', RuntimeWarning)
    
    #generate a matrix where each row and each column has exactly 1 entry
    np.random.seed(61)
    while True:
        eyematr = np.eye(len(dataidx.columns))
        np.random.shuffle(eyematr)
        if ( np.sum(np.diag(eyematr)) == 0 ):
            break
            
    #get the coordinates of the ones in the matrix
    tomove = np.argwhere(eyematr == 1)
    moveidx = []
    
    #for each coordinate (couple of reaction) take all the substituents for which there are data on both
    #write the indexes of those substituents in a list
    for i in tomove:
        tmpdf = dataidx[[dataidx.columns[i[0]], dataidx.columns[i[1]]]].dropna(axis=0,how='any')
        try:
            addendi = tmpdf.sample(data_per_field)
        except ValueError:
            addendi = tmpdf.sample(frac=1)
        moveidx = moveidx + addendi.values.ravel().tolist()

    #sort the indexes and convert them to integers
    moveidx = list(set(moveidx))
    moveidx = [ int(_) for _ in moveidx]
    #print('Initial number of training points: ', len(moveidx))

    #contruct a new df with only the indexes obtained above
    traindf = datalisted.iloc[moveidx]
    
    #buil another df with all the other indexes and shuffle it
    shuffrest = datalisted.drop(index=moveidx).sample(frac=1, random_state=61)
    
    #add to the first df new data chosen randomly, until the TS size is reached
    part2 = shuffrest.head(TSsize-len(datalisted.iloc[moveidx]))
    #print('len part2: ', len(part2), TSsize-len(datalisted.iloc[moveidx]))
    traindf = traindf.append(part2)
    
    #then the rest of the dataframe is moved to test
    testdf = shuffrest.tail(-(TSsize-len(datalisted.iloc[moveidx])))
        
    return(traindf,testdf)


def get_training_params(datalisted, record_repr_df, Record='Record', Field='Field', Value='Value', fixrho=True, TheilSenn=True, onehot=False):
    ''' This funciton calculates the Hammett parameters for a TRAINING set, 
        that is, it evaluates the sigmas for missing entries in the dataset from their representation (e.g. one-hot encoding)
    
    
        Parameters
        ----------
            datalisted: Pandas DataFrame
                One row for each Record+Field with the corresponding Value and index
    
            record_repr_df: Pandas DataFrame
                One row for each Record with the representation on the columns

	    Record, Field, Value = 'Record', 'Field', 'Value' str
		name of the respective columns
            
	    fixrho = True : Bool
		if True, use the fixrho function to improve rho quality
	
	    TheilSenn = True : Bool
		if True, use the TheilSenn estimator, for more robust linear regression
		if False, default to simple linear regression (much less memory and computationally expensive)

        Returns   
        ---------
            ham_train: Class 
                dicrho: dictionary
                    key:= Field, value:= rho value

                dicx0: dictionary
                    key:= Field, value:= E0 value

                dicnewsigmas: dictionary
                    key:= Record, value:= sigma (updated to include the values for the Records that are not in the training)
                
                rhos: 1D np.array
                    of rhos (same found in dictrhos)

                sigmas: 1D np.array
                    of sigmas (same found in dicsigmas)

    '''
    datareshaped = datalisted.pivot_table(index='Record', columns='Field', values='Value') #uncomment if requires pivoting

    train_params = core.get_params(datareshaped, 0, fixrho, TheilSenn)
    pd.DataFrame(train_params.dicsigmas.items(), columns=[Record, 'sigma']).to_csv('sigmas.csv', index=False)

    #evaluate the new sigmas
    if onehot:
        dummy_params = sigma_params(train_params.dicsigmas, record_repr_df) #up to here, we are working with the training set
        newsigm = np.dot(record_repr_df.values, dummy_params) #estimating the sigmas for the records that are not in the training set, the length is the same as the length of the representations df
        dicsigmtest = dict(zip(list(record_repr_df.index), newsigm))
        dicsigmtest.update(train_params.dicsigmas) #if we have a sigma value from Hammett, 
                                                #we use that over the categorical regression
                                                #update: replace the categorical regression sigmas with the Hammett ones, if available
        train_params.dicsigmas = dicsigmtest
    
    
    return train_params

def eval_df(datalisted, ham_params, Record='Record', Field='Field', drop_params=True):
    ''' This function calculates the Hammett prediction of a set of data give the Hammett parameters
    
    Parameters
    ----------
        datalisted: Pandas DataFrame
        One row for each Record+Field with the corresponding Value and index
    
        ham_params: Class
                dicrho: dictionary
                    key:= Field, value:= rho value

                dicx0: dictionary
                    key:= Field, value:= E0 value

                dicnewsigmas: dictionary
                    key:= Record, value:= sigma (updated to include the values for the Records that are not in the training)
                
                rhos: 1D np.array
                    of rhos (same found in dictrhos)

                sigmas: 1D np.array
                    of sigmas (same found in dicsigmas)

        Record, Field = 'Record', 'Field' : str
            name of the Record and Field column, default: Field, Record

	drop_params = True, Bool
	    if True, drop the columns with the Hammett parameters (you might want to keep them for debugging)

    Returns
    ---------
        pred: 1D np.array
            Hammett prediction of the given entries with the passed Hammett parameters
 

    '''
    #make a copy of the dataframe
    datalisted1 = datalisted.copy()
    datalisted1['rho'] = datalisted1[Field].map(ham_params.dicrho)
    datalisted1['sigma'] = datalisted1[Record].map(ham_params.dicsigmas)
    datalisted1['x0'] = datalisted1[Field].map(ham_params.dicx0)
    pred = datalisted1['rho'].values * datalisted1['sigma'].values + datalisted1['x0'].values

    if drop_params:
        datalisted1 = datalisted1.drop('rho sigma x0'.split(' '), axis=1)

    
    return(pred)


def sigma_params(dicsigmas, record_repr_df):
    ''' This function calculates the parameters necessary to evaluate sigma in a catergorical regression manner
        assuming: sigma = np.dot(A,x), where A is the repr_df and x is obtained via least squares
        
        Paramters
        ---------
        dicsigmas: dictionary
            key:= Record, value:= sigma
            
        record_repr_df: Pandas DataFrame
            One row for each Record with the representation on the columns
            
        Returns
        --------
        params: 1D np.array
            x vector used to evaluate the sigmas via np.dot(A,x), where A is the representation of the Records
    
    '''
    dfsigma = pd.DataFrame.from_dict(dicsigmas, orient='index', columns=['sigma'], dtype=float)
    # ('dfsigma', dfsigma, dfsigma.shape, dfsigma.index)print
    # print('record_repr_df', record_repr_df, record_repr_df.shape, record_repr_df.index)
    mergeddf = pd.merge(dfsigma, record_repr_df, how='inner', left_index=True, right_index=True) #sigmas df is shorter than repr_df because we only use part of the training set
    # print('mergeddf', mergeddf)
    
    barr = mergeddf['sigma'].values
    Amatr = mergeddf.drop('sigma', axis=1).values.astype(float)
    # print('barr', barr, barr.shape)
    # print('Amatr', Amatr, Amatr.shape)  
    
    params = scipy.sparse.linalg.lsmr(Amatr, barr)[0]
    
    return params


def sliceKernels(traindf, testdf, Kernel):
    '''This function extracts from the Kernel of the complete dataset the ones that contain only the Training or Test values
        
        Parameters
        ----------
            traindf: pandas dataframe
                df of molecules and Activation Energies for the TRAINING set

            traindf: pandas dataframe
                df of molecules and Activation Energies for the TEST set

            Kernel: 2D np.array
                matrix with the complete Kernel
       
        Returns
        -----------
            trainKernel: 2D np.array
                matrix with the kernel to be used for the training

            testKernel: 2D np.array
                matrix with the kernel to be used for the validation
       
    '''
    trainKernel = Kernel[np.ix_(list(traindf.index),list(traindf.index))]
    testKernel = Kernel[np.ix_(list(testdf.index),list(traindf.index))]
    return(trainKernel, testKernel)



def KRR_pred(Y_train, trainKernel, testKernel):
    '''This function gives the KRR prediction of the X_test values
        
        ParametersSimpleNamespace
        ----------
        Y_train: 1D np.array
            array with the training values (not the representations)
        
        trainKernel: 2D np.array
            matrix with the kernel to be used for the training
            
        testKernel: 2D np.array
            matrix with the kernel to be used for the validation
       
        Returns
        -----------
        pred: 1D np.array
            of the predicted observables 
       
    '''
    
    from qml.math import cho_solve
    
    alpha = cho_solve(trainKernel, Y_train)
    pred = np.dot(testKernel, alpha)
    return(pred)


def tanimoto(valuematrix):
    ''' This function calculates the Tanimoto similarity between all the data representation
        It is used to calculate the total Kernel when the representation is not a metric one,
        so you do not use a Gaussian/Laplacian Kernel but rather a similarity one.
        
        Tanimoto(A,B) = A dot B / ( |A| + |B| - A dot B
        
        Parameters
        ----------
        valuematrix: 2D np.ndarray
            representation of the Records
            
        Returns
        ----------
        num/-denom : 2D np.ndarray
            matrxi of the Tanimoto similarity between each row of the input matrix
        
        
    
    '''
    num = np.matmul(valuematrix, valuematrix.T)
    denom = num - np.linalg.norm(valuematrix,axis=1)**2 - (np.linalg.norm(valuematrix,axis=1)**2).T[:,None]

    return(num/-denom)
