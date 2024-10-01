import numpy as np
import pandas as pd
import scipy.stats


from types import SimpleNamespace



# In[5]:

def get_params(data, ref = 0, fixrho=True, TheilSenn=True):
    '''This function calculates the rhos, sigmas and E0 of the Hammett regression. 
        
        Parameters
        ----------
            data : Pandas DataFrame
            	shape = numSigm x numRho

            ref = 0 : int
            reference column for intial rho assignement

            fixrho = True : Bool
            if True, use the fixrho function to improve rho quality
        
            TheilSenn = True : Bool
            if True, use the TheilSenn estimator, for more robust linear regression
            if False, default to simple linear regression (much less memory and computationally expensive)

         
        Returns      (SimpleNamespace Class) with entries:
        -----------
            dicnewrho: dictionary
                key:= Field, value:= rho value

            dicx0: dictionary
                key:= Field, value:= E0 value

            dicsigmas: dictionary
                key:= Record, value:= sigma

            rhos: 1D np.array
                of rhos (same found in dictrhos)

            sigmas: 1D np.array
                of sigmas (same found in dicsigmas)
         
        
        Other
        ----------
            This function calls:
                - _calc_init_rho
                - _calc_sigmas
    '''
    x0 = data.median(axis=0).values
    dicrho, rhos = _calc_init_rho(data, ref, TheilSenn)
    #print('rhos', rhos)
    dicsigmas, sigmas = _calc_sigmas(data - x0, dicrho) 
    dicx0 = dict(zip(list(data.columns), x0))
    #print('sigmas', sigmas)
    
    
    if (fixrho):
        dicrho, dicx0, rhos, x0 = _fix_param(data, sigmas, TheilSenn=True)
    #E0 = E0 + x0
    #print('fixed rhos', rhos)
    return SimpleNamespace(dicrho=dicrho, dicx0=dicx0, dicsigmas=dicsigmas, rhos=rhos, sigmas=sigmas, x0=x0)


def prediction(data, params):
    ''' This function return a pandas df with the prediction from the Hammett parameters. It needs the original
        df for the labeling

        Parameters
        ----------
        data: pd.DataFrame with the observations

            params: Class
                contains the parameters, obtainable from the function get_params

        Returns
        ----------
            pred_df: pd.Dataframe with the predictions

    '''
    
    #this first part is done to make sure that the arrays of parameters have the same order as the df
    rhos = np.array([ params.dicrho[_] for _ in data.columns ])
    x0 = np.array([ params.dicx0[_] for _ in data.columns ])
    sigmas = np.array([ params.dicsigmas[_] for _ in data.index ])
    
    pred_df = pd.DataFrame(np.outer(sigmas, rhos) + x0)
    pred_df.columns = data.columns
    pred_df.index = data.index
    
    return(pred_df)


def ref_mae_calc(data, TheilSenn=True):
    ''' This function evaluated the Hammett coefficient for each reference Field, and returns the specific Field error
        as well as the error across all the Fields

        Parameters
        ----------
            data : pandas DataFrame
                of Values for each Field and Record

            TheiSenn: Bool, if True the function will use the Theil-Senn regressor for a more robust evaluation, 
                            if False it will default to simple linear regression (default = True)    

        Returns
        ---------
            field_mae : np.ndarray (N_fields x N_fields)
                contains the MAE for each Field (column) using each Filed as reference (row)

            tot_mae : 1D np.array  of len = N_fields
                contains the MAE across the entire dataset for each Field as reference
    
    '''
    
    field_names = list(data.columns)
    field_numbers = len(field_names)
    
    field_mae = np.array([])
    
    tot_mae = []

    for ref,j in enumerate(field_names):  #loop over columns, we use each one as reference
        ham_params = get_params(data, ref, TheilSenn = TheilSenn)
        
        pred_df = prediction(data, ham_params)
        residuals  = data - pred_df
        field_mae = np.concatenate((field_mae, np.mean(np.abs(residuals)).values))
        tot_mae.append(np.nanmean(np.abs(residuals.values)))
       
    field_mae = field_mae.reshape(field_numbers,field_numbers)
    
     
    return(field_mae, np.array(tot_mae))



def _fix_param(data, sigmas, TheilSenn=True):
    '''This functions improves the values of rho after the initial evaluation (from get_params) by evaluation the
    linear regression coefficient between the sigmas evaluated and the datapoints
    
    Parameters
    ----------
        data: pd.DataFrame with observations
        
        sigmas: 1D np.array of the sigma values
        
        TheiSenn: Bool, if True the function will use the Theil-Senn regressor for a more robust evaluation, 
                        if False it will default to simple linear regression (default = True)
        
    Returns
    ----------
            dicr2: dictionary
                key:= Field, value:= rho value

            dice2: dictionary
                key:= Field, value:= E0 value

            rhos: 1D np.array
                of rhos (same found in dictrhos)

            sigmas: 1D np.array
                of E0 (same found in dicsigmas)
    
    '''
    
    rho2 = []
    e2 = []
    
    for i in list(data.columns):
        varx = sigmas
        vary = data[i].values
        mask = ~np.isnan(varx) & ~np.isnan(vary) #the mask is used to remove the NaNs
        if TheilSenn:
            newR = scipy.stats.theilslopes(vary[mask], varx[mask])[0]
        else:
            newR = scipy.stats.linregress(varx[mask], vary[mask])[0]
        newE0 = scipy.stats.linregress(varx[mask], vary[mask])[1]
        rho2.append(newR)
        e2.append(newE0)
        
    dicr2 = dict(zip(list(data.columns), rho2))
    dice2 = dict(zip(list(data.columns), e2))
    
    return(dicr2, dice2, np.array(rho2), np.array(e2))


# In[6]:


def _calc_init_rho(data, ref = 0, TheilSenn=True):
    """ This function generates the INITIAL set of rhos (BEFORE the eventual Self-Consistency)
            WLSQ regression on the system Ax=b, where we fix the fisrt m (same as b) to be 1 to avoid trivial solutions
            For this reason the b vector is not only zeros, but contains the first column of the matrix, which has been
            dropped from the A matrix
    
        Parameters
        ----------
            data: pandas DataFrame
                one row for each sigma, one column for each rho
                
            ref = 0: float
                optional, choice of reference reaction
                
            TheiSenn: Bool, if True the function will use the Theil-Senn regressor for a more robust evaluation, 
                        if False it will default to simple linear regression (default = True)            
        
        Returns    (dicrho, rhos)
        ----------
            dicrho: dictionary
                key:= column label, value:= rho
                
            rhos: 1D np.array
                contains the values of rhos
                
        Other
        ----------
            This function calls:
                - _calc_m
    """
    slopesmatr = _calc_m(data, TheilSenn)
    A = np.delete(slopesmatr, ref, 1)
    
    
    b = -slopesmatr[:,ref]
    # print('data:', data.shape, data)
    # print('b:', b.shape, b)
    # print('A:', A.shape, A)
    
    regr_rhos = np.linalg.lstsq(A , b , rcond=None)[0]
   
    rhos = np.insert(regr_rhos, ref, 1)
    
    dicrho = dict(zip(list(data.columns), rhos))
    
    return(dicrho, rhos)


# In[7]:


def _calc_m(data, TheilSenn=True):
    """ Create the A matrix for the overdetermined system of equations Ax=0
            in this matrix, each entry is weighted by the variance of m
            The output matrix has a column for each reaction XY and a row for each pair of reaction XY_X'Y'
            If XY == X'Y' the row is skipped (132 rows in total)
            This matrix is used to solve the system (Rho_X'Y' * m - Rho_XY = 0)
        
        Parameters
        ----------
            data: pandas DataFrame
                of Values for each Field and Record
            
            TheiSenn: Bool, if True the function will use the Theil-Senn regressor for a more robust evaluation, 
                        if False it will default to simple linear regression (default = True)
        
        Returns    (m_matrix)
        ----------
            m_matrix : 2D np.array
                matrix A for the WLSQ
        
        Other
        ----------
            This function does not call any other function
            
            This function is CALLED by:
                - _calc_init_rho
    """
    
    
    cols = list(data.columns)
    
    numcol = len(data.columns)
    
    m_matrix=np.zeros(numcol)

    if TheilSenn:
        for idx1,field1 in enumerate(cols):
            for idx2,field2 in enumerate(cols):
                if (field1==field2): continue
                regrenda = data[[field1, field2]].dropna(axis=0,how='any') # get all the instances where one molecule has been tested with both catalysts
                # print('regrenda:', regrenda.shape) #regrenda is a dataframe with the two columns of interest, the length of the dataframe is the number of instances where both catalysts have been tested
                # print(regrenda)
                if (len(regrenda)<2):
                    continue
                else:
                    newline=np.zeros(numcol)
                    try:
                        slope = scipy.stats.mstats.theilslopes(regrenda[regrenda.columns[0]].values,regrenda[regrenda.columns[1]], alpha=0.5)[0]
                        # print('y:', regrenda[regrenda.columns[0]].values,'x:', regrenda[regrenda.columns[1]])
                        # print('slope:', slope)
                        #where y are the energies from field2 and x are the energies from field1
                        #field2 = m * field1
                        #so m is kind of the correlation coefficient between the two metals
                    except IndexError:
                        continue
                    newline[idx1]= (-1)
                    newline[idx2]= slope
                    m_matrix = np.vstack((m_matrix,newline))
        m_matrix=np.delete(m_matrix, 0, 0)
        # print('m_matrix:', m_matrix)
    #use regular linear regression instead of Theil-Senn (MUCH faster, use it when there are a lot of data)
    else:
        for idx1,field1 in enumerate(cols):
            for idx2,field2 in enumerate(cols):
                if (field1==field2): continue
                regrenda = data[[field1, field2]].dropna(axis=0,how='any')
                if (len(regrenda)<2):
                    continue
                else:
                    newline=np.zeros(numcol)
                    try:
                        slope = scipy.stats.linregress(regrenda[regrenda.columns[0]].values,regrenda[regrenda.columns[1]])[0]
                    except IndexError:
                        continue
                    newline[idx1]= (-1)
                    newline[idx2]= slope
                    m_matrix = np.vstack((m_matrix,newline))
        m_matrix=np.delete(m_matrix, 0, 0)
    return(m_matrix)


# In[8]:


def _calc_sigmas(data, dicrho):
    """ This function generates the set of sigmas
    
        Parameters
        ----------
            data : pandas DataFrame
                of Values for each Field and Record

            dicrhos :  dict
                key := Field, value := rho
    
        Returns   (dicsigmas, sigmas)
        ----------
            dicsigmas :  dict
                key := Record, value := sigma
        
            sigmas : 1D np.array
                set of the sigmas
                
        Others
        -----------
            This function does NOT CALL anything else
            
            This function is CALLED by:
                - Hammett_data
    """
    
    #heading = data.columns.str[-3:-1].values
    rhos = np.vectorize(dicrho.get)(list(data.columns))
        
    sigmas = (data / rhos).mean(axis=1).values #ignoring the NaNs?
    
    dicsigmas = dict(zip(data.index.values, sigmas))
    
    return(dicsigmas, sigmas)

'''Different combination rules'''
def additive(sigma_i, sigma_j):
    return (sigma_i + sigma_j) / 2 
def geometric(sigma_i, sigma_j):
    return ((sigma_i * sigma_j) ** (1/2)) 
def harmonic(sigma_i, sigma_j):
    return 2 * sigma_i * sigma_j / (sigma_i + sigma_j)
def sixpow(sigma_i, sigma_j):
    return (((sigma_i**6 + sigma_j**6) / 2) ** (1/6))

def calc_sigmas_average(substituents, sum_sigma_exp, TheilSen=False, get_dummies=True, sum_sigma_other=None):
    '''Implementation of the Combining rule-enhanced Hammett-Inspired Product (cHIP)
    This function calculates the individual sigmas for a multisubstituted system. 
    It solves the overdetermined system of equations Ax=b. A is a matrix where A_ij is the number of substituents j in the compound i,
    x is the vector of sigmas to be solved for, and b is the vector of the sum of the sigmas obtained experimentally or from HIP
    
    Parameters
    ----------
        substituents: pd.DataFrame with a compound in each row and the name of the substituents in the columns (representation)
        sum_sigma_other: optional, pd.Series with the sum of the sigmas obtained from another method for finding individual sigmas that 
                            you wish to compare with this one (for instance, the sum of the sigmas obtained from published Hammett 
                            parameters for monosubstituted compounds)
        sum_sigma_exp: pd.Series with the sum of the sigmas obtained experimentally or from HIP, used for fitting
        TheilSen: Bool, if True the function will use the Theil-Senn regressor for a more robust evaluation, 
                        if False it will default to simple linear regression (default = False)
        get_dummies: Bool, if True the function will create dummy variables for the substituents,
                        if False it will default to the representation of the substituents as they are (default = True)


    Returns
    ----------
        sigmas: pd.Series with the individual sigmas for each substituent
        sum_sigma: pd.DataFrame with the sum of the sigmas obtained experimentally and from the Hammett parameters
        '''
    
    if get_dummies:
        df_repr = pd.DataFrame()
        for i, column in enumerate(substituents.columns):
            df_column = pd.get_dummies(substituents[column], prefix=None)
            if i == 0:
                df_repr = df_column
            else:
                df_repr = df_repr.add(df_column, fill_value=0)
        df_repr = df_repr.reindex(sorted(df_repr.columns), axis=1)   

        A = df_repr.to_numpy()

    if isinstance(sum_sigma_exp, pd.Series) or isinstance(sum_sigma_exp, pd.DataFrame):
        b = sum_sigma_exp.to_numpy() 
    else:
        b = sum_sigma_exp

    # Solving the system of equations Ax=b using least squares
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Formatting the results
    d = {'Compound': df_repr.columns, 'Sigmas_calc': x}
    sum_sigma_calc = np.dot(A, x)
    d_sum = {'Sigma_calc': sum_sigma_calc, 'Sigma_exp': b}
    if sum_sigma_other is not None:
        d_sum['Sigma_hammett'] = sum_sigma_other
        #print('MAE lstsq hammett - exp:', np.mean(np.abs(sum_sigma_other - b)), 'std dev:', np.std(sum_sigma_other - b))

    #print('MAE lstsq calc - exp:', np.mean(np.abs(sum_sigma_calc - b)), 'std dev:', np.std(sum_sigma_calc - b))

    sigmas = pd.DataFrame(data=d)
    sum_sigma = pd.DataFrame(data=d_sum)
    mae_lstsq = np.mean(np.abs(sum_sigma_calc - b))

            
    if TheilSen:
        # Solving the system of equations Ax=b using Theil-Sen regression
        from sklearn.linear_model import TheilSenRegressor
        regressor = TheilSenRegressor()
        regressor.fit(A, b)
        x_theil = regressor.coef_
        d_theil = {'Compound': df_repr.columns, 'Sigmas_calc': x_theil}
        sum_sigma_calc_theil = np.dot(A, x_theil)
        d_sum_theil = {'Sigma_calc': sum_sigma_calc_theil, 'Sigma_exp': sum_sigma_exp, 'Sigma_hammett': sum_sigma_other}
        if sum_sigma_other is not None:
            d_sum_theil['Sigma_hammett'] = sum_sigma_other

        sigmas_theil = pd.DataFrame(data=d_theil)
        sum_sigma_calc_theil = pd.DataFrame(data=d_sum_theil)
    
        mae_theil = np.mean(np.abs(np.dot(A, x_theil) - sum_sigma_exp))

        if sum_sigma_other is not None:
            print('MAE theil hammett - exp:', mae_theil, 'std dev:', np.std(sum_sigma_other - sum_sigma_exp))
        #print('MAE theil calc - exp:', np.mean(np.abs(np.dot(A, x_theil) - sum_sigma_exp)), 'std dev:', np.std(np.dot(A, x_theil) - sum_sigma_exp))
    
        if mae_theil < mae_lstsq:
            print('Theil-Sen estimator is more accurate.')
            return sigmas_theil, sum_sigma_calc_theil
        else:
            print('Least squares estimator is more accurate.')
            return sigmas, sum_sigma
    
    return sigmas, sum_sigma