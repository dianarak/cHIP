import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cross_corr(data, figsz = 14, figname = 'cross_corr'):
    ''' This function plots each field of the initial dataser against eachother
        If Hammett is applicable to this dataset, each subplot will be a straight line
    
    Parameters
    ----------
            data : pandas DataFrame
                of Values for each Field and Record        
    '''
    
    num_rhos = len(data.columns)
    fig, axes = plt.subplots(num_rhos, num_rhos, figsize=(figsz, figsz), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)

    for row,xval in enumerate(list(data.columns)):
        #print(row, xval)
        for col,yval in enumerate(list(data.columns)):
            #create an adjustable subplot
        
            axes[row,col].plot(data[xval].values, data[yval].values, 'o', markersize = 3, alpha=0.7)

            # axes[row,col].set_xticks([-80, -40, 0, 40, 80], minor=False)
            # axes[row,col].set_yticks([-80, -40, 0, 40, 80], minor=False)
            # axes[row,col].set_xlim(-100, 90)
            # axes[row,col].set_ylim(-100, 90)

            #set the labels for only the axes on the left and bottom
            if row == num_rhos-1:
                axes[row,col].set_xlabel(yval, size=20)
            if col == 0:
                axes[row,col].set_ylabel(xval, size=20) 

            #plot an identity line for all plots
            #axes[row,col].plot(data[xval].values, data[xval].values, '-', color='black')
    #set a shared x and y label
    fig.text(0.5, 0.04, r'$E_{ref}^{bind}$ (kcal/mol)', ha='center', va='center', size=25)
    fig.text(0.04, 0.5, r'$E_{ref}^{bind}$ (kcal/mol)', ha='center', va='center', rotation='vertical', size=25)
    
    #plt.tight_layout()
    plt.savefig((figname +'.png'), dpi=300)
    #plt.show()


# In[1]:


def ref_dependence(data, field_mae, totmae, xlabl='Fields', bestref = 0,  ChemAcc=1, units='', \
                        show_median=False, median_same_axis=True):
    
    ''' This function plots the MAE as a function of the reference Field across all the Fileds.
        Each horizontal line correspond to a different reference, each red dot if the total MAE for that Field
        
        Paramters
        ---------
            data: pandas DataFrame
                of Values for each Field and Record
    
            field_mae : np.ndarray (N_fields x N_fields)
            contains the MAE for each Field (column) using each Filed as reference (row)

            totmae : 1D np.array  of len = N_fields
                contains the MAE across the entire dataset for each Field as reference 
               
            xlabel : str
                string to show on the botton, i.e. meaning of the Fields (e.g. reactions, solvents, ...)
                
            bestreference : int
                best reference reaction, this one will be highlighted in cyan
                
            ChemAcc : float
                value of the target numerical accuracy
            
            units : str (1 kcal/mol = 0.0434 eV,   1 eV = 23.06 kcal/mol)
                units of the MAE (e.g kcal/mol, eV, ...)
                
            show_median : Bool (default False)
                if True, it will add a bar plot with the median value for each Field, useful for comparisons
                the placement of the bar depends on the median_same_axis value
                
            median_same_axis : Bool (default True)
                if True, the median bars will use the same y axis as the MAE
                if False, the meadian bars will use a secondary axis on the right of the plot
                Ignored if show_median=False
    
    '''
    
    
    
    colnames = list(data.columns)
    colnumbers = len(colnames)
    
    fig , ax1 = plt.subplots()
    
    for idx,field in enumerate(colnames[0:]):
        ax1.plot(np.arange(colnumbers), field_mae[idx,:],  color='gray', alpha=0.5)
        #plot_points = ax1.scatter(plotindex[idx,:], (rxnrmse[idx,:]), s=dotsize[idx,:],  color='darkgray')

    ax1.plot(np.arange(colnumbers), totmae, 'o', color='red', alpha=0.5,  label='Total')
    #plot_thiswork, = ax1.plot(np.arange(colnumbers), hamm_rxnrmse, color='red', label='Total')

    
    ax1.plot(np.arange(colnumbers), field_mae[bestref,:], color='cyan', alpha=0.8, label='Best reference')


    ax1.plot(np.array([-0.5, colnumbers - 0.5 ]), np.array([ChemAcc, ChemAcc]), '--', label='Chemical Accuracy')

    ax1.set_xticks(np.arange(colnumbers))
    ax1.set_xticklabels(colnames,rotation=45, ha='right')
    ax1.set_ylabel("MAE " + units)
    ax1.set_xlim(-0.5,colnumbers - 0.5 )
    ax1.set_xlabel(xlabl) 
    plt.rcParams['axes.axisbelow'] = True
    
    if show_median:
        if median_same_axis:
            ax1.bar(np.arange(colnumbers), np.abs(data.median().values), 0.5, alpha=0.2, label="Abs Median")
        else:
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Abs Median', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.bar(np.arange(colnumbers), np.abs(data.median().values), 0.5, alpha=0.2)
        
    plt.legend()   
        
    
    #ax2.plot(np.arange(colnumbers), mixed_df.median().values, "_")
    
    fig.tight_layout()