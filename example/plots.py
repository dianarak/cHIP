import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from matplotlib.lines import Line2D
from tqdm import tqdm

sys.path.append('/home/diana/Documents/Chemspacelab/Hammet/crosscoupling_pred_data')



def learning_curve(full_dict, subsets_dir='subsets_', subset_prefix = 'chip_maes_', figurename='learning_curve.png'):
    
    ''' This function generates the learning curve plot in Fig. 8 of the manuscript. It plots the MAE vs the training set size for
        the different categories of Ligand1-Ligand2 pairs of the data.
        
    Arguments:
    ----------
        full_dict: dictionary with the filenames of the learning curve data and the labels to be used in the plot
        subsets_dir: directory where the learnig curve data for the subsets are stored
        subset_prefix: prefix of the learning curve data for the subsets
        figurename: name of the figure to be saved
    
    '''


    fig, ax = plt.subplots(1, 1, figsize=(5, 7))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([500,1000,4000,20000], ['0.5k','1k','4k','20k'])
    ax.set_yticks([0,1,2,4,10,20,40], [0,1,2,4,10,20,40], fontsize=16)
    ax.set_ylim(0.8,40)

    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    markers = ['s', 's', 's', 's']
    print(full_dict.keys())
    print(full_dict.values())
    perf_labels = []
    perf_colors = []
    perf_fill = []


    #also add the subsets results on the performance plot
    markers_sub = ['^', '>', 'v', 'p', 'h', '8', 'x', 'D', 'P', 'H',  ]
    labels_sub = ['NHC-Py', 'NHC-Other', 'Other-Other', 'P-Py', 'P-NHC', 
              'P-Other', 'Py-Py', 'P-P', 'NHC-NHC', 'Py-Other'] 
    
    
    subcounter = 0
    '''plot learning curves for subsets'''
    for file in os.listdir(subsets_dir):
        if file.startswith(subset_prefix):
            df = pd.read_csv(subsets_dir + file)
            filename = file.split('.csv')[0]
            filename = filename.split('_')[2]
            print(filename)
            N = df['Trainsize'].values
            maes_for_plot = df['MAE'].values
            s_for_plot = df['Std'].values

            ax.errorbar(N, maes_for_plot, yerr=s_for_plot, marker=markers_sub[subcounter], label=labels_sub[subcounter], markersize=8, linestyle='dotted', markeredgecolor='k', markeredgewidth=1, capsize=4)
            perf_labels.append(labels_sub[subcounter])
            perf_colors.append(ax.lines[-1].get_color())
            perf_fill.append(ax.lines[-1].get_markerfacecolor())
            subcounter += 1
   
    '''plot learning curves for cHIP, delta-ML, and the direct learning of the 7k DFT energies by KRR and MBDF'''
    for i,file in enumerate(full_dict.keys()):
        df2 = pd.read_csv(file)
        print(df2.columns)
        print(df2['MAE'])
        ax.errorbar(df2['Trainsize'], df2['MAE'], yerr=df2['Std'], linestyle=linestyles[i], color='k', markersize=6, marker=markers[i], capsize=4, label=full_dict[file], markerfacecolor='none')
        perf_labels.append(full_dict[file])
        perf_colors.append(ax.lines[-1].get_color())
        perf_fill.append('none')
        i += 1
    ax.set_xlabel(r'$N$', fontsize=20)
    ax.set_ylabel(r'MAE (kcal/mol)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    perf_markers = markers_sub[:len(labels_sub)] + markers
    perf_lines = ['dotted']*len(labels_sub) + linestyles
    perf_handles = [Line2D([0], [0], marker=perf_markers[i], color=perf_colors[i], label=perf_labels[i], linestyle=perf_lines[i], linewidth=1,
                            markerfacecolor=perf_fill[i], markersize=8, markeredgecolor='k', markeredgewidth=1) for i in range(len(perf_labels))]
    
    
    plt.legend(frameon=True, fontsize=14, ncol=2, loc='upper left', columnspacing=0.4, handletextpad=0.1, labelspacing=0.4, bbox_to_anchor=(0.15,1.0), handles=perf_handles)
    plt.minorticks_off()
    plt.tight_layout()
    plt.savefig(figurename, dpi=300)




def hammettplot(subsets_dir='subsets/', prefix='pred_fit_'):
    ''' Plots hammet plot of all subsets and 3 selected metals, Fig. 6 in the manuscript.
    
    Arguments:
    ----------
        subsets_dir: directory where the subsets data is stored
        prefix: prefix of the subsets data files
        '''
    from matplotlib.lines import Line2D
    L1L2_dict = {
        'PyPy': 'Py-Py',
        'NHCPy': 'NHC-Py',
        'OtherOther': 'Other-Other',
        'NHCNHC': 'NHC-NHC',
        'NHCOther': 'NHC-Other',
        'PyOther': 'Py-Other',
        'POther': 'P-Other',
        'PNHC': 'P-NHC',
        'PPy': 'P-Py',
        'PP': 'P-P',
    }

    counter = 0 #changes with each L1L2 combination, changes colors and markers
    fig, ax = plt.subplots(figsize=(6,7))
    colors = ['k', 'tab:red', 'tab:purple', 'tab:brown', 'c', 'm', 'darkgrey', 'goldenrod', 'slateblue', 'k']
    markers = ['o', 'v', '^', 's', 'p', 'P', 'X', 'D', 'd', 'h']
    df_energies = pd.DataFrame()
    L1L2_sorted = []

    #plot a line between the points of the same metal
    metals_plot = ['Ni','Au', 'Pd']
    linestyles = ['dashdot', 'dashed', '-']
    subcounter = 0 #changes with each metal, changes linestyles
    df_au = pd.DataFrame()
    df_ni = pd.DataFrame()
    df_pd = pd.DataFrame()

    '''creating 3 subsets of the data containing all instances of the 3 metals across all L1L2 combinations.'''
    for file in os.listdir(subsets_dir):
        if (file.startswith(prefix)): 

            filename = file.split('.csv')[0]
            filename = filename.split('_')[2]

            df_energy = pd.read_csv(subsets_dir + prefix + filename + '.csv') 

            #sample records to plot if the field is Ag, Pd or Ni
            df_energy_0 = df_energy[df_energy['Field'].isin(['Au', 'Pd', 'Ni'])]
            for metal in metals_plot:
                df_metal_0 = df_energy_0[df_energy_0['Field'] == metal]
                if metal == 'Au':
                    df_au = pd.concat([df_au, df_metal_0])
                elif metal == 'Ni':
                    df_ni = pd.concat([df_ni, df_metal_0])
                elif metal == 'Pd':
                    df_pd = pd.concat([df_pd, df_metal_0])

    #plot the regression lines with the continuous error bars
    df_metals = [df_ni, df_au, df_pd]
    for df in df_metals:
        metal = df['Field'].iloc[0]
        df = df.sort_values(by=['Sigma_lstsq'])
        rho_avg = df['Rho'].mean()
        x0_avg = df['x0'].mean()
        x = df['Sigma_lstsq']
        y = rho_avg*x + x0_avg
        y_err = x.std() * np.sqrt(1/len(x) +
                          (x - x.mean())**2 / np.sum((x - x.mean())**2))
        subcounter += 1
        counter += 1

            
    #creating other subsets for the selected scatter points
    for file in os.listdir(subsets_dir):
        if (file.startswith(prefix)): 

            filename = file.split('.csv')[0]
            filename = filename.split('_')[2]
            

            df_energy = pd.read_csv(subsets_dir + prefix + filename + '.csv') 

            #sample records to plot if the field is Ag, Pd or Ni
            df_energy_s = df_energy[df_energy['Field'].isin(['Au', 'Pd', 'Ni'])]
            df_energy_s = df_energy_s.sort_values(by=['Sigma_lstsq','Record'])
            #include in the sample the 2 records with the highest and lowest Sigma_lstsq values
            rec_unique = df_energy_s['Record'].unique()
            df_energy_head = df_energy_s[df_energy_s['Record'].isin(rec_unique[:3])]
            df_energy_tail = df_energy_s[df_energy_s['Record'].isin(rec_unique[-3:])]
            df_energy_middle = df_energy_s[df_energy_s['Record'].isin(rec_unique[len(rec_unique)//2-1:len(rec_unique)//2+2])]
            df_energy_s = pd.concat([df_energy_head, df_energy_middle, df_energy_tail])
            print(filename)
            # print(df_energy_s)
            df_energy_s['L1L2'] = L1L2_dict[filename]
            #append the subset to the dataframe
            df_energies = pd.concat([df_energies, df_energy_s])
            counter += 1


    subcounter = 0
    for metal in metals_plot:
        df_energy_metal = df_energies[df_energies['Field'] == metal]
        print(metal)
        rho_avg = df_energy_metal['Rho'].mean()
        x0_avg = df_energy_metal['x0'].mean()
        
        #plot a line with a slope of rho
        x = df_energy_metal['Sigma_lstsq']
        y = rho_avg * x + x0_avg

        counter = 0
        for L1L2 in df_energy_metal['L1L2'].unique():
            df_energy_L1L2 = df_energy_metal[df_energy_metal['L1L2'] == L1L2]
            [ax.scatter(df_energy_L1L2['Sigma_lstsq'], df_energy_L1L2['Value'], marker=markers[counter], color=colors[subcounter], s=30, edgecolors='k', linewidths=0.7)]
            #print(markers[counter], colors[subcounter], L1L2)
            if subcounter == 0:
                L1L2_sorted.append(L1L2)
            counter += 1

        
        #plot a regression line with shaded uncertainty
        sns.regplot(x=df_energy_metal['Sigma_lstsq'], y=df_energy_metal['Predval_lstsq'], ax=ax, label=metal, line_kws={'color':colors[subcounter], 'linestyle':linestyles[subcounter]},
                   scatter=False, ci=100, robust=True)
        
        subcounter += 1
    
    L1L2_handles = [Line2D([0], [0], marker=markers[i], color='w', label=L1L2_sorted[i], 
                            markerfacecolor='w', markersize=8, markeredgecolor='k', markeredgewidth=1) for i in range(len(L1L2_sorted))]
    metal_handles = [Line2D([0], [0], color=colors[i], label=metals_plot[i], linestyle=linestyles[i]) for i in range(len(metals_plot))]


    ax.set_ylabel(r'$\Delta E^{\mathrm{r}}$ (kcal/mol)', fontsize=25)
    ax.set_xlabel(r'$\bar{\sigma}_{ij}$ (kcal/mol)', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(loc='lower center', bbox_to_anchor = (0.45,1), ncol=3, fontsize=17, frameon=False, handles=L1L2_handles + metal_handles, columnspacing=0.5, handletextpad=0.3)
    plt.tight_layout()    

    plt.savefig('chip_hammettplot.png', dpi=300)

def parityplot(ref, pred, figname='parityplot.png'):
    ''' Plots the parity plot of the experimental vs predicted values of the cHIP dataset, Fig. 5 in the manuscript.
    
    Arguments:
    ----------
        ref: array of the reference data
        pred: array of the predicted data
        figname: str, name of the figure to be saved
    '''

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(ref, pred, edgecolors='k', linewidths=0.7, s=30)
    ax.plot([ref.min(), ref.max()], [ref.min(), ref.max()], 'k--', lw=2)
    ax.set_xlabel(r'$\Delta E^{\mathrm{r}}$ (kcal/mol)', fontsize=25)
    ax.set_ylabel(r'$\Delta E^{\mathrm{p}}$ (kcal/mol)', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.savefig(figname, dpi=300)


def sigma_parityplot(ref, pred, figname='sigma_parityplot.png'):
    ''' Plots the parity plot of sigmas obtained from cHIP vs sigmas obtained from HIP, Fig. 7 in the manuscript.
    
    Arguments:
    ----------
        ref: array of the reference data
        pred: array of the predicted data
        figname: str, name of the figure to be saved'''

    fig, axes = plt.subplots(1,1, figsize=(6,6))
    ax0 = axes #scatter plot of sigma_ij vs sigma
    left, bottom, width, height = [0.717, 0.697, 0.25, 0.25]
    ax1= fig.add_axes([left, bottom, width, height]) #distribution of the error in sigma_ij

    markers = ['o']
    colors = ['tab:brown']
    marker_i = 0
    linestyles = ['-']    

    '''scatter plot of sigma cHIP vs sigma HIP'''
    ax0.scatter(ref, pred, marker=markers[marker_i], s=10, color=colors[marker_i])

    '''error distribution plot'''
    error = pred - ref
    sns.kdeplot(error, ax=ax1, color=colors[marker_i], linestyle=linestyles[marker_i], linewidth=2)
    
    ax0.set_ylabel(r'$\bar{\sigma}_{ij}$ (kcal/mol)', fontsize=24)
    ax0.set_xlabel(r'$\sigma_{ij}$ (kcal/mol)', fontsize=24)
    ax1.set_xlabel(r'Error (kcal/mol)', fontsize=15)
    ax1.set_ylabel(r'Density', fontsize=15)
    ax0.tick_params(axis='both', which='major', labelsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax0.set_xlim(-20, 65)
    ax0.set_ylim(-20, 65)
    ax0.plot([-20, 62], [-20,62], 'k--')
    ax0.set_aspect('equal')
    ax1.set_xlim(-40,40)

    plt.tight_layout()
    plt.savefig(figname)


def test_chip(ref, sum_sigma_1, sum_sigma_2, sum_sigma_exp, figname='sigma_scatter_lit.png'):
    '''plots the scatter plots of sigma_ij from two combining rule vs sigma from literature, Fig. 3 in the manuscript.
    
    Arguments:
    ----------
    ref: array of reference numbers, should be the same length as sum_sigma_1, sum_sigma_2, sum_sigma_exp
    sum_sigma_1: array of sum of sigma values from combining rule 1 (here, sum of published Hammett parameters, Eq. 7 in the manuscript)
    sum_sigma_2: array of sum of sigma values from combining rule 2 (here, sum of sigmas obtained from the combining rule in cHIP, Eq. 8 in the manuscript)
    sum_sigma_exp: array of sum of sigma values from literature (experimental sigma accounting for the combined effects of several substituents)
    figname: str, name of the figure to save
    
    
    '''

    fig, axes = plt.subplots(1,2, figsize=(7,4), sharey=True)
    ax0 = axes[0]
    ax1 = axes[1]
    ax0.set_aspect('equal')
    ax1.set_aspect('equal')
    markers = ['o', 's', 'D', 'v', 'P', 'X', 'p', 'h', 'd', '8']
    colors = ['tab:red', 'tab:purple', 'tab:brown', 'c', 'm', 'darkgrey', 'goldenrod', 'slateblue', 'k']

    plt.subplots_adjust(wspace=0.1)

    df = pd.DataFrame({'Ref': ref, 'Sum_sigma_1': sum_sigma_1, 'Sum_sigma_2': sum_sigma_2, 'Sum_sigma_exp': sum_sigma_exp})

    for i in range(len(ref.unique())):
        ref_no = ref.unique()[i]
        df_sub = df[df['Ref'] == ref_no]
        ax0.scatter(df_sub['Sum_sigma_exp'], df_sub['Sum_sigma_1'], label=ref_no, marker=markers[i], s=36, color=colors[i], edgecolors='k', linewidths=0.2)
        ax1.scatter(df_sub['Sum_sigma_exp'], df_sub['Sum_sigma_2'], label=ref_no, marker=markers[i], s=36, color=colors[i], edgecolors='k', linewidths=0.2)

    ax0.plot([min(df['Sum_sigma_exp']), max(df['Sum_sigma_exp'])], [min(df['Sum_sigma_exp']), max(df['Sum_sigma_exp'])], 'k--')
    ax1.plot([min(df['Sum_sigma_exp']), max(df['Sum_sigma_exp'])], [min(df['Sum_sigma_exp']), max(df['Sum_sigma_exp'])], 'k--')

    fig.text(0.02, 0.5, r'$\bar{\sigma}$ (kcal/mol)', va='center', rotation='vertical', fontsize=22)
    fig.text(0.5, 0.02, r'$\sigma$ (kcal/mol)', ha='center', fontsize=22)
    ax1.set_xticks([0, 1, 2])
    ax0.set_xticks([0, 1, 2])
    ax0.tick_params(axis='both', which='major', labelsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    
    
    ax0.text(0.06, 0.93, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes, fontsize=20)
    ax1.text(0.06, 0.93, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=20)

    ax1.legend(frameon=False, fontsize=16, loc='center left', bbox_to_anchor=(-0.12, 0.75), handletextpad=0.1, labelspacing=0.1)
    
    plt.savefig(figname, dpi=300)

def volcanoplot():

    ''' This script plots the two volcano plots in Fig. 9 of the paper. The first one ploets the result of a cHIP fit on the data from Busch et al.'s paper (Ref. 64 in the paper)]
        The second one plots the results of the cHIP fit on the new catalysts data. The data is stored in the csv files full_fit_pred_A.csv, full_fit_pred_B.csv, full_fit_pred_C.csv

    Reads:
    --------
    The following files are to be generated by catalyst_discovery.py.
        new_catalysts_A.csv: a csv file with the results of the cHIP fit on the new catalysts data for the oxidation step. 
        new_catalysts_B.csv: a csv file with the results of the cHIP fit on the new catalysts data for the transmetallation step
        new_catalysts_C.csv: a csv file with the results of the cHIP fit on the new catalysts data for the reduction step
    The following files are available in the repository:
        fit_pred_A.csv: a csv file with the cHIP fit parameters for the oxidative addition step
        fit_pred_B.csv: a csv file with the cHIP fit parameters for the migratory insertion step
        fit_pred_C.csv: a csv file with the cHIP fit parameters for the reductive elimination step

    '''

    Rxns = {'G(RxnA)': 'A', 'G(RxnB)': 'B', 'G(RxnC)': 'C'}


    fig, ax = plt.subplots(2, 1, figsize=(9,11), sharex=True)
    colors = ['m', 'c', 'tab:brown',  'tab:purple', 'tab:red','k']
    markers = ['s', 'v', 'o', 'D', 'X', 'P', 'H', 'd', 'p', 'h']
    linestyles = ['-', '-.', 'dashed']

    '''plot for the new catalysts. Fig. 9 b in the paper'''

    oxi_df = pd.read_csv('new_catalysts_A.csv') 
    oxi_df['Name'] = oxi_df['Field'] + '_' + oxi_df['Record']
    tra_df = pd.read_csv('new_catalysts_B.csv', usecols=['Field', 'Record', 'Predval_B'])
    tra_df['Name'] = tra_df['Field'] + '_' + tra_df['Record']
    red_df = pd.read_csv('new_catalysts_C.csv', usecols=['Field', 'Record', 'Predval_C'])
    red_df['Name'] = red_df['Field'] + '_' + red_df['Record']

    all_pds = oxi_df.copy()
    all_pds['Predval_B'] = tra_df['Name'].map(tra_df.set_index('Name')['Predval_B'])
    all_pds['Predval_C'] = red_df['Name'].map(red_df.set_index('Name')['Predval_C'])
    all_pds['E_pds'] = np.minimum(-all_pds['Predval_A'], np.minimum(-all_pds['Predval_B'], -all_pds['Predval_C']))
    print(all_pds[all_pds.duplicated(subset=['Name'], keep=False)])


    # scatter plot with different color for each field
    # one legend per field and one legend per reaction step
    ligands = np.unique(all_pds['Record'])
    metals = np.unique(all_pds['Field'])
    metals = sorted(metals, reverse=True)

    for i, field in enumerate(metals):
        print('field: ',field)
        for j, record in enumerate(ligands):
            df = all_pds[(all_pds['Field'] == field) & (all_pds['Record'] == record)]
            ax[1].scatter(df['Predval_A'], df['E_pds'], c=colors[i], marker=markers[i], s=70, linewidths=0.5, edgecolors='k', alpha=0.6)


    '''plot for the catalyts in Busch et al.'s paper. Fig. 9 a in the paper'''

    oxi_og = pd.read_csv('fit_pred_A.csv')
    oxi_og['Name'] = oxi_og['Field'] + '_' + oxi_og['Record'] + '_' + oxi_og['Record']
    oxi_og.rename(columns={'Predval': 'Predval_A'}, inplace=True)
    tra_og = pd.read_csv('fit_pred_B.csv', usecols=['Field', 'Record', 'Predval'])
    tra_og.rename(columns={'Predval': 'Predval_B'}, inplace=True)
    tra_og['Name'] = tra_og['Field'] + '_' + tra_og['Record'] + '_' + tra_og['Record']
    red_og = pd.read_csv('fit_pred_C.csv', usecols=['Field', 'Record', 'Predval'])
    red_og.rename(columns={'Predval': 'Predval_C'}, inplace=True)
    red_og['Name'] = red_og['Field'] + '_' + red_og['Record'] + '_' + red_og['Record']

    all_og = oxi_og.copy()
    all_og['Predval_B'] = tra_og['Name'].map(tra_og.set_index('Name')['Predval_B'])
    all_og['Predval_C'] = red_og['Name'].map(red_og.set_index('Name')['Predval_C'])
    all_og['E_og'] = np.minimum(-all_og['Predval_A'], np.minimum(-all_og['Predval_B'], -all_og['Predval_C']))
    print(all_og[all_og.duplicated(subset=['Name'], keep=False)])
    for i, field in enumerate(metals):
        print('field: ',field)
        for j, record in tqdm(enumerate(np.unique(all_og['Record']))):
            df = all_og[(all_og['Field'] == field) & (all_og['Record'] == record)]
            ax[0].scatter(df['Predval_A'], df['E_og'], c=colors[i], marker=markers[i], s=70, linewidths=0.5, edgecolors='k', alpha=0.6)


    '''plot the regression lines for each reaction step'''
    rxn_names = ['Oxi', 'Tra', 'Red']
    for i,reaction in enumerate(Rxns):
        line0 = sns.regplot(x=all_og['Predval_A'], y=-all_og['Predval_{}'.format(Rxns[reaction])],
                ax=ax[0], color='k', line_kws={'linewidth': 2.2, 'linestyle': linestyles[i]}, scatter=False, ci=None)

    # Create custom legend handles and labels
    metal_handles = [Line2D([0], [0], marker=markers[i], color='w', label=metals[i], markerfacecolor=colors[i], markersize=12, markeredgecolor='k', markeredgewidth=0.5) for i in range(len(metals))]
    rxn_handles = [Line2D([0], [0], color='k', label=rxn_names[i], linewidth=2.2, linestyle=linestyles[i]) for i in range(len(rxn_names))]

    #plot a vertical line at -32.1 and -23.0
    ax[0].axvline(x=-34.0, color='k', linestyle='dotted', linewidth=2)
    ax[0].axvline(x=-17, color='k', linestyle='dotted', linewidth=2)
    ax[1].axvline(x=-34, color='k', linestyle='dotted', linewidth=2)
    ax[1].axvline(x=-17, color='k', linestyle='dotted', linewidth=2)
    fig.text(0.34, 0.89, r'-34', fontsize=28, rotation=55)
    fig.text(0.59, 0.89, r'-17', fontsize=28, rotation=55)

    ax[0].set_ylim(-1,27)
    ax[0].set_xlim(-50, 1)
    ax[1].set_ylim(-4,25)
    ax[1].set_xlim(-50, 1)

    ax[1].set_xlabel(r'$\Delta E\mathrm{_{oxi}}$ (kcal/mol)', fontsize=30)
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[1].set_ylabel(None)
    fig.text(0.02, 0.5, r'-$\Delta E$ (kcal/mol)', va='center', rotation='vertical', fontsize=30)
    ax[0]
    ax[0].tick_params(axis='both', which='major', labelsize=28)
    ax[1].tick_params(axis='both', which='major', labelsize=28)

    legend2 = fig.legend(loc='upper right', ncol=3, fontsize=25, frameon=True, handles=metal_handles+rxn_handles, 
                        columnspacing=0.9, handlelength=1.15, scatterpoints=2, bbox_to_anchor=(0.916,0.888), labelspacing=0.1, handletextpad=0.3)

    ax[0].add_artist(legend2)
    plt.subplots_adjust(wspace=0, hspace=0.09)

    plt.savefig('catalyst_discovery/volcano_plot.png', dpi=300)

if __name__== '__main__':


    #Fig 3
    literature_df = pd.read_csv('test_chip/lit_sigmas_sum.csv')
    test_chip(literature_df['Ref no'], literature_df['Sigma_hammett'], literature_df['Sigma_calc'], literature_df['Sigma_exp'])

    #Fig 4 reproduces the work of Ref. 20 (B. Meyer, B. Sawatlon, S. Heinen, O. A. von Lilienfeld, and C. Corminboeuf, Chemical science 9, 7069 (2018)), with the addition of the MBDF-based model. 
    #It is simply used to justify the use of MBDF to expand the dataset.

    #Fig 5
    fitted_chip_df = pd.read_csv('fitted_chip.csv') #generated by example_chip.py
    parityplot(fitted_chip_df['Value'], fitted_chip_df['Predval_c'])
    #Fig 7
    sigma_parityplot(fitted_chip_df['Sigma'], fitted_chip_df['Sigma_c'])
        
    #Fig 6
    hammettplot()

    #Fig 8
    learning_curve({'chip_maes.csv':r'Full', #generated by chip_lc.py
                    'chip_delta_maes.csv':r'$\Delta\mathrm{-ML}$', #generated by example_delta.py
                    'ref_20/mbdf_maes_7k.csv':r'$\mathrm{ML}$'} #reproducing Ref. 20 from the paper with the new MBDF representation, available in the repository
                    )
    
    #Fig 9
    volcanoplot()

    