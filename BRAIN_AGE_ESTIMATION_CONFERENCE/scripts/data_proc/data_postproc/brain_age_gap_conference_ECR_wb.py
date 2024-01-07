import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mtick

"""
 { 'tab:blue': '#1f77b4',
      'tab:orange': '#ff7f0e',
      'tab:green': '#2ca02c',
      'tab:red': '#d62728',
      'tab:purple': '#9467bd',
      'tab:brown': '#8c564b',
      'tab:pink': '#e377c2',
      'tab:gray': '#7f7f7f',
      'tab:olive': '#bcbd22',
      'tab:cyan': '#17becf'}
"""
sites = ['ABIDE', 'RENJI', 'PET_CENTER', 'ADNI', 'OASIS', 'ADHD', 'CoRR', 'CBMFM'] # 1-8
site_dict = {1:'ABIDE', 2: 'RENJI', 3: 'PET_CENTER', 4: 'ADNI', 5: 'OASIS', 6: 'ADHD', 7:'CoRR', 8:'CBMFM'}
HCs_comp = {1: 6, 2:1, 3:2, 4:[4, 3], 5: [4, 5]}
HCs_comp = {1: 6, 2:1, 3:2, 4:[4, 3], 5: [4, 5]}
# Diag = {1:'ADHD', 2: 'ASD', 3:'VCI', 4:'MCI', 5:'AD', 6:'HC'} # subinfo_BDs_dive.csv
Diag = ["CN", "LMCI", "AD"]

def HC_comp(disease=0):
    root_path = "/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/636406_best_ours/FIGURES_CSV"
    test_csv =root_path + '/' + 'age_prediction_best_in_test_exp_0.csv'
    val_csv = root_path + '/' + 'age_prediction_val_exp_0.csv'
    train_csv = root_path + '/' + 'HCs_sMRI_age_prediction_train_exp_0.csv'

    root = "/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/ECR/csv_file"
    csv_file = root + "/ADNI_Five_Class.csv" # path to data

    df = pd.read_csv(csv_file)

    df = df.loc[df['diag'] == 0].copy()
    
    diff = df['BAG'].to_numpy()

    # idx = 4
    # # df1 = pd.read_csv(val_csv)
    # df2 = pd.read_csv(test_csv)
    # # pred1 = df1.loc[df1['site'] == idx].copy() 
    # pred2 = df2.loc[df2['site'] == idx].copy() 
    # # diff1 = pred1['diff'].to_numpy()
    # diff = pred2['diff'].to_numpy()

    # diff= np.concatenate((diff1, diff2))
    print(len(diff))
    BAG = np.sum(diff)/len(diff)
    # print(MAE)
    print("Comp_HCs: {}, MAE {:.2f} ".format(Diag[disease], BAG))
    return diff


def func():
    """
    obtain the brain age gap distribution
    """
    # inpath = '/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/visualization/data/636406_best_ours/TEST'
    inpath = "/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/ECR/csv_file"
    outpath = '/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/ECR/figure'

    font = {'family': 'Times New Roman',
            'color':  'black',
            'size': 16
            }
    leg_font = {'family': 'Times New Roman',
            'size': 16
            }
    plt.rc('font', family='Times New Roman')
    # Set font color to white
    # plt.rcParams['text.color'] = 'black'
    # plt.rcParams['axes.labelcolor'] = 'white'
    # plt.rcParams['xtick.color'] = 'black'
    # plt.rcParams['ytick.color'] = 'black'
    # plt.rcParams['axes.edgecolor'] = 'white'
    # plt.rcParams['axes.facecolor'] = 'white'
    # fig, ax = plt.subplots()
    fig, axs = plt.subplots(nrows=1, ncols=1)
    # fig.set_facecolor('white')

    # file = ["Autism_age_prediction_in_test_exp_0.csv","AD_age_prediction_in_test_exp_0.csv", 
    #         "VCI_age_prediction_in_test_exp_0.csv", "MCI-AD_age_prediction_in_test_exp_0.csv", "ADHD_age_prediction_in_test_exp_0.csv"]
    # Diag = {1:'Autism', 2: 'AD', 3:'VCI', 4:'MCI', 5:'ADHD'} # subinfo_BDs_dive.csv
    file = ["MCI_sMRI_age_prediction_in_test_exp_0.csv", "AD_sMRI_age_prediction_in_test_exp_0.csv"]
    # Diag = { 1:'MCI', 2:'AD', 3:'HC'} # subinfo_BDs_dive.csv
    # colors = ['#e377c2', '#9467bd', 'lightgray']
    # colors_l = ['limegreen', 'darkslategrey' , 'hotpink', 'deeppink',  'indigo'] # 
    colors = [ '#e377c2', '#9467bd', 'lightgray']
    colors_l = [ 'hotpink', 'indigo'] # 

    Diag = ["CN", "EMCI", "MCI", "LMCI", "AD"]
    Diag = ["LMCI", "AD"]

    # for idx, f in enumerate(file):
    root = "/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/ECR/csv_file"
    csv_file = root + "/ADNI_Five_Class.csv" # path to data

    df = pd.read_csv(csv_file)

    df_AD = df.loc[df['diag'] == 3].copy()
    df_LMCI = df.loc[df['diag'] == 4].copy()

    # df= df.loc[(df['site'] == 4)]
    # df['Seq'] = np.arange(len(df))
    # df = df.set_index('Seq')    

    diff_AD = df_AD['BAG'].to_numpy()
    BAG = np.sum(diff_AD)/df_AD.shape[0]
    print("Disease: {}, # {}, BAG {:.2f} ".format("AD", df_AD.shape[0], BAG))

    diff_LMCI = df_LMCI['BAG'].to_numpy()
    BAG = np.sum(diff_LMCI)/df_LMCI.shape[0]
    print("Disease: {}, # {},  BAG {:.2f} ".format("LMCI", df_LMCI.shape[0], BAG))

    alpha1 = 0.8
    alpha2 = 0.5
    color_h = 'dimgrey'
    
    idx = 0
    diff_HC_comp = HC_comp(idx)
    axs.hist(diff_LMCI, bins=100, range=(-15, 15),  color=colors[0], label=Diag[0], zorder=2, alpha=alpha2, density=True)
    axs.hist(diff_AD, bins=100, range=(-15, 15),  color=colors[1], label=Diag[1], zorder=2, alpha=alpha2, density=True)

    axs.hist(diff_HC_comp,  bins=100, range=(-15, 15), color='darkgrey', label='CN', density=True, alpha=alpha1)   
    axs.legend(loc='upper left', prop=leg_font, frameon=False, handleheight=0.6, handlelength=2, labelspacing=0.2, handletextpad=0.2)
    axs.axvline(x=np.mean(diff_LMCI), color=colors_l[0], linewidth=2)
    axs.axvline(x=np.mean(diff_AD), color=colors_l[1], linewidth=2)
    axs.axvline(x=np.mean(diff_HC_comp), color=color_h, linewidth=2) # gainsboro

    axs.set_xlim([-15, 15])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    axs.set_xlabel("Brain Age Gap (years)", fontdict=font, x=0.5, labelpad=1)
    axs.set_ylabel("Probability", fontdict=font, rotation='vertical')

    fig.tight_layout()
    plt.savefig(os.path.join(outpath, "BDs_BAG_ours_train_val_test_part_ECR.pdf"))
    plt.show()
    plt.close()

def ADNI_BAG_Statistic():
    root = "/home/zhaonan/ZHAONAN/BrainAgeEst/MICCAI-WORKSHOP/ECR/csv_file"
    data_csv = root + "/ADNI_Five_Class.csv" # path to data

    df = pd.read_csv(data_csv)
    
    Disease = ["CN", "EMCI", "MCI", "LMCI", "AD"]

    total = 5
    for i in [0, 4]:
        df_one = df.loc[df['diag'] == i].copy()
        BAG = df_one['BAG'].to_numpy()
        mean_BAG = np.sum(BAG)/len(BAG)  
        print("Disease: {} with # subject {}, mean BAG value {:.2f}".format(Disease[i], len(BAG), mean_BAG))

        MAE = df_one['BAG'].to_numpy()
        avg_MAE = np.sum(abs(MAE))/len(MAE)   
        print("Disease: {} with # subject {}, MAE value {:.2f}".format(Disease[i], len(MAE), avg_MAE))

    df_one = df.loc[(df['diag'] > 0) & (df['diag'] < 4)].copy()
    BAG = df_one['BAG'].to_numpy()
    mean_BAG = np.sum(BAG)/len(BAG)  
    print("Disease: {} with # subject {}, mean BAG value {:.2f}".format("MCI", len(BAG), mean_BAG))

    MAE = df_one['BAG'].to_numpy()
    avg_MAE = np.sum(abs(MAE))/len(MAE)   
    print("Disease: {} with # subject {}, MAE value {:.2f}".format("MCI", len(MAE), avg_MAE))

    df_CN = df.loc[df['diag'] == 0].copy()    
    BAG_CN = df_CN['BAG'].to_numpy()

    df_LMCI = df.loc[df['diag'] == 3].copy()
    df_AD = df.loc[df['diag'] == 4].copy()
    BAG_LMCI = df_LMCI['BAG'].to_numpy()
    BAG_AD = df_AD['BAG'].to_numpy()
    
    BAG_LMCI = np.sort(BAG_LMCI)
    BAG_AD = np.sort(BAG_AD)
    print("Disease: {} min BAG value {:.2f}, max BAG value {:.2f}".format("LMCI", np.min(BAG_LMCI), np.max(BAG_LMCI)))
    print("Disease: {} min BAG value {:.2f}, max BAG value {:.2f}".format("AD", np.min(BAG_AD), np.max(BAG_AD)))

    
    from scipy.stats import shapiro,ttest_rel, ranksums, wilcoxon, mannwhitneyu, kruskal
    # normal distribution verification
    print("LMCI:{}".format(shapiro(BAG_LMCI)))
    print("AD:{}".format(shapiro(BAG_AD)))

    statistic, p_value = mannwhitneyu(BAG_AD, BAG_LMCI, alternative="greater")

    print(f'statistic:{statistic:.2f}')
    print(f'P-value:{p_value}')

    # kruskal three groups
    statistic, p_value = kruskal(BAG_AD, BAG_LMCI, BAG_CN)

    print(f'statistic:{statistic:.2f}')
    print(f'P-value:{p_value}')
    # statistic, p_value = stats.ttest_rel(age_gap2, age_gap1, alternative="two-sided")

        # statistic, p_value = wilcoxon(age_gaps[i+1], age_gaps[0], alternative="greater")
        # # statistic, p_value = ranksums(age_gap2, age_gap1)

        # # diff_b = age_gap2- age_gap1
        # # sm.qqplot(diff_b, line='s')
        # # pylab.show()

        # print(shapiro(diff_b))
        # print("Paired t-test results for {}".format(i+1))
        # print(f'statistic:{statistic:.2f}')
        # print(f'P-value:{p_value}')

if __name__ == '__main__':
    func()
