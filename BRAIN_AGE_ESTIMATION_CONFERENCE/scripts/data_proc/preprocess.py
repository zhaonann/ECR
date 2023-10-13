import pandas as pd
import os
import numpy as np

def ADNI_expand():
    root = "/public_bme/share/sMRI/csv/ECR"
    data_csv = root + "/HUASHAN_ADNI_OASIS.csv" # path to data
    data_csv_org = root + "/subinfo_8_DK106_new.csv"
    out = "HUASHAN_ADNI_OASIS_EXPAND_ALL.csv"

    df1 = pd.read_csv(data_csv)
    df2 = pd.read_csv(data_csv_org)

    df1.rename(columns={'disease': 'diag'}, inplace=True)

    overlap = (set(df1.columns) & set(df2.columns)).difference({'filename'})
    print(overlap)
    df2 = df2.drop(columns=overlap)

    merged_df = pd.merge(df1, df2, on='filename')

    # print(merged_df.columns)
    # print(len(merged_df))

    merged_df.to_csv(os.path.join(root, out), index=False)  

def ADNI_expand_again():
    root = "/public_bme/share/sMRI/csv/ECR"
    data_csv = root + "/HUASHAN_ADNI_OASIS_EXPAND_ALL.csv" # path to data
    data_csv_org = root + "/subinfo_8_server.csv"
    out = "HUASHAN_ADNI_OASIS_EXPAND_ALL.csv"

    df1 = pd.read_csv(data_csv)
    df2 = pd.read_csv(data_csv_org)

    # df1.rename(columns={'disease': 'diag'}, inplace=True)

    overlap = (set(df1.columns) & set(df2.columns)).difference({'filename'})
    print(overlap)
    df2 = df2.drop(columns=overlap)

    merged_df = pd.merge(df1, df2, on='filename')
    merged_df = pd.merge(df1, df2, on='filename')

    merged_df.to_csv(os.path.join(root, out), index=False)  


def ADNI_CN_LMCI_AD():
    root = "/public_bme/share/sMRI/csv/ECR"
    data_csv = root + "/ADNI_EXPAND_ALL.csv" # path to data
    out = "ADNI_LMCI_AD.csv"

    df = pd.read_csv(data_csv)

    df["diag"].loc[df["diag"]==0] = 5 # AD ->4


    # print(df['diag'].value_counts())
    for i in range(df.shape[0]):
        file = df.loc[i, 'filename']
        if file.startswith("EMCI"):
            df.loc[i, 'diag'] = 5
        elif file.startswith("MCI"):
            df.loc[i, 'diag'] = 5
        elif file.startswith("LMCI"):
            df.loc[i, 'diag'] = 0


    df["diag"].loc[df["diag"]==2] = 1 # AD ->4
    df = df.loc[(df['diag']<=1)].copy()


    # df['diag'] = df['diag'] - 1
    # print(df['diag'].value_counts())    
    df['New_No'] = np.arange(df.shape[0])

    df.to_csv(os.path.join(root, out), index=False)  


def ADNI_MCI():
    root = "/public_bme/share/sMRI/csv/ECR"
    data_csv = root + "/ADNI_EXPAND_ALL.csv" # path to data
    out = "ADNI_Five_Class.csv"

    df = pd.read_csv(data_csv)

    df["diag"].loc[df["diag"]==2] = 4 # AD ->4

    for i in range(df.shape[0]):
        file = df.loc[i, 'filename']
        if file.startswith("EMCI"):
            df.loc[i, 'diag'] = 1    
        elif file.startswith("MCI"):
            df.loc[i, 'diag'] = 2
        elif file.startswith("LMCI"):
            df.loc[i, 'diag'] = 3

    print(df['diag'].value_counts())
    df.to_csv(os.path.join(root, out), index=False)  

def ADNI_BAG_Statistic():
    root = "/public_bme/share/sMRI/csv/ECR"
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

"""
Disease: CN with # subject 138, mean BAG value -0.50
Disease: CN with # subject 138, MAE value 3.16
Disease: EMCI with # subject 344, mean BAG value -0.52
Disease: EMCI with # subject 344, MAE value 4.51
Disease: MCI with # subject 100, mean BAG value -0.21
Disease: MCI with # subject 100, MAE value 4.63
Disease: LMCI with # subject 207, mean BAG value 2.47
Disease: LMCI with # subject 207, MAE value 4.01
Disease: AD with # subject 132, mean BAG value 4.43
Disease: AD with # subject 132, MAE value 7.39


Disease: CN with # subject 138, mean BAG value -0.50
Disease: CN with # subject 138, MAE value 3.16
Disease: AD with # subject 132, mean BAG value 4.43
Disease: AD with # subject 132, MAE value 7.39
Disease: MCI with # subject 651, mean BAG value 0.48
Disease: MCI with # subject 651, MAE value 4.37

"""

if __name__ == '__main__':
    # ADNI_expand()
    # ADNI_expand_again()
    # ADNI_CN_MCI()
    ADNI_BAG_Statistic()
    # ADNI_CN_LMCI_AD()