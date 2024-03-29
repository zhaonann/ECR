from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import os
import torch
import random
from monai.transforms import Compose, EnsureChannelFirst, RandFlip, RandRotate, RandZoom
import utils
import pandas as pd
from sklearn.model_selection import train_test_split

vol_max = 2165702 # maximum brain volume among all subjects

def normalize_norm(img):
    # -1 1
    img = img + 1 # 0 - 2
    min_v = np.min(img)
    max_v = np.max(img)
    ratio = 1.0 * (1 - 0) / (max_v - min_v)
    img = 1.0 * (img - min_v) * ratio
    return img

class ExCustomDataset():
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        self.transforms_op = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            RandFlip(prob=0.5, spatial_axis=1),
            RandRotate(prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
            RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        ])
    def __getitem__(self, index):
        sMRI_path = self.df.loc[index, 'sMRI_path']
        ID = self.df.loc[index, 'No']
        filename = self.df.loc[index, 'filename']
        site = self.df.loc[index, 'site']
        sMRI = nib.load(os.path.join(sMRI_path, 'T1_n4_skullstrip_reg_norm.nii.gz'))
        data_sMRI = np.array(sMRI.dataobj) # or sMRI.get_fdata()

        data_sMRI = normalize_norm(data_sMRI)
        data_sMRI = torch.tensor(data_sMRI, dtype=torch.float32)
        # data_sMRI = data_sMRI/data_sMRI.mean()

        age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)

        if self.transforms:
            data_sMRI = self.transforms_op(data_sMRI)[0]
        
        data_sMRI = utils.crop_center(data_sMRI, (160, 192, 160))

        return data_sMRI.unsqueeze(0), age, sex, ID, filename, site
    
    def __len__(self):
        return self.df.shape[0]


class  ExCustomDataset_threedim_3view():
    def __init__(self, df, transforms=False):
        self.df = df
        self.transforms = transforms
        self.transforms_op = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            RandFlip(prob=0.5, spatial_axis=1),
            RandRotate(prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
            RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        ])    
    def __getitem__(self, index):
        sMRI_path = self.df.loc[index, 'sMRI_path']
        ID = self.df.loc[index, 'No']
        filename = self.df.loc[index, 'filename']
        site = self.df.loc[index, 'site']
        sMRI = nib.load(os.path.join(sMRI_path, 'T1_n4_skullstrip_reg_norm.nii.gz'))
        data_sMRI = np.array(sMRI.dataobj)
        
        data_sMRI = normalize_norm(data_sMRI)
        data_sMRI = torch.tensor(data_sMRI, dtype=torch.float32)

        age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)

        if self.transforms:
            data_sMRI = self.transforms_op(data_sMRI)[0]

        data_sMRI = utils.crop_center(data_sMRI, (160, 192, 160))

        start = 44 
        end = 113
        idx = random.randint(start, end) # a <= N <= b
        axial = data_sMRI[:, :, idx:idx+1].permute(2, 0, 1)

        start = 45
        end = 114
        idx = random.randint(start, end) 
        sagittal = data_sMRI[idx:idx+1, :, :]

        start = 55 
        end = 134
        idx = random.randint(start, end) 
        coronal = data_sMRI[:, idx:idx+1, :].permute(1, 0, 2)

        return data_sMRI.unsqueeze(0), axial, sagittal, coronal, age, sex, ID, filename, site
    
    def __len__(self):
        return self.df.shape[0]

class ExCustomDataset_3view():
    def __init__(self, df, transforms=False):
        self.df = df
        self.transforms = transforms
        self.transforms_op = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            RandFlip(prob=0.5, spatial_axis=1),
            RandRotate(prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
            RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        ])    
    def __getitem__(self, index):
        sMRI_path = self.df.loc[index, 'sMRI_path']
        ID = self.df.loc[index, 'No']
        filename = self.df.loc[index, 'filename']
        site = self.df.loc[index, 'site']
        sMRI = nib.load(os.path.join(sMRI_path, 'T1_n4_skullstrip_reg_norm.nii.gz'))
        data_sMRI = np.array(sMRI.dataobj) # or sMRI.get_fdata()
        
        data_sMRI = normalize_norm(data_sMRI)
        data_sMRI = torch.tensor(data_sMRI, dtype=torch.float32)

        age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)

        if self.transforms:
            data_sMRI = self.transforms_op(data_sMRI)[0]

        data_sMRI = utils.crop_center(data_sMRI, (160, 192, 160))

        start = 44 
        end = 113
        idx = random.randint(start, end) # a <= N <= b
        axial = data_sMRI[:, :, idx:idx+1].permute(2, 0, 1)

        start = 45
        end = 114
        idx = random.randint(start, end) 
        sagittal = data_sMRI[idx:idx+1, :, :]

        start = 55 
        end = 134
        idx = random.randint(start, end)
        coronal = data_sMRI[:, idx:idx+1, :].permute(1, 0, 2)

        return axial, sagittal, coronal, age, sex, ID, filename, site


class ExCustomDataset_threedim_3view_GAF():
    def __init__(self, df, transforms=False):
        self.df = df
        self.transforms = transforms
        self.transforms_op = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            RandFlip(prob=0.5, spatial_axis=1),
            RandRotate(prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
            RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        ])    
    def __getitem__(self, index):

        sMRI_path = self.df.loc[index, 'sMRI_path']
        ID = self.df.loc[index, 'No']
        filename = self.df.loc[index, 'filename']
        site = self.df.loc[index, 'site']
        sMRI = nib.load(os.path.join(sMRI_path, 'T1_n4_skullstrip_reg_norm.nii.gz'))
        data_sMRI = np.array(sMRI.dataobj) # or sMRI.get_fdata()
        
        data_sMRI = normalize_norm(data_sMRI)
        data_sMRI = torch.tensor(data_sMRI, dtype=torch.float32)

        age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)

        if self.transforms:
            data_sMRI = self.transforms_op(data_sMRI)[0]

        data_sMRI = utils.crop_center(data_sMRI, (160, 192, 160))

        start = 44 
        end = 113
        idx = random.randint(start, end)
        axial = data_sMRI[:, :, idx:idx+1].permute(2, 0, 1)

        start = 45
        end = 114
        idx = random.randint(start, end) 
        sagittal = data_sMRI[idx:idx+1, :, :]

        start = 55 
        end = 134
        idx = random.randint(start, end)
        coronal = data_sMRI[:, idx:idx+1, :].permute(1, 0, 2)

        vol_brain = self.df.loc[index, 'vol_brain']
        norm_v1 = self.df.loc[index, 'vol_brain':'vol_WM'].to_numpy()/vol_max
        per_v1 = self.df.loc[index, 'vol_CSF':'vol_WM'].to_numpy()/vol_brain
        vols_dk109 = self.df.loc[index, 'dk1':'dk106'].to_numpy()
        total_vols = np.sum(vols_dk109)
        per_v2 = vols_dk109/total_vols
        norm_v2 = vols_dk109/vol_max # normalization
        GAF = np.concatenate([norm_v1, norm_v2, per_v1, per_v2], axis=0).astype(np.float32)
        GAF = torch.from_numpy(GAF) # 4 +  3 + 106*2
        return data_sMRI.unsqueeze(0), axial, sagittal, coronal, GAF, age, sex, ID, filename, site
    
    def __len__(self):
        return self.df.shape[0]

class ExCustomDataset_threedim_3view_GAF_BDs():
    def __init__(self, df, disease=1, transforms=False):
        self.df = df.loc[df['disease'] == disease].copy()  # certain diseases
        L = self.df.shape[0]
        self.df['queue'] = np.arange(L)
        self.df = self.df.set_index('queue')
        
        self.transforms = transforms
        self.transforms_op = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            RandFlip(prob=0.5, spatial_axis=1),
            RandRotate(prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
            RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        ])    
    def __getitem__(self, index):

        sMRI_path = self.df.loc[index, 'sMRI_path']
        ID = self.df.loc[index, 'No']
        filename = self.df.loc[index, 'filename']
        site = self.df.loc[index, 'site']

        sMRI = nib.load(os.path.join(sMRI_path, 'T1_n4_skullstrip_reg_norm.nii.gz'))
        data_sMRI = np.array(sMRI.dataobj) 
        
        data_sMRI = normalize_norm(data_sMRI)
        data_sMRI = torch.tensor(data_sMRI, dtype=torch.float32)

        age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)

        if self.transforms:
            data_sMRI = self.transforms_op(data_sMRI)[0]

        data_sMRI = utils.crop_center(data_sMRI, (160, 192, 160))

        start = 44 
        end = 113
        idx = random.randint(start, end)
        axial = data_sMRI[:, :, idx:idx+1].permute(2, 0, 1)

        start = 45
        end = 114
        idx = random.randint(start, end) 
        sagittal = data_sMRI[idx:idx+1, :, :]

        start = 55 
        end = 134
        idx = random.randint(start, end)
        coronal = data_sMRI[:, idx:idx+1, :].permute(1, 0, 2)
        
        vol_brain = self.df.loc[index, 'vol_brain']
        norm_v1 = self.df.loc[index, 'vol_brain':'vol_WM'].to_numpy()/vol_max
        per_v1 = self.df.loc[index, 'vol_CSF':'vol_WM'].to_numpy()/vol_brain
        vols_dk109 = self.df.loc[index, 'dk1':'dk106'].to_numpy()
        total_vols = np.sum(vols_dk109)
        per_v2 = vols_dk109/total_vols
        norm_v2 = vols_dk109/vol_max # normalization
        GAF = np.concatenate([norm_v1, norm_v2, per_v1, per_v2], axis=0).astype(np.float32)
        GAF = torch.from_numpy(GAF) # 4 + 3 + 106*2

        return data_sMRI.unsqueeze(0), axial, sagittal, coronal, GAF, age, sex, ID, filename, site
    
    def __len__(self):
        return self.df.shape[0]

class ExCustomDataset_BDs_MLP(Dataset):
    def __init__(self, df, transforms=False):
        self.df = df
        self.transforms = transforms
        self.transforms_op = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            RandFlip(prob=0.5, spatial_axis=1),
            RandRotate(prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
            RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        ])    
    def __getitem__(self, index):

        sMRI_path = self.df.loc[index, 'sMRI_path']
        ID = self.df.loc[index, 'New_No']
        filename = self.df.loc[index, 'filename']
        site = self.df.loc[index, 'site']
        diag_gt = self.df.loc[index, 'diag']
        self.label = diag_gt
        sMRI = nib.load(os.path.join(sMRI_path, 'T1_n4_skullstrip_reg_norm.nii.gz'))
        data_sMRI = np.array(sMRI.dataobj) # or sMRI.get_fdata()
        
        data_sMRI = normalize_norm(data_sMRI)
        data_sMRI = torch.tensor(data_sMRI, dtype=torch.float32)

        age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)

        if self.transforms:
            data_sMRI = self.transforms_op(data_sMRI)[0]

        data_sMRI = utils.crop_center(data_sMRI, (160, 192, 160))

        start = 44 
        end = 113
        idx = random.randint(start, end)
        axial = data_sMRI[:, :, idx:idx+1].permute(2, 0, 1)

        start = 45
        end = 114
        idx = random.randint(start, end) 
        sagittal = data_sMRI[idx:idx+1, :, :]

        start = 55 
        end = 134
        idx = random.randint(start, end)
        coronal = data_sMRI[:, idx:idx+1, :].permute(1, 0, 2)

        vol_brain = self.df.loc[index, 'vol_brain']
        norm_v1 = self.df.loc[index, 'vol_brain':'vol_WM'].to_numpy()/vol_max
        per_v1 = self.df.loc[index, 'vol_CSF':'vol_WM'].to_numpy()/vol_brain
        vols_dk109 = self.df.loc[index, 'dk1':'dk106'].to_numpy()
        total_vols = np.sum(vols_dk109)
        per_v2 = vols_dk109/total_vols
        norm_v2 = vols_dk109/vol_max # normalization
        GAF = np.concatenate([norm_v1, norm_v2, per_v1, per_v2], axis=0).astype(np.float32)
        GAF = torch.from_numpy(GAF) # 4 +  3 + 106*2
        return data_sMRI.unsqueeze(0), axial, sagittal, coronal, GAF, age, sex, diag_gt, ID, filename, site
    
    def __len__(self):
        return self.df.shape[0]
    
    def get_labels(self):
        return self.df['diag']
    
class ExCustomDataset_GAF():
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        ID = self.df.loc[index, 'No']
        filename = self.df.loc[index, 'filename']
        site = self.df.loc[index, 'site']

        age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)

        vol_brain = self.df.loc[index, 'vol_brain']
        norm_v1 = self.df.loc[index, 'vol_brain':'vol_WM'].to_numpy()/vol_max
        per_v1 = self.df.loc[index, 'vol_CSF':'vol_WM'].to_numpy()/vol_brain
        vols_dk109 = self.df.loc[index, 'dk1':'dk106'].to_numpy()
        total_vols = np.sum(vols_dk109)
        per_v2 = vols_dk109/total_vols
        norm_v2 = vols_dk109/vol_max # normalization
        GAF = np.concatenate([norm_v1, norm_v2, per_v1, per_v2], axis=0).astype(np.float32)
        GAF = torch.from_numpy(GAF) # 4 +  3 + 106*2
        return  GAF, age, sex, ID, filename, site
    
    def __len__(self):
        return self.df.shape[0]


class ExCustomDataset_threedim_GAF():
    def __init__(self, df, transforms=False):
        self.df = df
        self.transforms = transforms
        self.transforms_op = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            RandFlip( prob=0.5, spatial_axis=1),
            RandRotate( prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
            RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        ])    
    def __getitem__(self, index):
        sMRI_path = self.df.loc[index, 'sMRI_path']
        ID = self.df.loc[index, 'No']
        filename = self.df.loc[index, 'filename']
        site = self.df.loc[index, 'site']
        sMRI = nib.load(os.path.join(sMRI_path, 'T1_n4_skullstrip_reg_norm.nii.gz'))
        data_sMRI = np.array(sMRI.dataobj) # or sMRI.get_fdata()
        
        data_sMRI= normalize_norm(data_sMRI)
        data_sMRI = torch.tensor(data_sMRI, dtype=torch.float32)

        age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)

        if self.transforms:
            data_sMRI = self.transforms_op(data_sMRI)[0]

        data_sMRI = utils.crop_center(data_sMRI, (160, 192, 160))

        vol_brain = self.df.loc[index, 'vol_brain']
        norm_v1 = self.df.loc[index, 'vol_brain':'vol_WM'].to_numpy()/vol_max
        per_v1 = self.df.loc[index, 'vol_CSF':'vol_WM'].to_numpy()/vol_brain
        vols_dk109 = self.df.loc[index, 'dk1':'dk106'].to_numpy()
        total_vols = np.sum(vols_dk109)
        per_v2 = vols_dk109/total_vols
        norm_v2 = vols_dk109/vol_max # normalization
        GAF = np.concatenate([norm_v1, norm_v2, per_v1, per_v2], axis=0).astype(np.float32)
        GAF = torch.from_numpy(GAF) # 4 +  3 + 106*2
        return data_sMRI.unsqueeze(0), GAF, age, sex, ID, filename, site
    
    def __len__(self):
        return self.df.shape[0]


def data_split(dataset=None):
    """
    inner dataset split without external validation, train/val/test/ 8:1:1
    """
    n_out_tests = 0

    in_indices = dataset.df['No'].values.tolist()
    in_L = len(in_indices)
    np.random.shuffle(in_indices)

    n_in_tests = int(np.floor(in_L*0.1))
    n_vals = int(np.floor(in_L*0.1))

    split = n_in_tests + n_vals
    n_trains = in_L - split

    in_test_indices, val_indices, train_indices = in_indices[:n_in_tests], in_indices[n_in_tests: split], in_indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    in_test_sampler = SubsetRandomSampler(in_test_indices)
    out_test_sampler = None
    return train_sampler, val_sampler, in_test_sampler, out_test_sampler, n_trains, n_vals, n_in_tests, n_out_tests

def data_split_diag(dataset=None):
    """
    inner dataset split without external validation, train/val/test/ 8:1:1
    """
    n_out_tests = 0

    in_indices = dataset.df['New_No'].values.tolist()
    in_L = len(in_indices)
    np.random.shuffle(in_indices)

    n_in_tests = int(np.floor(in_L*0.2))
    n_vals = int(np.floor(in_L*0.2))

    split = n_in_tests + n_vals
    n_trains = in_L - split

    val_indices, in_test_indices, train_indices = in_indices[:n_in_tests], in_indices[n_in_tests: split], in_indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    in_test_sampler = SubsetRandomSampler(in_test_indices)
    out_test_sampler = None
    return train_sampler, val_sampler, in_test_sampler, out_test_sampler, n_trains, n_vals, n_in_tests, n_out_tests

def train_val_test_split(data_csv):

    df = pd.read_csv(data_csv)

    in_indices = df['New_No'].values.tolist()

    in_L = len(in_indices)
    np.random.shuffle(in_indices)

    n_in_tests = int(np.floor(in_L*0.2))
    n_vals = int(np.floor(in_L*0.2))

    split = n_in_tests + n_vals
    n_trains = in_L - split

    in_test_indices, val_indices, train_indices = in_indices[:n_in_tests], in_indices[n_in_tests: split], in_indices[split:]
    
    df_train = df.iloc[train_indices].copy()
    df_train['New_No'] = np.arange(df_train.shape[0])
    df_train['INDEX'] = np.arange(df_train.shape[0])
    df_train.set_index("INDEX", inplace=True)

    df_val = df.iloc[val_indices].copy()
    df_val['New_No'] = np.arange(df_val.shape[0])
    df_val['INDEX'] = np.arange(df_val.shape[0])
    df_val.set_index("INDEX", inplace=True)

    df_in_test = df.iloc[in_test_indices].copy()
    df_in_test['New_No'] = np.arange(df_in_test.shape[0])
    df_in_test['INDEX'] = np.arange(df_in_test.shape[0])
    df_in_test.set_index("INDEX", inplace=True)

    return df_train, df_val, df_in_test, None, n_trains, n_vals, n_in_tests, 0

def train_val_test_split_stratify(data_csv):

    df = pd.read_csv(data_csv)

    in_indices = df['New_No'].values.tolist()
    X = in_indices
    y = df['diag'].values.tolist()

    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.4, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, stratify=y_remaining)

    # in_L = len(in_indices)
    # np.random.shuffle(in_indices)

    # n_in_tests = int(np.floor(in_L*0.2))
    # n_vals = int(np.floor(in_L*0.2))

    # split = n_in_tests + n_vals
    # n_trains = in_L - split

    # in_test_indices, val_indices, train_indices = in_indices[:n_in_tests], in_indices[n_in_tests: split], in_indices[split:]

    train_indices = X_train
    val_indices = X_val
    in_test_indices = X_test   

    n_in_tests = len(in_test_indices)
    n_vals = len(val_indices)
    n_trains = len(train_indices)

    df_train = df.iloc[train_indices].copy()
    df_train['New_No'] = np.arange(df_train.shape[0])
    df_train['INDEX'] = np.arange(df_train.shape[0])
    df_train.set_index("INDEX", inplace=True)

    df_val = df.iloc[val_indices].copy()
    df_val['New_No'] = np.arange(df_val.shape[0])
    df_val['INDEX'] = np.arange(df_val.shape[0])
    df_val.set_index("INDEX", inplace=True)

    df_in_test = df.iloc[in_test_indices].copy()
    df_in_test['New_No'] = np.arange(df_in_test.shape[0])
    df_in_test['INDEX'] = np.arange(df_in_test.shape[0])
    df_in_test.set_index("INDEX", inplace=True)

    return df_train, df_val, df_in_test, None, n_trains, n_vals, n_in_tests, 0

def train_test_split_stratify(data_csv):

    df = pd.read_csv(data_csv)

    in_indices = df['New_No'].values.tolist()
    X = in_indices
    y = df['diag'].values.tolist()

    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.4, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, stratify=y_remaining)

    # in_L = len(in_indices)
    # np.random.shuffle(in_indices)

    # n_in_tests = int(np.floor(in_L*0.2))
    # n_vals = int(np.floor(in_L*0.2))

    # split = n_in_tests + n_vals
    # n_trains = in_L - split

    # in_test_indices, val_indices, train_indices = in_indices[:n_in_tests], in_indices[n_in_tests: split], in_indices[split:]

    train_indices = X_train
    val_indices = X_val
    in_test_indices = X_test   

    n_in_tests = len(in_test_indices)
    n_vals = 0
    n_trains = len(train_indices) + len(val_indices)
    
    print(sorted(set(val_indices)), flush=True)
    print(sorted(set(in_test_indices)), flush=True)

    print("train_val and in_test, intersection:", flush=True)
    u_inter = set(val_indices).intersection(set(in_test_indices))
    print(sorted(u_inter), flush=True)
    print("train_val and in_test, union:", flush=True)
    u_set = set(val_indices).union(set(in_test_indices))
    print(sorted(u_set), flush=True)

    df_train = df.iloc[train_indices + val_indices].copy()
    df_train['New_No'] = np.arange(df_train.shape[0])
    df_train['INDEX'] = np.arange(df_train.shape[0])
    df_train.set_index("INDEX", inplace=True)

    # df_val = df.iloc[val_indices].copy()
    # df_val['New_No'] = np.arange(df_val.shape[0])
    # df_val['INDEX'] = np.arange(df_val.shape[0])
    # df_val.set_index("INDEX", inplace=True)

    df_in_test = df.iloc[in_test_indices].copy()
    df_in_test['New_No'] = np.arange(df_in_test.shape[0])
    df_in_test['INDEX'] = np.arange(df_in_test.shape[0])
    df_in_test.set_index("INDEX", inplace=True)

    return df_train, df_in_test, None, None, n_trains, n_in_tests, 0, 0



if __name__ == "__main__":
    pass