import os

import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class FGD():
    def __init__(self,dir_root,test_ratio=0.3):
        self.dir_root = dir_root
        self.test_ratio = test_ratio
        # Read 

        self.pd_file = pd.DataFrame(columns={'file'})
        self.pd_file = self.pd_file.\
            append(pd.DataFrame({'file':\
            [ dir_root+'/same/' + x for x in os.listdir(dir_root + '/same')]\
            ,'label':1}),ignore_index=True,sort=True)
            
        self.pd_file = self.pd_file.\
            append(pd.DataFrame({'file':\
            [ dir_root+'/diff/' + x for x in os.listdir(dir_root + '/diff')]\
            ,'label':0}),ignore_index=True,sort=True)

        # Distribute
        train, test = train_test_split(self.pd_file, test_size=test_ratio)
        self.train = train
        self.test = test

        # Build
        self.trainset = FGD_train(train)
        self.testset = FGD_test(test)

    def get(self):
        return self.trainset, self.testset

class FGD_train(torch.utils.data.Dataset):
    def __init__(self,df):
        self.pd_train = df
        self.length = len(self.pd_train)
        print('FacenetGapTrain')
        return
    
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.pd_train.iloc[idx]['file']
        values = torch.Tensor(np.load(path))
        label = int( self.pd_train.iloc[idx]['label'])

        sample = (values,label)

        return sample

class FGD_test(torch.utils.data.Dataset):
    def __init__(self,df):
        self.pd_test = df
        self.length = len(self.pd_test)
        print('FacenetGapTest')
        return
    
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.pd_test.iloc[idx]['file']
        values = torch.Tensor(np.load(path))
        label = int( self.pd_test.iloc[idx]['label'])

        sample = (values,label)

        return sample

