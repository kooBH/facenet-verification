import os
import torch
import numpy as np
import pandas as pd

# Based on : https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class FacenetGapDataset(torch.utils.data.Dataset):
    def __init__(self,dir_root):
        self.dir_root = dir_root

        self.pd_file = pd.DataFrame(columns={'file'})
        self.pd_file = self.pd_file.\
            append(pd.DataFrame({'file':\
            [ dir_root+'/same/' + x for x in os.listdir(dir_root + '/same')]\
            }),ignore_index=True)
            
        self.pd_file = self.pd_file.\
            append(pd.DataFrame({'file':\
            [ dir_root+'/diff/' + x for x in os.listdir(dir_root + '/diff')]\
            }),ignore_index=True)

        self.len = self.pd_file.size

        print('FacenetGapDataset::__init__')
        return
    
    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.pd_file.iloc[idx]['file']
        sample = torch.Tensor(np.load(path))


        return sample
