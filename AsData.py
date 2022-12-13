from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class AsData(Dataset):        
    def __init__(self,df,label, transform = False):        
        self.data = np.array(df)
        self.lable = np.array(label)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.lable[idx], dtype = torch.float32)
        y = y.unsqueeze_(0)
        if self.transform:
            x = self.transform(x)
        return x,y