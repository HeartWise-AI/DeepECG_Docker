import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

class ProjectDataset:
    def __init__(self, df: pd.DataFrame):
        self.diagnosis = df['diagnosis']
        self.npy_path = df['npy_path']
    def __len__(self):
        return len(self.diagnosis)
    
    def __getitem__(self, idx):
        ecg_signal = np.load(self.npy_path.iloc[idx]).squeeze(-1)
        ecg_signal = ecg_signal.transpose(1, 0)
        return self.diagnosis.iloc[idx], torch.from_numpy(ecg_signal).float()  

def create_dataloader(df: pd.DataFrame):
    dataset = ProjectDataset(df)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader




