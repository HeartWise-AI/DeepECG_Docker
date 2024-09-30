import os
import torch
import pandas as pd
from utils.files_handler import ECGFileHandler
from torch.utils.data import DataLoader, Dataset
    
class ProjectDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.diagnosis = df['diagnosis']
        self.ecg_path = df['ecg_path']
        self.df = df
    def __len__(self):
        return len(self.diagnosis)
    
    def __getitem__(self, idx):
        diagnosis = self.diagnosis.loc[idx]
        file_name = os.path.basename(self.ecg_path.loc[idx])
        ecg_signal = ECGFileHandler.load_ecg_signal(self.ecg_path.loc[idx])
        ecg_signal = ecg_signal.transpose(1, 0)
        return {
            'diagnosis': diagnosis, 
            'ecg_signal': torch.from_numpy(ecg_signal).float(), 
            'file_name': file_name
        }

def create_dataloader(df: pd.DataFrame, batch_size: int = 1, shuffle: bool = False):
    dataset = ProjectDataset(df)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




