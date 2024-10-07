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
        # Check if DataFrame is empty
    if df.empty:
        raise ValueError("The DataFrame is empty. Please provide a valid data file.")
    
    # Check if 'diagnosis' column exists
    if 'diagnosis' not in df.columns:
        raise ValueError("'diagnosis' column is missing in the DataFrame.")
        
    # Check if 'diagnosis' column is empty and remove such rows
    initial_count = len(df)
    df = df[df['diagnosis'].notna()]
    removed_count = initial_count - len(df)
    print(f"Removed {removed_count} rows with empty 'diagnosis' field.")

    # Check if 'ecg_path' column exists
    if 'ecg_path' not in df.columns:
        raise ValueError("'ecg_path' column is missing in the DataFrame.")
    
    # Check if the files in 'ecg_path' column exist
    missing_files = df[~df['ecg_path'].apply(os.path.exists)]
    if not missing_files.empty:
        raise FileNotFoundError(f"The following files are missing: {missing_files['ecg_path'].tolist()[:5]}")

    dataset = ProjectDataset(df)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)




