import pandas as pd
from torch.utils.data import DataLoader, Dataset

from utils.image_preprocessing import load_ecg_image


class ECGImageDataset(Dataset):
    """PyTorch Dataset that yields diagnosis, the preprocessed ECG page image and file name per row.

    Rows may have no label (inference only): a missing numeric label is NaN, a missing
    report text is the empty string.
    """

    def __init__(self, df: pd.DataFrame, preprocessing: dict, text_labels: bool = False):
        self.df = df.reset_index(drop=True)
        self.preprocessing = preprocessing
        self.text_labels = text_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Return dict with 'diagnosis', 'ecg_signal' (the image tensor, the model input) and 'file_name'."""
        row = self.df.loc[idx]
        diagnosis = row['diagnosis']
        if self.text_labels:
            diagnosis = "" if pd.isna(diagnosis) else str(diagnosis)
        else:
            diagnosis = float('nan') if pd.isna(diagnosis) else float(diagnosis)
        return {
            'diagnosis': diagnosis,
            'ecg_signal': load_ecg_image(row['ecg_path'], self.preprocessing),
            'file_name': str(row['file_name'])
        }


def create_image_dataloader(
    df: pd.DataFrame,
    preprocessing: dict,
    batch_size: int = 1,
    text_labels: bool = False,
    num_workers: int = 0
):
    """Build a DataLoader over the given DataFrame using ECGImageDataset."""
    dataset = ECGImageDataset(df, preprocessing=preprocessing, text_labels=text_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
