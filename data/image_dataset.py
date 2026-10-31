import os

import pandas as pd
from torch.utils.data import DataLoader, Dataset, default_collate

from utils.image_preprocessing import is_pdf, load_ecg_page
from utils.log_config import get_logger

logger = get_logger(__name__)

# PDF rows whose page could not be extracted while the DataLoader read them, ecg_path -> reason.
# main lists them with the unreadable images at the end of the run (and clears this at its start
# and end).
PAGE_ERRORS: dict[str, str] = {}
# The file as it was when it failed (size, mtime): a PDF that changed since is read again
_FAILED_AS: dict[str, tuple] = {}


def clear_page_errors() -> None:
    """Forget the PDFs that failed (main: at the start and the end of a run)."""
    PAGE_ERRORS.clear()
    _FAILED_AS.clear()


def _file_signature(path: str) -> tuple | None:
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_size, info.st_mtime_ns)


class ECGImageDataset(Dataset):
    """PyTorch Dataset that yields diagnosis, the preprocessed ECG page image and file name per row.

    Rows may have no label (inference only): a missing numeric label is NaN, a missing
    report text is the empty string.

    A PDF row's page is extracted here, when the DataLoader asks for the row (in memory, never
    before): a PDF whose page cannot be extracted yields None, is recorded in PAGE_ERRORS (and not
    extracted again while it is there and unchanged on disk) and is left out of its batch
    (collate_pages). PNG / JPEG rows were checked before inference.
    """

    def __init__(self, df: pd.DataFrame, preprocessing: dict, text_labels: bool = False):
        self.df = df.reset_index(drop=True)
        self.preprocessing = preprocessing
        self.text_labels = text_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Return dict with 'diagnosis', 'ecg_signal' (the image tensor, the model input) and 'file_name'
        (None for a PDF whose page cannot be extracted)."""
        row = self.df.loc[idx]
        diagnosis = row['diagnosis']
        if self.text_labels:
            diagnosis = "" if pd.isna(diagnosis) else str(diagnosis)
        else:
            diagnosis = float('nan') if pd.isna(diagnosis) else float(diagnosis)
        path = row['ecg_path']
        if is_pdf(path):
            if path in PAGE_ERRORS and _FAILED_AS.get(path) == _file_signature(path):
                return None                    # failed for an earlier task of this run, unchanged since
            try:
                page = load_ecg_page(path, self.preprocessing)
            except ValueError as e:            # utils.ecg_pdf.EcgPdfError is a ValueError
                PAGE_ERRORS.setdefault(path, f"{type(e).__name__}: {e}")   # the first reason is kept
                _FAILED_AS[path] = _file_signature(path)
                return None
        else:
            page = load_ecg_page(path, self.preprocessing)
        return {
            'diagnosis': diagnosis,
            'ecg_signal': page,
            'file_name': str(row['file_name'])
        }


def collate_pages(items):
    """default_collate of the rows that loaded; None when none of the batch did."""
    items = [item for item in items if item is not None]
    return default_collate(items) if items else None


def create_image_dataloader(
    df: pd.DataFrame,
    preprocessing: dict,
    batch_size: int = 1,
    text_labels: bool = False,
    num_workers: int = 0
):
    """Build a DataLoader over the given DataFrame using ECGImageDataset.

    With any PDF row the DataLoader runs in the main process (num_workers 0), so only one PDF
    is extracted at a time.
    """
    if num_workers and df['ecg_path'].map(is_pdf).any():
        logger.info("PDF rows: DataLoader num_workers %d -> 0 (one PDF extracted at a time)", num_workers)
        num_workers = 0
    dataset = ECGImageDataset(df, preprocessing=preprocessing, text_labels=text_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=collate_pages)
