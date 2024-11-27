import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from heartwise_statplots.files_handler import DicomReader
import concurrent.futures
import multiprocessing

def find_dcm_files_in_subfolders(folder_path):
    dcm_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                dcm_files.append(os.path.join(root, file))
    return dcm_files

def process_dicom(args):
    index, dicom_folder_path, ecg_path = args
    try:
        # Read DICOM file
        dicom = DicomReader.read_dicom_file(dicom_folder_path)
        diagnosis = DicomReader.extract_diagnosis_from_dicom(dicom)
        
        # Extract and save ECG signal
        ecg_signal = DicomReader.extract_ecg_from_dicom(dicom)
        np_file_name = f"ecg_signal_{index}.npy"
        ecg_npy_path = os.path.join(ecg_path, np_file_name)
        np.save(ecg_npy_path, ecg_signal)
        
        return {
            "dicom_path": dicom_folder_path,
            "77_classes_ecg_file_name": np_file_name,
            "ecg_machine_diagnosis": diagnosis
        }
    except Exception as e:
        print(f'Failed for dicom at: {dicom_folder_path}. Error: {str(e)}')
        return None

def main():
    # Input and output paths
    dicom_folder_paths = find_dcm_files_in_subfolders("/tmp/dcm_input")
    ecg_path = "/tmp/dcm_output"
    cv_file_path = "/tmp/ECG_metadata.csv"
    os.makedirs(ecg_path, exist_ok=True)

    # Prepare arguments for parallel processing
    args_list = [(i, path, ecg_path) for i, path in enumerate(dicom_folder_paths)]

    # Calculate optimal number of processes
    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores, 12)  # Use up to 12 cores or all available if less

    # Process files in parallel using ProcessPoolExecutor
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Process results with progress bar
        futures = list(tqdm(
            executor.map(process_dicom, args_list),
            total=len(dicom_folder_paths),
            desc=f"Processing dicoms using {num_processes} processes"
        ))
        
        results = [f for f in futures if f is not None]

    # Create and save the DataFrame
    df = pd.DataFrame(results)
    df.to_csv(cv_file_path, index=False)

if __name__ == "__main__":
    main()