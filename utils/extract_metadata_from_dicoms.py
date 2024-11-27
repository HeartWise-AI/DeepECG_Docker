import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from heartwise_statplots.files_handler import DicomReader
import concurrent.futures
from functools import partial

def find_dcm_files_in_subfolders(folder_path):
    dcm_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                dcm_files.append(os.path.join(root, file))
    return dcm_files

def process_dicom(index_and_path, ecg_path):
    index, dicom_folder_path = index_and_path
    try:
        dicom = DicomReader.read_dicom_file(dicom_folder_path)
        diagnosis = DicomReader.extract_diagnosis_from_dicom(dicom)
        
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
    dicom_folder_paths = find_dcm_files_in_subfolders("/tmp/dcm_input")
    ecg_path = "/tmp/dcm_output"
    cv_file_path = "/tmp/ECG_metadata.csv"
    os.makedirs(ecg_path, exist_ok=True)

    docker_list = []
    
    # Create a partial function with fixed ecg_path
    process_dicom_partial = partial(process_dicom, ecg_path=ecg_path)
    
    # Create index-path pairs
    indexed_paths = list(enumerate(dicom_folder_paths))
    
    # Create a thread pool executor with 12 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Process results with progress bar
        for result in tqdm(
            executor.map(process_dicom_partial, indexed_paths),
            total=len(dicom_folder_paths),
            desc="Processing dicoms"
        ):
            if result is not None:
                docker_list.append(result)

    # Create and save the DataFrame
    df = pd.DataFrame(docker_list)
    df.to_csv(cv_file_path, index=False)

if __name__ == "__main__":
    main()