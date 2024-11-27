
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from heartwise_statplots.files_handler import DicomReader


def find_dcm_files_in_subfolders(folder_path):
    dcm_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dcm"):
                dcm_files.append(os.path.join(root, file))
    return dcm_files


dicom_folder_paths = find_dcm_files_in_subfolders("/tmp/dcm_input")

ecg_path = "/tmp/dcm_output"
cv_file_path = "/tmp/ECG_metadata.csv"
os.makedirs(ecg_path, exist_ok=True)

docker_list = []
for i, dicom_folder_path in enumerate(tqdm(dicom_folder_paths, desc="Processing dicoms", total=len(dicom_folder_paths))):
    dicom = DicomReader.read_dicom_file(dicom_folder_path)
    diagnosis = DicomReader.extract_diagnosis_from_dicom(dicom)
    
    ecg_signal = DicomReader.extract_ecg_from_dicom(dicom)
    np_file_name = f"ecg_signal_{i}.npy"
    ecg_npy_path = os.path.join(ecg_path, np_file_name)
    np.save(ecg_npy_path, ecg_signal)
    docker_list.append(
        {
            "77_classes_ecg_file_name": np_file_name,
            "ecg_machine_diagnosis": diagnosis
        }
    )

df = pd.DataFrame(docker_list)
df.to_csv(cv_file_path, index=False)


