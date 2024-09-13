import os
import csv
import json
import struct
import base64
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def load_csv_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def set_path(df: pd.DataFrame, path: str) -> pd.DataFrame:
    df['ecg_path'] = df['ecg_file_name'].apply(lambda x: os.path.join(path, x))
    return df

def save_to_csv(metrics: dict, path: str) -> None:
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))

    # Collect all unique subkeys
    all_subkeys = set()
    for key, value in metrics.items():
        if isinstance(value, dict):
            all_subkeys.update(value.keys())

    # Sort the subkeys for consistent column order
    sorted_subkeys = sorted(all_subkeys)

    # Prepare the rows
    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            row = {'Key': key}
            for subkey in sorted_subkeys:
                row[subkey] = value.get(subkey, '')
            rows.append(row)

    # Write to CSV
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Key'] + sorted_subkeys)
        writer.writeheader()
        writer.writerows(rows)

def save_to_json(data: dict, path: str) -> None:
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def read_api_key(path: str) -> dict[str, str]:
    with open(path) as f:
        api_key = json.load(f)
    return api_key


class ECGFileHandler:
    @staticmethod
    def save_ecg_signal(ecg_signal, filename) -> None:  
        if filename.endswith('.base64'):
            ecg_signal = ecg_signal.astype(np.float32)
            base64_str = base64.b64encode(ecg_signal.tobytes()).decode('utf-8')
            with open(filename, 'w') as f:
                f.write(base64_str)
        else:
            np.save(filename, ecg_signal)
    
    @staticmethod
    def load_ecg_signal(filename) -> np.ndarray:
        if filename.endswith('.base64'):
            with open(filename, 'r') as f:
                base64_str = f.read()
            np_array = np.frombuffer(base64.b64decode(base64_str), dtype=np.float32)
        else:
            np_array = np.load(filename)
        writable_array = np.copy(np_array)
        return writable_array.reshape(-1, 12)
    
    @staticmethod
    def list_files(directory_path: str) -> list[str]:
        return [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

class XMLProcessor:
    def __init__(self):
        self.report = []
        self.tmp_folder = "./tmp"
        self.expected_shape = (2500, 12)

    @staticmethod
    def parse_xml_to_dict(element):
        if len(element) == 0:
            return element.text
        result = {}
        for child in element:
            child_result = XMLProcessor.parse_xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_result)
            else:
                result[child.tag] = child_result
        return result

    @staticmethod
    def flatten_dict(d, parent_key=''):
        items = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f'{parent_key}.{k}' if parent_key else k
                items.extend(XMLProcessor.flatten_dict(v, new_key).items())
        elif isinstance(d, list):
            for i, item in enumerate(d):
                items.extend(XMLProcessor.flatten_dict(item, f'{parent_key}.{i}').items())
        else:
            items.append((parent_key, d))
        return dict(items)

    @staticmethod
    def decode_as_base64(raw_wave):
        arr = base64.b64decode(bytes(raw_wave, "utf-8"))
        unpack_symbols = "".join([char * (len(arr) // 2) for char in "h"])
        byte_array = struct.unpack(unpack_symbols, arr)
        return np.array(byte_array, dtype=np.float32)

    def process_waveform_data(self, data_dict, waveform_keys, expected_shape, decode_base64=False):
        waveforms = {}
        correct_lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        for key in waveform_keys:
            if key in data_dict:
                data = data_dict[key]
                if decode_base64 and isinstance(data, str):
                    data = self.decode_as_base64(data)
                else:
                    data = np.array(data.split(','), dtype=float)
                lead_name = correct_lead_order[waveform_keys.index(key)]
                waveforms[lead_name] = data

        if "I" in waveforms and "II" in waveforms:
            if "III" not in waveforms:
                waveforms["III"] = np.subtract(waveforms["II"], waveforms["I"])
            if "aVR" not in waveforms:
                waveforms["aVR"] = np.add(waveforms["I"], waveforms["II"]) * (-0.5)
            if "aVL" not in waveforms:
                waveforms["aVL"] = np.subtract(waveforms["I"], 0.5 * waveforms["II"])
            if "aVF" not in waveforms:
                waveforms["aVF"] = np.subtract(waveforms["II"], 0.5 * waveforms["I"])

        leads = []
        for lead in correct_lead_order:
            if lead in waveforms:
                leads.append(waveforms[lead])
            else:
                leads.append(np.full(expected_shape[1], np.nan))

        return np.vstack(leads)
    
    @staticmethod
    def xml_to_dict(xml_file: str) -> dict:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        return XMLProcessor.flatten_dict(XMLProcessor.parse_xml_to_dict(root))

    def process_single_file(self, file_path: str) -> tuple[tuple[str, str, str, str], str, np.ndarray]:
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        try:
            data_dict = self.xml_to_dict(file_path)
            
            if 'RestingECGMeasurements.MeasurementTable.LeadOrder' in data_dict and any(f'RestingECGMeasurements.MedianSamples.WaveformData.{i}' in data_dict for i in range(12)):
                xml_type = 'CLSA'
                self._process_clsa_xml(data_dict, file_id)
            elif any(f'Waveform.1.LeadData.{j}.LeadID' in data_dict for j in range(12)):
                xml_type = 'MHI'
                self._process_mhi_xml(data_dict, file_id)
            else:
                xml_type = 'Unknown'
                return (file_id, xml_type, 'Failed', 'Unknown XML format'), None, None

            return (file_id, xml_type, 'Success', ''), file_id, self.full_leads_array
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return (file_id, 'Unknown', 'Failed', str(e)), file_id, None

    def process_batch(self, df: pd.DataFrame, num_workers: int = 32) -> tuple[list[str], np.ndarray]:
        xml_files = df['ecg_path'].tolist()
        ecgs = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(self.process_single_file, file): file for file in xml_files}
            for future in tqdm(as_completed(future_to_file), total=len(xml_files), desc="Processing files"):
                file = future_to_file[future]
                try:
                    report_entry, file_id, lead_array = future.result()
                    self.report.append(report_entry)
                    
                    if lead_array is not None:
                        if lead_array.shape[0] == self.expected_shape[1]: # if is (12, X) then transpose it to (X, 12)
                            lead_array = lead_array.transpose(1, 0)
                        if lead_array.shape[0] != self.expected_shape[0]: # if X != 2500 then resize it
                            if lead_array.shape[0] < self.expected_shape[0]: # if X < 2500 then skip it
                                print(f"Warning: Lead array length is less than 2500 for file {file_id}.")
                                continue
                            else: # if X > 2500 then resize it
                                step = lead_array.shape[0] // self.expected_shape[0]
                                lead_array = lead_array[::step, :]
                        
                        # store the lead array
                        ecgs.append(lead_array)
                        
                        # store the file path
                        df['ecg_path'] = os.path.join(self.tmp_folder, f"{file_id}.base64")
                except Exception as e:
                    print(f"Error processing file: {str(e)}")
                    file_id = os.path.splitext(os.path.basename(file))[0]
                    self.report.append((file_id, 'Unknown', 'Failed', str(e)))

        ecgs = np.array(ecgs)
        return df, ecgs

    def _process_clsa_xml(self, data_dict: dict, file_id: str) -> None:
        try:
            strip_data_keys = [f'StripData.WaveformData.{i}' for i in range(12)]
            leads = []
            for key in strip_data_keys:
                if key in data_dict:
                    lead_data = data_dict[key].lstrip('\t').split(',')
                    lead_data = np.array(lead_data, dtype=float)
                    leads.append(lead_data)

            if len(leads) == 12:
                self.full_leads_array = np.vstack(leads)
            else:
                self.full_leads_array = self.process_waveform_data(
                    data_dict=data_dict, 
                    waveform_keys=strip_data_keys, 
                    expected_shape=(12, 2500), 
                    decode_base64=False
                )
        except Exception as e:
            raise ValueError(f"Error processing CLSA XML for file {file_id}: {str(e)}") from e

    def _process_mhi_xml(self, data_dict: dict, file_id: str) -> None:
        try:
            strip_data_keys = [f'Waveform.1.LeadData.{i}.WaveFormData' for i in range(12)]
            leads = []

            for key in strip_data_keys:
                if key in data_dict:
                    lead_data = self.decode_as_base64(data_dict[key])
                    leads.append(lead_data)
        
            if len(leads) == 12:
                self.full_leads_array = np.vstack(leads)
            else:
                self.full_leads_array = self.process_waveform_data(
                    data_dict=data_dict, 
                    waveform_keys=strip_data_keys, 
                    expected_shape=(12, 2500), 
                    decode_base64=True
                )
        except Exception as e:
            raise ValueError(f"Error processing MHI XML for file {file_id}: {str(e)}") from e

    def save_report(self, output_folder: str) -> None:        
        report_df = pd.DataFrame(self.report, columns=['file_id', 'xml_type', 'status', 'message'])
        
        # Calculate summary statistics
        total_files = len(report_df)
        successful_files = sum(report_df['status'] == 'Success')
        failed_files = sum(report_df['status'] == 'Failed')
        xml_type_distribution = report_df['xml_type'].value_counts().to_dict()

        # Create summary DataFrame
        summary_data = {
            'Metric': ['Total Files', 'Successful Files', 'Failed Files'] + [f'XML Type: {k}' for k in xml_type_distribution.keys()],
            'Value': [total_files, successful_files, failed_files] + list(xml_type_distribution.values())
        }
        summary_df = pd.DataFrame(summary_data)

        # Save detailed report
        detailed_report_path = os.path.join(output_folder, 'ecg_processing_detailed_report.csv')
        report_df.to_csv(detailed_report_path, index=False)

        # Save summary report
        summary_report_path = os.path.join(output_folder, 'ecg_processing_summary_report.csv')
        summary_df.to_csv(summary_report_path, index=False)



if __name__ == "__main__":
    root_dir = "/path/to/your/xml/files"
    output_folder = "test_results"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Process single file
    xml_processor = XMLProcessor()
    
    # Process all XML files in a directory
    processed_files = xml_processor.process_batch(
        directory_path=root_dir,
        num_workers=4
    )
    print(f"Processed {len(processed_files)} files.")
    
    # Save report
    xml_processor.save_report(output_folder=output_folder)