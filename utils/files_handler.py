import os
import csv
import json
import struct
import base64

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from concurrent.futures import ProcessPoolExecutor, as_completed


def save_to_csv(metrics: dict, path: str) -> None:
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))

    # Open the file and create a CSV writer
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for metric in metrics:
            for value in metrics[metric]:
                writer.writerow([metric, value, metrics[metric][value]])

def save_to_json(data: dict, path: str) -> None:
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def read_api_key(path: str) -> dict[str, str]:
    with open(path) as f:
        api_key = json.load(f)
    return api_key

def save_ecg_signal(data, filename):
    base64_str = base64.b64encode(data.tobytes()).decode('utf-8')
    with open(filename, 'w') as f:
        f.write(base64_str)

def load_ecg_signal(filename):
    with open(filename, 'r') as f:
        base64_str = f.read()
    binary_data = base64.b64decode(base64_str)
    return np.frombuffer(binary_data, dtype=np.float32).reshape((2500, 12))

class XMLProcessor:
    def __init__(self):
        self.report = []

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

    @staticmethod
    def process_waveform_data(df, waveform_keys, expected_shape, decode_base64=False):
        waveforms = {}
        correct_lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        for key in waveform_keys:
            if key in df.columns:
                data = df[key].iloc[0]
                if decode_base64 and isinstance(data, str):
                    data = XMLProcessor.decode_as_base64(data)
                else:
                    data = np.array(data.split(','), dtype=float)
                # actual_lengths.append(len(data))
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
    def xml_to_dataframe(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        xml_dict = XMLProcessor.parse_xml_to_dict(root)
        flattened_dict = XMLProcessor.flatten_dict(xml_dict)
        df = pd.DataFrame([flattened_dict])
        return df

    def process_single_file(self, file_path):
        try:
            df = self.xml_to_dataframe(file_path)
            file_id = os.path.splitext(os.path.basename(file_path))[0]

            if 'RestingECGMeasurements.MeasurementTable.LeadOrder' in df.columns and any(f'RestingECGMeasurements.MedianSamples.WaveformData.{i}' in df.columns for i in range(12)):
                xml_type = 'CLSA'
                self._process_clsa_xml(df, file_id)
            elif any(f'Waveform.1.LeadData.{j}.LeadID' in df.columns for j in range(12)):
                xml_type = 'MHI'
                self._process_mhi_xml(df, file_id)
            else:
                xml_type = 'Unknown'
                return (file_id, xml_type, 'Failed', 'Unknown XML format'), None, None

            df['file_id'] = file_id
            return (file_id, xml_type, 'Success', ''), df, self.full_leads_array
        except Exception as e:
            return (file_id, 'Unknown', 'Failed', str(e)), None, None

    def process_batch(self, directory_path, num_workers=None):
        xml_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.xml')]
        
        if not xml_files:
            self.report.append(('N/A', 'N/A', 'Failed', f'No XML files found in the directory: {directory_path}'))
            return [], []

        dfs = []
        lead_arrays = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(self.process_single_file, file): file for file in xml_files}
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    report_entry, df, lead_array = future.result()
                    self.report.append(report_entry)
                    if df is not None and lead_array is not None:
                        dfs.append(df)
                        lead_arrays.append(lead_array)
                except Exception as e:
                    file_id = os.path.splitext(os.path.basename(file))[0]
                    self.report.append((file_id, 'Unknown', 'Failed', str(e)))

        return dfs, lead_arrays

    def _process_clsa_xml(self, df, file_id):
        try:
            strip_data_keys = [f'StripData.WaveformData.{i}' for i in range(12)]
            leads = []
            for key in strip_data_keys:
                if key in df.columns:
                    lead_data = df[key].iloc[0].lstrip('\t').split(',')
                    lead_data = np.array(lead_data, dtype=float)
                    leads.append(lead_data)

            if len(leads) == 12:
                self.full_leads_array = np.vstack(leads)
            else:
                self.full_leads_array = self.process_waveform_data(df, strip_data_keys, expected_shape=(12, 2500), decode_base64=False)
        except Exception as e:
            print(f"\033[91mError processing file {file_id}: {str(e)}\033[0m")

    def _process_mhi_xml(self, df, file_id):
        try:
            strip_data_keys = [f'Waveform.1.LeadData.{i}.WaveFormData' for i in range(12)]
            leads = []

            for key in strip_data_keys:
                if key in df.columns:
                    lead_data = self.decode_as_base64(df[key].iloc[0])
                    leads.append(lead_data)
        
            if len(leads) == 12:
                self.full_leads_array = np.vstack(leads)
                print(self.full_leads_array.shape)
            else:
                self.full_leads_array = self.process_waveform_data(df, strip_data_keys, expected_shape=(12, 2500), decode_base64=True)
        except Exception as e:
            print(f"\033[91mError processing file {file_id}: {str(e)}\033[0m")

    def save_report(self, output_directory):
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
        detailed_report_path = os.path.join(output_directory, 'ecg_processing_detailed_report.csv')
        report_df.to_csv(detailed_report_path, index=False)

        # Save summary report
        summary_report_path = os.path.join(output_directory, 'ecg_processing_summary_report.csv')
        summary_df.to_csv(summary_report_path, index=False)



if __name__ == "__main__":
    root_dir = "/path/to/your/xml/files"
    
    # # Process single file
    xml_processor = XMLProcessor()
    report_entry,df, full_leads_array = xml_processor.process_single_file(
        file_path=os.path.join(root_dir, ".xml")
    )
    print(full_leads_array.shape)
    
    # Process all XML files in a directory
    dfs, lead_arrays = xml_processor.process_batch(
        directory_path=root_dir,
        num_workers=4
    )
    print(f"Processed {len(dfs)} files.")
    
    # Save report
    xml_processor.save_report(output_directory=".")