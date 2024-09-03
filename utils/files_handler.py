import os
import csv
import json

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


import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import base64
import struct
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['figure.figsize'] = 20, 20
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False

def parse_xml_to_dict(element):
    """Recursively parses an XML element and its children into a dictionary."""
    if len(element) == 0:
        return element.text
    result = {}
    for child in element:
        child_result = parse_xml_to_dict(child)
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_result)
        else:
            result[child.tag] = child_result
    return result

def flatten_dict(d, parent_key=''):
    """Flattens a nested dictionary."""
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f'{parent_key}.{k}' if parent_key else k
            items.extend(flatten_dict(v, new_key).items())
    elif isinstance(d, list):
        for i, item in enumerate(d):
            items.extend(flatten_dict(item, f'{parent_key}.{i}').items())
    else:
        items.append((parent_key, d))
    return dict(items)

def decode_ekg_muse(raw_wave):
    """Ingest the base64 encoded waveforms and transform them to numeric."""
    arr = base64.b64decode(bytes(raw_wave, "utf-8"))
    unpack_symbols = "".join([char * (len(arr) // 2) for char in "h"])
    byte_array = struct.unpack(unpack_symbols, arr)
    return np.array(byte_array, dtype=np.float32)  # Convert to float32 for consistency


def process_waveform_data(df, waveform_keys, expected_shape, decode_base64=False):
    """Process specific waveform data fields into arrays."""
    waveforms = {}
    actual_lengths = []
    correct_lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    # Collect data for each lead
    for key in waveform_keys:
        if key in df.columns:
            data = df[key].iloc[0]
            if decode_base64 and isinstance(data, str):
                data = decode_ekg_muse(data)
            else:
                data = np.array(data.split(','), dtype=float)
            actual_lengths.append(len(data))
            lead_name = correct_lead_order[waveform_keys.index(key)]  # Use the correct lead order
            waveforms[lead_name] = data

    # Compute missing leads if I and II are available
    if "I" in waveforms and "II" in waveforms:
        if "III" not in waveforms:
            waveforms["III"] = np.subtract(waveforms["II"], waveforms["I"])
        if "aVR" not in waveforms:
            waveforms["aVR"] = np.add(waveforms["I"], waveforms["II"]) * (-0.5)
        if "aVL" not in waveforms:
            waveforms["aVL"] = np.subtract(waveforms["I"], 0.5 * waveforms["II"])
        if "aVF" not in waveforms:
            waveforms["aVF"] = np.subtract(waveforms["II"], 0.5 * waveforms["I"])
        print(f"\033[93mComputed missing leads: {'III', 'aVR', 'aVL', 'aVF'}\033[0m")

    # Construct the waveform array based on the correct lead order
    leads = []
    for lead in correct_lead_order:
        if lead in waveforms:
            leads.append(waveforms[lead])
        else:
            leads.append(np.full(expected_shape[1], np.nan))  # Fill missing leads with NaNs

    waveform_array = np.vstack(leads)

    # Calculate NaN count and flat lead count
    nan_count = np.isnan(waveform_array).sum()
    flat_lead_count = np.sum(np.all(waveform_array == 0, axis=1))

    return waveform_array, nan_count, flat_lead_count, actual_lengths, correct_lead_order

def xml_to_dataframe(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    xml_dict = parse_xml_to_dict(root)
    flattened_dict = flatten_dict(xml_dict)
    df = pd.DataFrame([flattened_dict])
    return df

def cleanup_PTB_labels(labels):
    return ' - '.join(labels)

def add_line_breaks(text):
    if len(text) <= 175:
        return text

    result = []
    start = 0
    while start < len(text):
        end = start + 175
        if end >= len(text):
            result.append(text[start:])
            break

        delimiter_index = text.rfind(' - ', start, end)
        if delimiter_index == -1:
            result.append(text[start:end])
            start = end
        else:
            result.append(text[start:delimiter_index])
            start = delimiter_index + 3  # Skip over the ' - ' delimiter

    return '\n'.join(result)

def plot_the_ECG_UKB(matrix, label, lead_order, resolution):
    resolution_value = resolution * 0.001  # Convert to desired units
    matrix *= resolution_value  # Apply the resolution to the matrix values

    if matrix.shape == (12, 5000):
        matrix = matrix[:, 1::2]  # Downsample to 2500

    print(lead_order)
    lead_dict = dict(zip(lead_order.split(', '), matrix))

    activation = [0] * 5 + [10] * 50 + [0] * 5

    pannel_1_y = [i + 50 for i in activation] + \
                 [i * 10 + 50 for i in lead_dict['I'][60:625]] + \
                 [i * 10 + 50 for i in lead_dict['aVR'][625:1190]] + \
                 [i * 10 + 50 for i in lead_dict['V1'][1190:1755]] + \
                 [i * 10 + 50 for i in lead_dict['V4'][1755:]]

    pannel_2_y = [i + 15 for i in activation] + \
                 [i * 10 + 15 for i in lead_dict['II'][60:625]] + \
                 [i * 10 + 15 for i in lead_dict['aVL'][625:1190]] + \
                 [i * 10 + 15 for i in lead_dict['V2'][1190:1755]] + \
                 [i * 10 + 15 for i in lead_dict['V5'][1755:]]

    pannel_3_y = [i - 15 for i in activation] + \
                 [i * 10 - 15 for i in lead_dict['III'][60:625]] + \
                 [i * 10 - 15 for i in lead_dict['aVF'][625:1190]] + \
                 [i * 10 - 15 for i in lead_dict['V3'][1190:1755]] + \
                 [i * 10 - 15 for i in lead_dict['V6'][1755:]]

    pannel_4_y = [i - 50 for i in activation] + \
                 [i * 10 - 50 for i in lead_dict['II'][60:]]

    fig, ax = plt.subplots(figsize=(40, 20))
    ax.minorticks_on()

    # Vertical lines for lead labels
    ax.vlines(60, -10, -20, label='III', linewidth=4)
    ax.text(60, -10, 'III', fontsize=44)
    ax.vlines(625, -10, -20, label='aVF', linewidth=4)
    ax.text(625, -10, 'aVF', fontsize=44)
    ax.vlines(1250, -10, -20, label='V3', linewidth=4)
    ax.text(1250, -10, 'V3', fontsize=44)
    ax.vlines(1875, -10, -20, label='V6', linewidth=4)
    ax.text(1875, -10, 'V6', fontsize=44)
    ax.vlines(60, 10, 20, label='II', linewidth=4)
    ax.text(60, 20, 'II', fontsize=44)
    ax.vlines(625, 10, 20, label='aVL', linewidth=4)
    ax.text(625, 20, 'aVL', fontsize=44)
    ax.vlines(1250, 10, 20, label='V2', linewidth=4)
    ax.text(1250, 20, 'V2', fontsize=44)
    ax.vlines(1875, 10, 20, label='V5', linewidth=4)
    ax.text(1875, 20, 'V5', fontsize=44)
    ax.vlines(60, 45, 55, label='I', linewidth=4)
    ax.text(60, 55, 'I', fontsize=44)
    ax.vlines(625, 45, 55, label='aVR', linewidth=4)
    ax.text(625, 55, 'aVR', fontsize=44)
    ax.vlines(1250, 45, 55, label='V1', linewidth=4)
    ax.text(1250, 55, 'V1', fontsize=44)
    ax.vlines(1875, 45, 55, label='V4', linewidth=4)
    ax.text(1875, 55, 'V4', fontsize=44)
    ax.vlines(60, -55, -45, label='II', linewidth=4)
    ax.text(60, -45, 'II', fontsize=44)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    ax.grid(ls='-', color='red', linewidth=1.2)
    ax.grid(which="minor", ls=':', color='red', linewidth=1)

    ax.axis([0 - 100, 2500 + 100, min(pannel_4_y) - 10, max(pannel_1_y) + 10])

    x = [pos for pos in range(0, len(pannel_1_y))]

    ax.plot(x, pannel_1_y, linewidth=3, color='#000000')
    ax.plot(x, pannel_2_y, linewidth=3, color='#000000')
    ax.plot(x, pannel_3_y, linewidth=3, color='#000000')
    ax.plot(x, pannel_4_y, linewidth=3, color='#000000')

    label = cleanup_PTB_labels(label)
    label = add_line_breaks(str(label))

    plt.title(label, fontsize=27)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the plot layout

    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

def process_single_file(file_path):
    df = xml_to_dataframe(file_path)
    file_id = os.path.splitext(os.path.basename(file_path))[0]

    # Detect if XML is CLSA or MHI
    if 'RestingECGMeasurements.MeasurementTable.LeadOrder' in df.columns and any(f'RestingECGMeasurements.MedianSamples.WaveformData.{i}' in df.columns for i in range(12)):
        print(f"\033[92mDetected CLSA XML type for file {file_id}.\033[0m")

        # Print relevant CLSA parameters
        print("\033[96mCLSA Parameters:\033[0m")
        print(f"Resolution (StripData.Resolution): \033[93m{df.get('StripData.Resolution', 'Not found')}\033[0m")
        print(f"Sample Rate (RestingECGMeasurements.MedianSamples.SampleRate): \033[93m{df.get('RestingECGMeasurements.MedianSamples.SampleRate', 'Not found')}\033[0m")
        print(f"Low Pass Filter (FilterSetting.LowPass): \033[93m{df.get('FilterSetting.LowPass', 'Not found')}\033[0m")
        print(f"60Hz Filter (FilterSetting.Filter60Hz): \033[93m{df.get('FilterSetting.Filter60Hz', 'Not found')}\033[0m")
        print(f"50Hz Filter (FilterSetting.Filter50Hz): \033[93m{df.get('FilterSetting.Filter50Hz', 'Not found')}\033[0m")
        print(f"Cubic Spline Filter (FilterSetting.CubicSpline): \033[93m{df.get('FilterSetting.CubicSpline', 'Not found')}\033[0m")
        
        # Print the LeadOrder
        lead_order_series = df.get('RestingECGMeasurements.MeasurementTable.LeadOrder', pd.Series(['Not found']))
        lead_order = lead_order_series.iloc[0]  # Ensure lead_order is a string
        print(f"Lead Order: \033[93m{lead_order}\033[0m")

        # Process 12-lead data from StripData.WaveformData.{0-11}
        strip_data_keys = [f'StripData.WaveformData.{i}' for i in range(12)]
        leads = []

        for key in strip_data_keys:
            if key in df.columns:
                lead_data = df[key].iloc[0].lstrip('\t').split(',')
                lead_data = np.array(lead_data, dtype=float)
                leads.append(lead_data)
            else:
                print(f"\033[91mMissing data for {key}\033[0m")
        
        if len(leads) == 12:
            full_leads_array = np.vstack(leads)
            print(f"\033[92mProcessed full 12-lead data for file {file_id}.\033[0m")
            print(full_leads_array.shape)
            # Plot the 12-lead ECG data
            plot_the_ECG_UKB(full_leads_array, f"File ID: {file_id}", lead_order, float(df['StripData.Resolution'].iloc[0]))
        else:
            print(f"\033[91mIncomplete lead data for file {file_id}. Only {len(leads)} leads available.\033[0m")
      
    elif any(f'Waveform.1.LeadData.{j}.LeadID' in df.columns for j in range(12)):
        print(f"\033[92mDetected MHI XML type for file {file_id}.\033[0m")

        # Print relevant MHI parameters
        print("\033[96mMHI Parameters:\033[0m")
        print(f"Resolution (Waveform.1.LeadData.0.LeadAmplitudeUnitsPerBit): \033[93m{df.get('Waveform.1.LeadData.0.LeadAmplitudeUnitsPerBit', 'Not found')}\033[0m")
        print(f"Sample Rate (Waveform.1.LeadData.0.LeadSampleCountTotal): \033[93m{df.get('Waveform.1.LeadData.0.LeadSampleCountTotal', 'Not found')}\033[0m")
        print(f"Low Pass Filter (Waveform.1.LowPassFilter): \033[93m{df.get('Waveform.1.LowPassFilter', 'Not found')}\033[0m")
        print(f"High Pass Filter (Waveform.1.HighPassFilter): \033[93m{df.get('Waveform.1.HighPassFilter', 'Not found')}\033[0m")
        print(f"AC Filter (Waveform.1.ACFilter): \033[93m{df.get('Waveform.1.ACFilter', 'Not found')}\033[0m")

        # Print Lead IDs as Lead Order
        lead_ids = {f"Lead {j}": df[f'Waveform.1.LeadData.{j}.LeadID'].iloc[0] for j in range(12) if f'Waveform.1.LeadData.{j}.LeadID' in df.columns}
        lead_order = [lead_ids.get(f"Lead {j}", "Missing") for j in range(12)]
        print(f"Lead Order: \033[93m{lead_order}\033[0m")

        # Filter out the "Missing" entries
        lead_order = [lead for lead in lead_order if lead != "Missing"]

        # Process 12-lead data from Waveform.1.LeadData.{0-11}.WaveFormData
        strip_data_keys = [f'Waveform.1.LeadData.{i}.WaveFormData' for i in range(12)]
        leads = []

        resolution = float(df['Waveform.1.LeadData.0.LeadAmplitudeUnitsPerBit'].iloc[0])
        for key in strip_data_keys:
            if key in df.columns:
                lead_data = decode_ekg_muse(df[key].iloc[0])  # Decode base64 data
                leads.append(lead_data)
            else:
                print(f"\033[91mMissing data for {key}\033[0m")
        
        print(np.vstack(leads))
        if len(leads) == 12:
            full_leads_array = np.vstack(leads)
            print(f"\033[92mProcessed full 12-lead data for file {file_id}.\033[0m")
            print(full_leads_array.shape)
            # Plot the 12-lead ECG data
            plot_the_ECG_UKB(full_leads_array, f"File ID: {file_id}", ', '.join(lead_order), resolution)
        else:
            print(f"\033[91mIncomplete lead data for file {file_id}. Only {len(leads)} leads available.\033[0m")
            full_leads_array, _, _, _, lead_order = process_waveform_data(df, strip_data_keys, expected_shape=(12, 2500), decode_base64=True)
            print(full_leads_array)
            plot_the_ECG_UKB(full_leads_array, f"File ID: {file_id}", ', '.join(lead_order), resolution)

    else:
        print(f"\033[91mUnknown XML format for file {file_id}. No specific processing applied.\033[0m")

    df['file_id'] = file_id
    return df, full_leads_array
# Example Usage
file_path = '/media/data1/anolin/XML_test/CLSA_sample.xml'  # Replace with your XML file path

final_df, full_leads_array = process_single_file(file_path)
final_df.head()  # Display the resulting DataFrame in the notebook
