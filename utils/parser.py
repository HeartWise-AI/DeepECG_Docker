import argparse



class HearWiseArgs:
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Script to process ECG data.')
        parser.add_argument('--diagnosis_classifier_device', help='Device to run the diagnosis classifier on', type=str, required=True)
        parser.add_argument('--signal_processing_device', help='Device to run the signal processing on', type=str, required=True)
        parser.add_argument('--data_path', help='Path to the data rows csv file', type=str, required=True)
        parser.add_argument('--batch_size', help='Batch size', type=int, required=True)
        parser.add_argument('--output_file', help='Name of the output file', type=str, required=True)
        parser.add_argument('--output_folder', help='Path to the output folder', type=str, required=True)
        parser.add_argument('--hugging_face_api_key_path', help='Path to the Hugging Face API key', type=str, required=True)
        parser.add_argument('--signal_processing_model_name', help='Name of the signal processing model', type=str, required=True)
        parser.add_argument('--diagnosis_classifier_model_name', help='Name of the diagnosis classifier model', type=str, required=True)
        parser.add_argument('--ecg_signals_path', help='Path to the ECG signals files', type=str, required=True)
        return parser.parse_args()