import argparse



class HearWiseArgs:
    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Script to process ECG data.')
        parser.add_argument('--diagnosis_classifier_device', help='Device to run the diagnosis classifier on', type=str, required=True)
        parser.add_argument('--signal_processing_device', help='Device to run the signal processing on', type=str, required=True)
        parser.add_argument('--data_path', help='Path to the data rows csv file', type=str, required=True)
        parser.add_argument('--batch_size', help='Batch size', type=int, required=True)
        parser.add_argument('--output_folder', help='Path to the output folder', type=str, required=True)
        parser.add_argument('--hugging_face_api_key_path', help='Path to the Hugging Face API key', type=str, required=True)
        parser.add_argument('--use_wcr', help='Use WCR for signal processing', type=HearWiseArgs.str2bool, required=True)
        parser.add_argument('--use_efficientnet', help='Use EfficientNet for signal processing', type=HearWiseArgs.str2bool, required=True)
        parser.add_argument('--ecg_signals_path', help='Path to the ECG signals files', type=str, required=True)
        parser.add_argument('--mode', help='Mode of the script', type=str, required=True)
        parser.add_argument('--preprocessing_folder', help='Path to the preprocessing folder', type=str, required=True)
        parser.add_argument('--preprocessing_n_workers', help='Number of workers for the preprocessing', type=int, default=16)
        return parser.parse_args()