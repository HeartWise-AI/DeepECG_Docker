import argparse



class HearWiseArgs:
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Script to process ECG data.')
        parser.add_argument('--device', type=str, required=True)
        parser.add_argument('--data_path', type=str, required=True)
        parser.add_argument('--batch_size', type=int, required=True)
        parser.add_argument('--output_file', type=str, required=True)
        parser.add_argument('--output_folder', type=str, required=True)
        parser.add_argument('--signal_processing_model_name', type=str, required=True)
        parser.add_argument('--diagnosis_classifier_model_name', type=str, required=True)
        return parser.parse_args()