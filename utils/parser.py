import argparse



class HearWiseArgs:
    """Parse and validate command-line arguments for the ECG pipeline."""

    @staticmethod
    def str2bool(v):
        """Convert string to bool (e.g. 'true'/'false', 'yes'/'no'). Raises ArgumentTypeError if invalid."""
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    @staticmethod
    def parse_models(v):
        """Parse "signal_models:efficientnet,wcr;image_models:deepecg_png" into {key: [model, ...]}."""
        if isinstance(v, dict):
            return v
        models = {}
        for entry in filter(None, (e.strip() for e in v.split(';'))):
            key, sep, names = entry.partition(':')
            if not sep or not key.strip().endswith('_models'):
                raise argparse.ArgumentTypeError(f"Invalid models entry '{entry}' (expected <key>_models:<model>,...)")
            models[key.strip()] = [n.strip() for n in names.split(',') if n.strip()]
        return models

    @staticmethod
    def parse_arguments():
        """Parse command-line arguments and return the namespace (diagnosis_classifier_device, data_path, etc.)."""
        parser = argparse.ArgumentParser(description='Script to process ECG data.')
        parser.add_argument('--diagnosis_classifier_device', help='Device to run the diagnosis classifier on', type=str, required=True)
        parser.add_argument('--signal_processing_device', help='Device to run the signal processing on', type=str, required=True)
        parser.add_argument('--data_path', help='Path to the data rows csv file', type=str, required=True)
        parser.add_argument('--batch_size', help='Batch size', type=int, required=True)
        parser.add_argument('--output_folder', help='Path to the output folder', type=str, required=True)
        parser.add_argument('--hugging_face_api_key_path', help='Path to the Hugging Face API key', type=str, required=True)
        parser.add_argument('--models', help='Models to run, from heartwise.config: "key:model,model;..." with keys '
                            'signal_models, image_models and optional <task>_signal_models / <task>_image_models',
                            type=HearWiseArgs.parse_models, default={})
        # Deprecated (replaced by signal_models): still accepted for older configs and scripts
        parser.add_argument('--use_wcr', help='Deprecated: use signal_models', type=HearWiseArgs.str2bool, default=None)
        parser.add_argument('--use_efficientnet', help='Deprecated: use signal_models', type=HearWiseArgs.str2bool, default=None)
        parser.add_argument('--ecg_signals_path', help='Path to the ECG signals files', type=str, required=True)
        parser.add_argument('--mode', help='Mode of the script', type=str, required=True)
        parser.add_argument('--preprocessing_folder', help='Path to the preprocessing folder', type=str, required=True)
        parser.add_argument('--preprocessing_n_workers', help='Number of workers for the preprocessing', type=int, default=16)
        return parser.parse_args()
