import argparse



class HearWiseArgs:
    """Parse and validate command-line arguments for the ECG pipeline."""
    
    @staticmethod
    def parse_models(v):
        """Parse "signal_models:efficientnet,wcr;image_models:deepecg_siglip2" into {key: [model, ...]}."""
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
        parser.add_argument('--hugging_face_api_key_path', help='Optional API-key JSON (created empty if missing)', type=str, default='api_key.json')
        parser.add_argument('--models', help='Models to run, from heartwise.config: '
                            '"signal_models:model,...;image_models:model,..."',
                            type=HearWiseArgs.parse_models, default={})
        parser.add_argument('--ecg_signals_path', help='Path to the ECG signals files', type=str, required=True)
        parser.add_argument('--mode', help='Mode of the script', type=str, required=True)
        parser.add_argument('--preprocessing_folder', help='Path to the preprocessing folder', type=str, required=True)
        parser.add_argument('--preprocessing_n_workers', help='Number of workers for the preprocessing', type=int, default=16)
        parser.add_argument('--single_lead', help="Single-lead NPY input: 'all' copies the lead to all 12 leads, a lead name (I ... V6) puts it in that lead only; 'none' = 12-lead input", type=str, default='none')
        return parser.parse_args()
