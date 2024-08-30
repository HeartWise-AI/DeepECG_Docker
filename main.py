import os
import torch
import argparse

from utils.parser import HearWiseArgs
from models import HeartWiseModelFactory
from utils.analysis_pipeline import AnalysisPipeline
from utils.files_handler import save_to_csv, save_to_json


def main(args: HearWiseArgs):
    data_path = args.data_path
    batch_size = args.batch_size
    output_file = args.output_file
    output_folder = args.output_folder
    device = torch.device(args.device)
    signal_processing_model_name = args.signal_processing_model_name
    diagnosis_classifier_model_name = args.diagnosis_classifier_model_name
    
    diagnosis_classifier_model = HeartWiseModelFactory.create_model(
        {
            'model_name': diagnosis_classifier_model_name,
            'map_location': device
        }
    )
        
    signal_processing_model = HeartWiseModelFactory.create_model(
        {
            'model_name': signal_processing_model_name,
            'map_location': device
        }
    )
    
    metrics = AnalysisPipeline.run_analysis(
        data_path=data_path,
        batch_size=batch_size,
        signal_processing_model=signal_processing_model,
        diagnosis_classifier_model=diagnosis_classifier_model
    )

    save_to_json(metrics, os.path.join(output_folder, f'{output_file}.json'))        
    save_to_csv(metrics, os.path.join(output_folder, f'{output_file}.csv'))

        
if __name__ == '__main__':
    args = HearWiseArgs.parse_arguments()
    main(args)

