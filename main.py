import os
import json
import torch
from models import ModelFactory
from utils.analysis_pipeline import AnalysisPipeline, convert_to_df, save_to_csv, save_to_json



def main():
    output_folder = 'results'
    df_path = 'data/df.parquet'
    device = torch.device('cuda:0')
    bert_classifier = ModelFactory.create_model(
        {
            'model_name': 'bert_diagnosis2classification',
            'map_location': device
        }
    )
        
    efficient_netV2 = ModelFactory.create_model(
        {
            'model_name': 'efficientnetv2',
            'map_location': device
        }
    )
    
    metrics = AnalysisPipeline.run_analysis(
        df_path=df_path,
        signal_processing_model=efficient_netV2,
        classification_model=bert_classifier
    )
    
    with open(os.path.join(output_folder, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    df = convert_to_df(df_path)
    save_to_csv(df, os.path.join(output_folder, 'df.csv'))
    save_to_json(metrics, os.path.join(output_folder, 'metrics.json'))
    

    
        
if __name__ == '__main__':
    main()

