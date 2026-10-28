"""Exercise runtime model selection and failure status with synthetic inference."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as pipeline
from utils.files_handler import ECGFileHandler
from utils.model_manifest import ModelPreparationError


@pytest.mark.parametrize('failure', [None, ModelPreparationError('Access denied'), ValueError('Invalid model weights')])
def test_only_selected_models_load_and_requested_failure_exits(tmp_path, monkeypatch, capsys, failure):
    preprocessed = tmp_path / 'preprocessed'
    preprocessed.mkdir()
    for name in ('one', 'two'):
        ECGFileHandler.save_ecg_signal(np.ones((2500, 12)), str(preprocessed / f'{name}.base64'))
    pd.DataFrame({'afib_5y': [0, 1], 'afib_ecg_file_name': ['one.npy', 'two.xml']}).to_csv(tmp_path / 'input.csv', index=False)
    selected = []

    class SyntheticModel:
        model = torch.nn.Identity()
        def __call__(self, signals):
            return torch.full((len(signals), 1), 0.5)

    def factory(config):
        selected.append(config['model_name'])
        if failure:
            raise failure
        return SyntheticModel()

    monkeypatch.setattr('utils.analysis_pipeline.HeartWiseModelFactory.create_model', factory)
    args = SimpleNamespace(
        mode='analysis', data_path=str(tmp_path / 'input.csv'), ecg_signals_path=str(tmp_path),
        output_folder=str(tmp_path / 'outputs'), preprocessing_folder=str(preprocessed),
        use_wcr=False, use_efficientnet=True, batch_size=2,
        signal_processing_device='cpu', diagnosis_classifier_device='cpu',
        hugging_face_api_key_path=str(tmp_path / 'api_key.json'),
    )
    if failure:
        with pytest.raises(SystemExit) as caught:
            pipeline.main(args)
        assert caught.value.code == 1
        assert str(failure) in capsys.readouterr().out
    else:
        pipeline.main(args)
        assert len(list((tmp_path / 'outputs').glob('*probabilities.csv'))) == 1
    assert selected == ['efficientnetv2_afib_5y']  # Neither WCR nor BERT is loaded.
