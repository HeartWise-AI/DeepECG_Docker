from pathlib import Path
import hashlib
import numpy as np
import pytest
from container_validate_cvd import verify_artifacts,synthetic_inputs


def test_all_hashes_required_and_renamed_mounts_work(tmp_path):
    weights=tmp_path/'weights';weights.mkdir();base=tmp_path/'encoder.pt';base.write_bytes(b'base');params=tmp_path/'params.json';params.write_bytes(b'params')
    expected={'deepecg-ssl-v2-wcr-amp-preserved-encoder.pt':hashlib.sha256(base.read_bytes()).hexdigest(),'model_params.json':hashlib.sha256(params.read_bytes()).hexdigest()}
    for k in range(5):
        p=weights/f'deepecg_ssl_clsa_cvd_fold{k}.safetensors';p.write_bytes(bytes([k]));expected[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    cfg={'weights_dir':str(weights),'base_checkpoint':str(base),'params_path':str(params)}
    assert verify_artifacts(cfg,expected)==expected
    params.write_bytes(b'changed')
    with pytest.raises(ValueError):verify_artifacts(cfg,expected)
    with pytest.raises(ValueError):verify_artifacts(cfg,{})


def test_synthetic_input_generator_is_deterministic_and_in_mv():
    a=synthetic_inputs();np.testing.assert_array_equal(a,synthetic_inputs())
    assert a.shape==(3,12,2500) and a.dtype==np.float32 and np.isfinite(a).all()
    assert float(np.max(abs(a)))<1


def test_runner_rejects_output_overlapping_model_inputs_before_docker(tmp_path):
    import subprocess
    weights=tmp_path/'weights';weights.mkdir();encoder=tmp_path/'encoder.pt';encoder.write_bytes(b'test')
    out=weights/'reports'
    r=subprocess.run(['bash',str(Path(__file__).parents[1]/'validate_cvd_container.bash'),str(weights),str(encoder),str(out)],capture_output=True,text=True)
    assert r.returncode!=0 and 'does not overlap model inputs' in r.stderr
    assert not out.exists()


def test_runner_preserves_previous_reports_and_requires_empty_output(tmp_path):
    import subprocess
    weights=tmp_path/'weights';weights.mkdir();encoder=tmp_path/'encoder.pt';encoder.write_bytes(b'test')
    out=tmp_path/'old_run';out.mkdir();report=out/'container_validation.json';report.write_text('{"previous":true}')
    r=subprocess.run(['bash',str(Path(__file__).parents[1]/'validate_cvd_container.bash'),str(weights),str(encoder),str(out)],capture_output=True,text=True)
    assert r.returncode!=0 and 'empty output directory' in r.stderr
    assert report.read_text()=='{"previous":true}'
