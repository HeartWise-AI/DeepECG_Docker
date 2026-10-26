import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.signal import resample_poly
from clsa_cvd.preprocessing import prepare_waveform,validate_clinical,LEADS
from clsa_cvd.model import outputs_from_folds
from clsa_cvd.cli import run_table

PARAMS=json.loads((Path(__file__).parents[1]/'clsa_cvd/model_params.json').read_text())


def test_raw_and_mv_inputs_have_identical_units_and_orientation():
    t=np.arange(5000)/500
    adc=np.stack([100*(i+1)*np.sin(2*np.pi*(i+1)*t) for i in range(12)])
    expected=resample_poly(adc*.005,1,2,axis=-1).astype(np.float32)
    np.testing.assert_array_equal(prepare_waveform(adc,'clsa'),expected)
    np.testing.assert_array_equal(prepare_waveform(adc.T,'clsa'),expected)
    np.testing.assert_array_equal(prepare_waveform(expected,'mv250'),expected)
    np.testing.assert_array_equal(prepare_waveform(np.ones((2500,12))*100,'muse'),np.full((12,2500),.488,dtype=np.float32))
    with pytest.raises(ValueError):prepare_waveform(adc,'mv250')
    with pytest.raises(ValueError):prepare_waveform(adc,'clsa',LEADS[::-1])


@pytest.mark.parametrize('x',[np.zeros((12,2499)),np.full((12,2500),np.nan),np.full((12,2500),np.inf),np.zeros((11,2500)),np.zeros((2500,))])
def test_invalid_waveforms_rejected(x):
    with pytest.raises(ValueError):prepare_waveform(x,'mv250')


@pytest.mark.parametrize('age,female,htn',[(np.nan,0,0),(17,0,0),(65,np.nan,0),(65,2,0),(65,1,np.nan),(65,1,3)])
def test_invalid_clinical_rejected(age,female,htn):
    with pytest.raises(ValueError):validate_clinical(age,female,htn)


def test_calibration_order_and_external_path_against_hand_equations():
    raw=np.array([[.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6]])
    out=outputs_from_folds(raw,PARAMS,[60,70],[0,1],[1,0])
    def lg(p):return np.log(p/(1-p))
    expected=np.mean([1/(1+np.exp(-(PARAMS['per_fold_platt'][str(i)]['a']*lg(raw[i])+PARAMS['per_fold_platt'][str(i)]['b']))) for i in range(5)],axis=0)
    np.testing.assert_allclose(out['p_ecg'],expected,atol=1e-12)
    c=PARAMS['model_A_plus'];z=c['intercept']+c['b_ecg']*lg(expected)+c['b_age']*np.array([60,70])+c['b_female']*np.array([0,1])+c['b_htn']*np.array([1,0])
    np.testing.assert_allclose(out['risk_Aplus'],1/(1+np.exp(-z)))
    np.testing.assert_allclose(out['ecg_logit_uncal'],lg(raw.mean(0)))
    assert not np.allclose(out['ecg_logit_uncal'],lg(expected))
    a=outputs_from_folds(raw[:,::-1],PARAMS,[70,60],[1,0],[0,1])
    for k in out:np.testing.assert_allclose(a[k][::-1],out[k])
    b=outputs_from_folds(raw[:,:1],PARAMS,[60],[0])
    assert 'risk_Aplus' not in b
    np.testing.assert_allclose(b['p_ecg'],out['p_ecg'][:1])


def test_table_handles_mixed_validity_without_changing_order_or_requiring_labels(tmp_path):
    f=tmp_path/'good.npy';np.save(f,np.zeros((12,2500)))
    class Dummy:
        def predict(self,x,age,female,hypertension=None,batch_size=32):
            return outputs_from_folds(np.tile(np.arange(1,6)[:,None]/10,(1,len(x))),PARAMS,age,female,hypertension)
    d=pd.DataFrame({'npy_path':[str(f),str(tmp_path/'missing.npy'),str(f),str(f)],'age':[60,60,70,70],'female':[0,1,1,1],'hypertension':[1,0,0,np.nan]},index=[9,9,2,1])
    a=run_table(d,Dummy(),'mv250',batch_size=1);b=run_table(d,Dummy(),'mv250',batch_size=4)
    pd.testing.assert_frame_equal(a,b)
    assert a.index.tolist()==[9,9,2,1]
    assert a.ecg_ok.tolist()==[True,False,True,False]
    assert a.iloc[[1,3]].p_ecg.isna().all()
    assert 'missing.npy' not in ' '.join(a.error)


def test_protected_synthetic_model_golden():
    """Optional real-weight regression: no download and no participant inputs."""
    import os
    from clsa_cvd.model import CLSACVDModel
    bundle=os.environ.get('CLSA_CVD_TEST_BUNDLE')
    base=os.environ.get('CLSA_CVD_TEST_ENCODER')
    if not bundle or not base:
        pytest.skip('Set CLSA_CVD_TEST_BUNDLE and CLSA_CVD_TEST_ENCODER to approved local artifacts')
    golden=json.loads((Path(__file__).parent/'fixtures/clsa_cvd_synthetic.json').read_text())
    t=np.arange(5000)/500
    raw=np.stack([np.stack([120*np.sin(2*np.pi*(1.1+j*.1)*t+i*.15)+30*np.sin(2*np.pi*17*t) for i in range(12)]) for j in range(3)])
    x=np.stack([prepare_waveform(a,'clsa') for a in raw])
    m=CLSACVDModel(Path(bundle)/'weights',base,Path(__file__).parents[1]/'clsa_cvd/model_params.json',device=os.environ.get('CLSA_CVD_TEST_DEVICE','cuda:0'))
    c=golden['clinical'];r=m.predict(x,c['age'],c['female'],c['hypertension'],batch_size=2)
    np.testing.assert_allclose(r['p_ecg'],golden['canonical_p_ecg'],atol=1e-6,rtol=0)
    np.testing.assert_allclose(r['risk_Aplus'],golden['full_precision_Aplus'],atol=1e-6,rtol=0)
    np.testing.assert_allclose(r['risk_ecg_age_sex'],golden['external'],atol=1e-6,rtol=0)
    assert all(k['strict_load'] for k in m.checkpoint_audit)
