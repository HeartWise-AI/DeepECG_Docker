"""Verify the mounted artifacts and synthetic goldens inside the candidate image.
Image identity/build provenance must be captured by the external Docker runner.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from clsa_cvd.preprocessing import prepare_waveform

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify_artifacts(config,expected):
    files=[('deepecg-ssl-v2-wcr-amp-preserved-encoder.pt',Path(config['base_checkpoint'])),('model_params.json',Path(config['params_path']))]+[(f'deepecg_ssl_clsa_cvd_fold{k}.safetensors',Path(config['weights_dir'])/f'deepecg_ssl_clsa_cvd_fold{k}.safetensors') for k in range(5)]
    if len(expected)!=7:raise ValueError('Seven pinned artifact hashes required')
    result={}
    for name,p in files:
        digest=sha(p)
        if expected.get(name)!=digest:raise ValueError('Artifact hash mismatch')
        result[name]=digest
    return result


def synthetic_inputs():
    t=np.arange(5000)/500
    raw=np.stack([np.stack([120*np.sin(2*np.pi*(1.1+j*.1)*t+i*.15)+30*np.sin(2*np.pi*17*t) for i in range(12)]) for j in range(3)])
    return np.stack([prepare_waveform(a,'clsa') for a in raw])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--artifacts-only',action='store_true');a=p.parse_args(argv)
    cp=Path(a.config).resolve();c=json.loads(cp.read_text())
    for k in ['weights_dir','base_checkpoint','params_path']:
        path=Path(c[k]);c[k]=str(path if path.is_absolute() else cp.parent/path)
    expected=json.loads((HERE/'clsa_cvd/artifacts.json').read_text())['sha256_by_filename']
    hashes=verify_artifacts(c,expected)
    report={'artifact_hashes_match':True,'artifacts':hashes,'scope':'Artifact hashes only','model_execution_verified':False}
    if not a.artifacts_only:
        import torch
        from clsa_cvd.model import CLSACVDModel
        # The public synthetic goldens were independently generated in IEEE float32.
        if c.get('precision','float32')!='float32':raise ValueError('Synthetic validation config must request float32')
        golden=json.loads((HERE/'tests/fixtures/clsa_cvd_synthetic.json').read_text());clinical=golden['clinical']
        m=CLSACVDModel(c['weights_dir'],c['base_checkpoint'],c['params_path'],c.get('device','cuda:0'),'float32')
        x=synthetic_inputs();out=m.predict(x,clinical['age'],clinical['female'],clinical['hypertension'],batch_size=2)
        checks={}
        for key,ref in [('p_ecg','canonical_p_ecg'),('risk_Aplus','full_precision_Aplus'),('risk_ecg_age_sex','external')]:
            checks[key]=float(np.max(np.abs(out[key]-golden[ref])))
        if max(checks.values())>1e-6:raise ValueError('Synthetic reference mismatch')
        one=m.predict(x[:1],clinical['age'][:1],clinical['female'][:1],clinical['hypertension'][:1],batch_size=1)
        batch_errors={k:float(np.max(np.abs(one[k]-out[k][:1]))) for k in out}
        if max(batch_errors.values())>1e-5:raise ValueError('Single/batch mismatch')
        report.update(scope='Strict checkpoints plus synthetic single/batch inference; no participant data used',model_execution_verified=True,n=3,max_abs_reference_errors=checks,single_batch_errors=batch_errors,checkpoint_audit=m.checkpoint_audit,torch_version=torch.__version__,cuda_version=torch.version.cuda,device=str(m.device))
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'artifact_hashes_match':True,'model_execution_verified':report['model_execution_verified']}))
    return 0
if __name__=='__main__':raise SystemExit(main())
