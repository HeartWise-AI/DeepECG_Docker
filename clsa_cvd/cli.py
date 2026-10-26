"""Label-free table inference; explicit per-row errors, no diagnostic pipeline scaling."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .preprocessing import load_waveform,validate_clinical


def run_table(df,model,source_format,batch_size=32):
    required={'npy_path','age','female'}
    if not required.issubset(df):raise ValueError('Input requires npy_path, age, female; hypertension is optional for A+')
    if batch_size<1:raise ValueError('batch_size must be positive')
    # Position-based iteration preserves arbitrary/duplicate input indices and row ordering.
    result=pd.DataFrame(index=np.arange(len(df)))
    for c in ('p_ecg','ecg_logit_uncal','risk_Aplus','risk_ecg_age_sex'):result[c]=np.nan
    result['ecg_ok']=False;result['error']=''
    for start in range(0,len(df),batch_size):
        xs=[];clinical=[];positions=[]
        for i in range(start,min(start+batch_size,len(df))):
            r=df.iloc[i]
            try:
                c=validate_clinical(r.age,r.female,r.hypertension if 'hypertension' in df else None)
                xs.append(load_waveform(r.npy_path,source_format));clinical.append(c);positions.append(i)
            except (ValueError,TypeError,OSError,EOFError):
                # Keep details local; no potentially identifiable file paths in logs.
                result.loc[i,'error']='Invalid or missing waveform/clinical input'
        if not xs:continue
        clinical=np.asarray(clinical,dtype=float)
        # Inference/checkpoint errors are fatal, not misreported as invalid patient data.
        output=model.predict(np.stack(xs),clinical[:,0],clinical[:,1],clinical[:,2] if 'hypertension' in df else None,batch_size=batch_size)
        for name,values in output.items():result.loc[positions,name]=values
        result.loc[positions,'ecg_ok']=True
    answer=df.copy()
    for c in result:answer[c]=result[c].to_numpy()
    return answer


def main(argv=None):
    p=argparse.ArgumentParser(description='CLSA incident-F2 CVD; A+ needs HTN. External age/sex output is unrecalibrated.')
    p.add_argument('--task',choices=['clsa_cvd'],default='clsa_cvd')
    p.add_argument('--config',required=True,help='JSON with weights_dir, base_checkpoint, params_path, format, device, batch_size')
    p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(argv)
    config_path=Path(a.config).resolve()
    with config_path.open() as f:c=json.load(f)
    if c.get('format') not in ('clsa','muse','mv250'):p.error('config format must be clsa, muse, or mv250')
    for name in ('weights_dir','base_checkpoint','params_path'):
        if name not in c:p.error(f'missing config {name}')
        value=Path(c[name]);c[name]=str(value if value.is_absolute() else config_path.parent/value)
    from .model import CLSACVDModel
    df=pd.read_parquet(a.input) if Path(a.input).suffix=='.parquet' else pd.read_csv(a.input)
    m=CLSACVDModel(c['weights_dir'],c['base_checkpoint'],c['params_path'],c.get('device','cpu'),c.get('precision','float32'))
    ans=run_table(df,m,c['format'],c.get('batch_size',32))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    ans.to_parquet(out,index=False) if out.suffix=='.parquet' else ans.to_csv(out,index=False)
    print(json.dumps({'rows':len(ans),'scored':int(ans.ecg_ok.sum()),'failed':int((~ans.ecg_ok).sum())}))
    return 0 if ans.ecg_ok.all() else 2

if __name__=='__main__':raise SystemExit(main())
