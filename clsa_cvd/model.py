"""Five-fold CVD predictor. Dependencies are loaded only when a model is requested."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit


def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p))


def outputs_from_folds(fold_probabilities,params,age,female,hypertension=None):
    """Preserve the reference clipping convention and calibrated/uncalibrated paths."""
    raw=np.asarray(fold_probabilities)
    if raw.ndim!=2 or raw.shape[0]!=5 or not np.isfinite(raw).all() or ((raw<0)|(raw>1)).any():
        raise ValueError('Expected five finite fold-probability vectors in [0,1]')
    n=raw.shape[1]
    def vector(v,label):
        a=np.asarray(v,dtype=float)
        if a.shape!=(n,) or not np.isfinite(a).all():
            raise ValueError(f'{label} must be a finite vector with one value per ECG')
        return a
    age=vector(age,'age');female=vector(female,'female')
    if ((age<18)|(age>100)).any() or not np.isin(female,[0,1]).all():
        raise ValueError('Invalid age or female values')
    cal=[]
    for k in range(5):
        c=params['per_fold_platt'][str(k)]
        z=c['a']*logit(raw[k])+c['b']
        cal.append(1/(1+np.exp(-z)))
    p=np.mean(cal,axis=0)
    uncal=logit(np.mean(raw,axis=0))
    ext=params['model_ecg_age_sex_external']
    result={'p_ecg':p,'ecg_logit_uncal':uncal,'risk_ecg_age_sex':expit(ext['intercept']+ext['b_ecg']*uncal+ext['b_age']*age+ext['b_female']*female)}
    if hypertension is not None:
        htn=vector(hypertension,'hypertension')
        if not np.isin(htn,[0,1]).all():raise ValueError('hypertension must be binary')
        c=params['model_A_plus']
        result['risk_Aplus']=expit(c['intercept']+c['b_ecg']*logit(p)+c['b_age']*age+c['b_female']*female+c['b_htn']*htn)
    return result


class CLSACVDModel:
    """Load all folds once; a single ensemble pass supplies all risk outputs."""
    def __init__(self,weights_dir,base_checkpoint,params_path,device='cpu',precision='float32'):
        import torch
        from safetensors.torch import load_file
        from fairseq_signals.utils import checkpoint_utils
        self.torch=torch
        # TF32 convolution/matmul kernels can introduce batch-size-dependent drift.
        # This dedicated task uses IEEE float32 to make single/batch checks reproducible.
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32=False
        self.device=torch.device(device)
        if precision not in ('float32','bfloat16'):
            raise ValueError('precision must be float32 or bfloat16')
        if precision=='bfloat16' and self.device.type!='cuda':
            raise ValueError('Historical bfloat16 compatibility mode requires CUDA')
        self.precision=precision
        with Path(params_path).open() as f:self.params=json.load(f)
        self.models=[];self.checkpoint_audit=[]
        for k in range(5):
            models,_,_=checkpoint_utils.load_model_and_task(str(base_checkpoint))
            enc=models[0] if isinstance(models,list) else models
            for name in ('quantizer','project_q','final_proj','target_glu'):
                if hasattr(enc,name):setattr(enc,name,None)
            class Net(torch.nn.Module):
                def __init__(self,encoder):
                    super().__init__();self.enc=encoder
                    self.head=torch.nn.Sequential(torch.nn.Dropout(.3),torch.nn.Linear(768,1))
                def forward(self,x):
                    return self.head(self.enc(source=x,mask=False,features_only=True)['x'].mean(1)).squeeze(1)
            net=Net(enc)
            state=load_file(str(Path(weights_dir)/f'deepecg_ssl_clsa_cvd_fold{k}.safetensors'))
            # No silently missing encoder/head parameters; no unexpected state allowed.
            net.load_state_dict(state,strict=True)
            self.checkpoint_audit.append({'fold':k,'strict_load':True,'keys':len(state)})
            self.models.append(net.to(self.device).eval())

    def predict(self,ecg_mv,age,female,hypertension=None,batch_size=32):
        x=np.asarray(ecg_mv,dtype=np.float32)
        if x.ndim!=3 or x.shape[1:]!=(12,2500) or not np.isfinite(x).all():
            raise ValueError('Expected finite (N,12,2500) mV waveforms')
        if not len(x) or batch_size<1:
            raise ValueError('Nonempty input and positive batch size required')
        folds=[]
        with self.torch.no_grad():
            for m in self.models:
                values=[]
                for i in range(0,len(x),batch_size):
                    xx=self.torch.as_tensor(np.array(x[i:i+batch_size],copy=True),device=self.device)
                    with self.torch.autocast(self.device.type,dtype=self.torch.bfloat16,enabled=self.precision=='bfloat16'):
                        values.append(self.torch.sigmoid(m(xx)).float().cpu().numpy())
                folds.append(np.concatenate(values))
        return outputs_from_folds(np.stack(folds),self.params,age,female,hypertension)
