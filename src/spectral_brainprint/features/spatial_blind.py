"""Fixed single-record descriptors; no peak fitting or dataset-learned reference."""
import numpy as np
from spectral_brainprint.constants import CHANNELS
from spectral_brainprint.spectra.welch import welch_psd, band_power

ROI=[CHANNELS.index(c) for c in ["O1","O2","P3","P4"]]
UP=np.triu_indices(18);OFF=np.triu_indices(18,k=1)
WEIGHTS=np.where(UP[0]==UP[1],1.,np.sqrt(2.))


def upper(a,weighted=False):
    x=a[...,UP[0],UP[1]]
    return x*WEIGHTS if weighted else x


def covariance_blocks(eeg):
    x=np.asarray(eeg,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(1200,18) or not np.isfinite(x).all():raise ValueError("Expected finite N x1200x18")
    x=x-x.mean(axis=1,keepdims=True)
    empirical=(x.transpose(0,2,1)@x)/1200
    mu=np.trace(empirical,axis1=1,axis2=2)/18
    if np.any(mu<=0):raise ValueError("Record has zero total variance; not an unknown-identity decision")
    alpha=np.mean(empirical**2,axis=(1,2));numerator=alpha+mu**2
    denominator=1201*(alpha-mu**2/18)
    shrink=np.ones(len(x));np.divide(numerator,denominator,out=shrink,where=denominator>0)
    shrink=np.clip(shrink,0,1)
    cov=(1-shrink[:,None,None])*empirical
    cov[:,np.arange(18),np.arange(18)]+=shrink[:,None]*mu[:,None]
    eigenvalues,vectors=np.linalg.eigh(cov)
    floor=1e-8*mu[:,None];clipped=eigenvalues<floor
    regularized=np.maximum(eigenvalues,floor)
    cov=np.einsum("nik,nk,njk->nij",vectors,regularized,vectors)
    trace=regularized.sum(axis=1)
    cov_shape=cov/trace[:,None,None]
    logcov=np.einsum("nik,nk,njk->nij",vectors,np.log(regularized),vectors)
    logshape=logcov-np.log(trace)[:,None,None]*np.eye(18)[None,:,:]
    diagonal=np.diagonal(empirical,axis1=1,axis2=2)
    denominator_corr=np.sqrt(diagonal[:,:,None]*diagonal[:,None,:])
    corr=np.divide(empirical,denominator_corr,out=np.zeros_like(empirical),where=denominator_corr>0)
    corr=np.clip(corr,-1,1)
    blocks={"cov_oas_upper":upper(cov),"cov_trace_upper":upper(cov_shape),"corr_pearson_off":corr[:,OFF[0],OFF[1]],
            "logcov_upper":upper(logcov,True),"logcov_trace_upper":upper(logshape,True)}
    quality={"shrinkage":shrink,"trace":trace,"eigenvalues_before_floor":eigenvalues,"floor_clipped_eigenvalues":clipped,
             "zero_variance_channels":diagonal<=0}
    return blocks,cov,corr,quality


def local_contrast(p,f,centers):
    values=[]
    for center in centers:
        main=band_power(p,f,center-.5,center+.5)
        neighbours=(band_power(p,f,center-2.5,center-1.5)+band_power(p,f,center+1.5,center+2.5))/2
        values.append(np.log10(np.maximum(main,1e-300))-np.log10(np.maximum(neighbours,1e-300)))
    return np.stack(values,axis=-1)


def blind_blocks(eeg):
    f,p=welch_psd(eeg)
    roi=p[:,:,ROI]
    powers=np.stack([band_power(roi,f,left,left+1) for left in range(4,40)],axis=-1)
    total=band_power(roi,f,4,40)
    if np.any(total<=0):raise ValueError("Zero ROI4-40Hz power")
    log=np.log10(np.maximum(powers,1e-300));shape=powers/total[:,:,None]
    contrast=local_contrast(roi,f,range(6,36))
    fundamentals=np.arange(4,19)
    harmonic=np.minimum(local_contrast(roi,f,fundamentals),local_contrast(roi,f,2*fundamentals))
    blocks={"blind_logpower":log.reshape(len(eeg),144),"blind_shape":shape.reshape(len(eeg),144),
            "blind_local_contrast":contrast.reshape(len(eeg),120),"blind_harmonic_pattern":harmonic.reshape(len(eeg),60)}
    return blocks,f,p


def dictionary():
    rows=[]
    for block in ["cov_oas_upper","cov_trace_upper","logcov_upper","logcov_trace_upper"]:
        for column,(i,j) in enumerate(zip(*UP)):
            rows.append({"block":block,"column":column,"name":f"{block}__{CHANNELS[i]}__{CHANNELS[j]}"})
    for column,(i,j) in enumerate(zip(*OFF)):
        rows.append({"block":"corr_pearson_off","column":column,"name":f"corr__{CHANNELS[i]}__{CHANNELS[j]}"})
    for block,values in [("blind_logpower",range(4,40)),("blind_shape",range(4,40)),
                         ("blind_local_contrast",range(6,36)),("blind_harmonic_pattern",range(4,19))]:
        column=0
        for channel in ["O1","O2","P3","P4"]:
            for value in values:
                label=f"{value}_{value+1}Hz" if block in ["blind_logpower","blind_shape"] else f"{value}Hz"
                rows.append({"block":block,"column":column,"name":f"{block}__{channel}__{label}"});column+=1
    return rows
