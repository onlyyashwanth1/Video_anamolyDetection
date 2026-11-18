#!/usr/bin/env python3
"""Shared bank-free nonlinear image/caption fusion search on saved evidence.

Despite the historical filename this evaluates exactly the same recipe on both
datasets and selects jointly on the existing source-group validation split.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from search_xd_configs import load_run, evaluate, DEFAULT_CONFIG, weighted_metrics
from search_shared_configs import RUNS, postprocess


def fuse(image_margin, text_margin, temperature, image_weight, method, text_temperature=None):
    """Deterministic gates based only on evidence, never dataset identity."""
    i, t = expit(image_margin / temperature), expit(text_margin / (text_temperature or temperature))
    w = image_weight
    if method == 'linear':
        return w * i + (1-w) * t
    if method == 'geometric':
        return np.exp(w * np.log(np.maximum(i, 1e-15)) + (1-w) * np.log(np.maximum(t, 1e-15)))
    if method == 'harmonic':
        return 1 / (w / np.maximum(i, 1e-15) + (1-w) / np.maximum(t, 1e-15))
    if method == 'union':
        return 1 - (1-i) ** w * (1-t) ** (1-w)
    if method == 'caption_confidence':
        tw = (1-w) * np.abs(2*t-1)
        return (1-tw) * i + tw * t
    if method == 'image_confidence':
        iw = w * np.abs(2*i-1)
        return iw * i + (1-iw) * t
    if method == 'agreement_gate':
        tw = (1-w) * (1-np.abs(i-t))
        return (1-tw) * i + tw * t
    if method == 'positive_image_gate':
        tw = (1-w) * i
        return (1-tw) * i + tw * t
    if method == 'maximum':
        return w * i + (1-w) * np.maximum(i,t)
    if method == 'minimum':
        return w * i + (1-w) * np.minimum(i,t)
    raise ValueError(method)


def pool(cache, values, method):
    ids, counts = cache['ids'], cache['counts']
    gathered = np.where(ids >= 0, values[np.maximum(ids,0)], np.nan)
    mean = np.nansum(gathered,axis=1) / counts
    if method == 'mean': return mean
    if method == 'max': return np.nanmax(gathered,axis=1)
    if method == 'mean_max_half': return .5*(mean + np.nanmax(gathered,axis=1))
    raise ValueError(method)


def scores(cache, config):
    f = cache['features']
    values = fuse(f[:,0],f[:,1],config['temperature'],config['image_weight'],config['fusion'],config.get('text_temperature'))
    s = pool(cache,values,config['pooling'])
    return postprocess(cache,s,config.get('normalization','none'),config.get('strength',0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/results/ucf_config_search_20261026')
    p.add_argument('--scope',choices=['validation','all_benchmark'],default='validation')
    p.add_argument('--image-variant',default=None,help='Replace image evidence with one shared cached prompt variant')
    p.add_argument('--image-reducer',default='logmeanexp')
    p.add_argument('--separate-temperatures',action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic(); caches={n:load_run(path) for n,path in RUNS.items()}
    baseline={n:{s:evaluate(c,DEFAULT_CONFIG,s) for s in ('validation','evaluation','all_benchmark')} for n,c in caches.items()}
    if args.image_variant:
        from search_shared_prompts import margins
        feature_dir=ROOT/'outputs/results/shared_prompt_features_20261026'
        metadata=json.loads((feature_dir/'prompt_features.json').read_text())
        for name,short in [('UCF-Crime','ucf'),('XD-Violence','xd')]:
            with np.load(feature_dir/f'{short}_image_prompt_features.npz') as z:
                new=margins(z['image_similarities'],metadata['variants'][args.image_variant],metadata['prompt_ids'],args.image_reducer)
            caches[name]['features']=caches[name]['features'].copy()
            caches[name]['features'][:,0]=new
    u0=baseline['UCF-Crime'][args.scope]['roc_auc'];x0=baseline['XD-Violence'][args.scope]['average_precision']
    rows=[];seen=set()
    def score(config,stage):
        key=json.dumps(config,sort_keys=True)
        if key in seen:return
        seen.add(key); metrics={}
        for name,c in caches.items():
            s=scores(c,config);m=c['masks'][args.scope]
            metrics[name]=weighted_metrics(s[m],c['positive'][m],c['negative'][m])
        gains=(metrics['UCF-Crime']['roc_auc']/u0-1,metrics['XD-Violence']['average_precision']/x0-1)
        rows.append(dict(config=config,selection=metrics,joint_gain=float(np.mean(gains)),minimum_gain=float(min(gains)),stage=stage))
        if len(rows)%100==0:print(f'{len(rows)} nonlinear fusion candidates in {time.monotonic()-start:.1f}s',flush=True)
    for temp,weight,method in itertools.product([.01,.02,.025,.05,.1,.2,.5],[.25,.5,.65,.75,.8125,.875],['linear','geometric','harmonic','union','caption_confidence','image_confidence','agreement_gate','positive_image_gate','maximum','minimum']):
        score(dict(temperature=temp,image_weight=weight,fusion=method,pooling='mean'),'nonlinear_fusion')
    if args.separate_temperatures:
        for ti,tt,w in itertools.product([.005,.01,.02,.04,.08,.16,.32,1.],[.01,.025,.05,.1,.2,.5,1.],[.25,.5,.65,.75,.8125,.875,.9375]):
            score(dict(temperature=ti,text_temperature=tt,image_weight=w,fusion='linear',pooling='mean'),'independent_temperatures')
        seeds=sorted(rows,key=lambda r:r['joint_gain'],reverse=True)[:15]
        for row,method in itertools.product(seeds,['geometric','harmonic','union','caption_confidence','image_confidence','agreement_gate','positive_image_gate','maximum','minimum']):
            score({**row['config'],'fusion':method},'independent_temperature_nonlinear_fusion')
    seeds=sorted(rows,key=lambda r:r['joint_gain'],reverse=True)[:15]+sorted(rows,key=lambda r:r['minimum_gain'],reverse=True)[:10]+sorted(rows,key=lambda r:r['selection']['UCF-Crime']['roc_auc'],reverse=True)[:10]
    for row,method in itertools.product(seeds,['mean','max','mean_max_half']):
        score({**row['config'],'pooling':method},'pooling')
    seeds=sorted(rows,key=lambda r:r['joint_gain'],reverse=True)[:10]+sorted(rows,key=lambda r:r['minimum_gain'],reverse=True)[:10]
    for row,mode,strength in itertools.product(seeds,['center','standardize'],[.0125,.025,.05,.075,.1,.15,.25]):
        score({**row['config'],'normalization':mode,'strength':strength},'weak_normalization')
    ranked=sorted(rows,key=lambda r:r['joint_gain'],reverse=True)
    eligible=[r for r in ranked if r['minimum_gain']>=0]
    selected=(eligible or ranked)[0]
    shortlist=ranked[:20]+sorted(rows,key=lambda r:r['minimum_gain'],reverse=True)[:10]
    if selected not in shortlist:shortlist.append(selected)
    for row in shortlist:
        for part in ('validation','evaluation','all_benchmark'):
            row[part]={}
            for n,c in caches.items():
                s=scores(c,row['config']);m=c['masks'][part]
                row[part][n]=weighted_metrics(s[m],c['positive'][m],c['negative'][m])
    result=dict(selection_scope=args.scope,image_variant=args.image_variant,image_reducer=args.image_reducer,baseline=baseline,candidate_count=len(rows),selected=selected,shortlist=shortlist,candidates=rows,elapsed_seconds=time.monotonic()-start,
        limitation='Reused benchmark exploration. Recipe identical across datasets; no normal bank, no dataset-specific normalization or learned state. Only unlabeled within-video scores used by optional normalization.')
    (args.output/'search.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Shared nonlinear image/caption fusion search','','| Recipe | UCF validation AUC | XD validation AP | UCF all AUC | XD all AP |','|---|---:|---:|---:|---:|']
    for r in shortlist[:10]:
        lines.append('| '+json.dumps(r['config'])+' | '+' | '.join(f'{r[p][n][m]*100:.3f}' for p,n,m in [('validation','UCF-Crime','roc_auc'),('validation','XD-Violence','average_precision'),('all_benchmark','UCF-Crime','roc_auc'),('all_benchmark','XD-Violence','average_precision')])+' |')
    lines+=['',result['limitation']]
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'selected':selected,'count':len(rows),'seconds':time.monotonic()-start},indent=2),flush=True)

if __name__=='__main__':main()
