#!/usr/bin/env python3
"""Joint shared-prompt search from frozen CPU CLIP features, without bank state."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from search_shared_configs import RUNS, assess
from search_xd_configs import load_run, evaluate, DEFAULT_CONFIG, normalize_config, POOLING


def margins(similarities, definition, ids, reducer):
    a=similarities[:,[ids[p] for p in definition['anomalous']]]
    n=similarities[:,[ids[p] for p in definition['normal']]]
    def reduce(x):
        if reducer=='max':return np.max(x,axis=1)
        if reducer=='mean':return np.mean(x,axis=1)
        if reducer=='top2':return np.partition(x,-2,axis=1)[:,-2:].mean(axis=1)
        if reducer=='logmeanexp':return .05*(logsumexp(x/.05,axis=1)-np.log(x.shape[1]))
        raise ValueError(reducer)
    return reduce(a)-reduce(n)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-mode',choices=['image','validation','full'],default='image')
    p.add_argument('--selection-scope',choices=['validation','all_benchmark'],default='validation')
    p.add_argument('--features',type=Path,default=ROOT/'outputs/results/shared_prompt_features_20261026')
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args();start=time.monotonic()
    if args.feature_mode=='validation' and args.selection_scope!='validation':raise ValueError('Incomplete inference cannot score full benchmark')
    output=args.output or ROOT/f'outputs/results/shared_prompt_search_{args.feature_mode}_{args.selection_scope}_20261026'
    output.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((args.features/'prompt_features.json').read_text());defs=metadata['variants'];ids=metadata['prompt_ids']
    caches={name:load_run(path) for name,path in RUNS.items()};source={}
    for dataset,short in [('UCF-Crime','ucf'),('XD-Violence','xd')]:
        suffix={'image':'image_prompt_features','validation':'validation_prompt_features','full':'prompt_features'}[args.feature_mode]
        with np.load(args.features/f'{short}_{suffix}.npz') as z:source[dataset]={k:z[k] for k in z.files}
        src=source[dataset];c=caches[dataset]
        if list(src['names'])!=c['names'] or len(src['start_frame'])!=len(c['features']):raise ValueError('Feature ordering differs')
        if not np.array_equal(np.diff(src['offsets']),np.bincount(src['video_index'],minlength=len(c['names']))):raise ValueError('Invalid offsets')
    baseline={name:{partition:evaluate(c,DEFAULT_CONFIG,partition) for partition in ['validation','evaluation','all_benchmark']} for name,c in caches.items()}
    b=baseline;scope=args.selection_scope
    u0=b['UCF-Crime'][scope]['roc_auc'];x0=b['XD-Violence'][scope]['average_precision']
    features={};rows=[];seen=set()
    def cache_for(dataset,iv,tv,image_reducer,text_reducer):
        key=(dataset,iv,tv,image_reducer,text_reducer)
        if key not in features:
            src=source[dataset];cache=caches[dataset];f=cache['features'].copy()
            f[:,0]=margins(src['image_similarities'],defs[iv],ids,image_reducer)
            if args.feature_mode=='image':
                if tv!='expanded26':raise ValueError('Only original text margins exist in image mode')
            elif 'text_similarities' in src:
                f[:,1]=margins(np.nan_to_num(src['text_similarities']),defs[tv],ids,text_reducer)
            else:
                vi=list(src['variant_names']).index(tv)
                f[:,1]=np.nan_to_num(src['action_text_margin'][:,vi])
                if text_reducer!='max':raise ValueError('Reducer needs complete text similarities')
            features[key]={**cache,'features':f}
        return features[key]
    def score(cfg,stage):
        cfg={**cfg,**normalize_config(cfg),'action_weight':1.,'memory_image_weight':1.,'memory_scale':'calibrated'}
        cfg['image_prompt_reducer']=cfg.get('image_prompt_reducer',cfg['prompt_reducer'])
        cfg['text_prompt_reducer']=cfg.get('text_prompt_reducer','max' if args.feature_mode=='image' else cfg['prompt_reducer'])
        key=json.dumps(cfg,sort_keys=True)
        if key in seen:return
        seen.add(key)
        metrics={name:assess(cache_for(name,cfg['image_prompt_variant'],cfg['text_prompt_variant'],cfg['image_prompt_reducer'],cfg['text_prompt_reducer']),cfg,scope) for name in caches}
        gains=(metrics['UCF-Crime']['roc_auc']/u0-1,metrics['XD-Violence']['average_precision']/x0-1)
        rows.append(dict(config=cfg,selection_metrics=metrics,joint_relative_gain=float(np.mean(gains)),minimum_relative_gain=float(min(gains)),stage=stage))
        if len(rows)%300==0:print(f'{len(rows)} shared prompt configs in {time.monotonic()-start:.1f}s',flush=True)
    for iv,reducer,temp,image in itertools.product(defs,['max','top2','mean','logmeanexp'],[.02,.05,.1,.2,.5],[.25,.5,.75,1.]):
        tv='expanded26' if args.feature_mode=='image' else iv
        if args.feature_mode=='validation' and 'text_similarities' not in source['UCF-Crime'] and reducer!='max':continue
        score(dict(image_prompt_variant=iv,text_prompt_variant=tv,prompt_reducer=reducer,
                   action_temperature=temp,action_image_weight=image,action_transform='before_fusion',pooling='mean'),'coarse_prompts')
    for temp,image,pooling in itertools.product([.01,.02,.03,.05,.1],[.625,.75,.8125,.875,1.],['mean','mean_max_half']):
        score(dict(image_prompt_variant='semantic_shared16_video',text_prompt_variant='expanded26',prompt_reducer='logmeanexp',image_prompt_reducer='logmeanexp',text_prompt_reducer='max',action_temperature=temp,action_image_weight=image,action_transform='before_fusion',pooling=pooling),'prior_strong_shared_probe_pair')
    # Test cross-probe prompt combinations under one fixed, shared design.
    if args.feature_mode!='image':
        for iv,tv,image in itertools.product(defs,defs,[.375,.625,.875]):
            score(dict(image_prompt_variant=iv,text_prompt_variant=tv,prompt_reducer='max',
                       action_temperature=.05,action_image_weight=image,action_transform='before_fusion',pooling='mean'),'cross_probe_prompts')
    seeds=sorted(rows,key=lambda r:r['joint_relative_gain'],reverse=True)[:15]+sorted(rows,key=lambda r:r['minimum_relative_gain'],reverse=True)[:10]
    for row,pooling,transform in itertools.product(seeds,POOLING,['before_fusion','after_fusion']):
        score({**row['config'],'pooling':pooling,'action_transform':transform},'pooling_transform')
    seeds=sorted(rows,key=lambda r:r['joint_relative_gain'],reverse=True)[:10]+sorted(rows,key=lambda r:r['minimum_relative_gain'],reverse=True)[:5]
    for row,mode,strength in itertools.product(seeds,['center','standardize'],[.25,.5,.75,1.]):
        score({**row['config'],'video_normalization':mode,'normalization_strength':strength},'normalization')
    seeds=sorted(rows,key=lambda r:r['joint_relative_gain'],reverse=True)[:10]+sorted(rows,key=lambda r:r['minimum_relative_gain'],reverse=True)[:10]
    for row,mult,delta in itertools.product(seeds,[.6,1,1.5],[-.125,-.0625,0,.0625,.125]):
        score({**row['config'],'action_temperature':float(row['config']['action_temperature']*mult),
               'action_image_weight':float(np.clip(row['config']['action_image_weight']+delta,0,1))},'refine')
    eligible=[r for r in rows if r['minimum_relative_gain']>=0]
    selected=max(eligible or rows,key=lambda r:r['joint_relative_gain'])
    shortlist=sorted(rows,key=lambda r:r['joint_relative_gain'],reverse=True)[:20]+sorted(rows,key=lambda r:r['minimum_relative_gain'],reverse=True)[:10]
    if selected not in shortlist:shortlist.append(selected)
    if args.feature_mode!='validation':
        for row in shortlist:
            for partition in ['validation','evaluation','all_benchmark']:
                row[partition]={name:assess(cache_for(name,row['config']['image_prompt_variant'],row['config']['text_prompt_variant'],row['config']['image_prompt_reducer'],row['config']['text_prompt_reducer']),row['config'],partition) for name in caches}
    result={'feature_mode':args.feature_mode,'selection_scope':scope,'candidate_count':len(rows),'elapsed_seconds':time.monotonic()-start,
            'objective':'Maximize average relative UCF AUC and XD AP gain, require no regression in either if feasible.',
            'baseline':baseline,'selected':selected,'shortlist':shortlist,'candidates':rows,
            'fixed_prompt_metadata':str((args.features/'prompt_features.json').resolve()),
            'limitations':['Benchmark tuning on previously inspected data, not an independent generalization estimate.',
                           'One common recipe and prompt lists for both datasets, no memory bank or per-dataset fitted state.',
                           'Full-clip normalization, if selected, requires offline access to complete video.',
                           'Partial validation text features cannot report evaluation results until inference completes.']}
    (output/'search.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'selected':selected,'count':len(rows)},indent=2),flush=True)

if __name__=='__main__':main()
