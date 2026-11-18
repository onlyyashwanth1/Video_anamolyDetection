#!/usr/bin/env python3
"""Paired source-group bootstrap and curves from actual saved predictions.

The evaluation partition reuses already inspected benchmark data. Its estimates
and bootstrap intervals describe this development comparison; they are not a
new independent test or an adjustment for configuration-search uncertainty.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINES = {
    'UCF-Crime': ROOT/'outputs/ucf_expanded95_evaluation/20261022T192245Z',
    'XD-Violence': ROOT/'outputs/xd_violence_multi_gpu_evaluation/20261022T192245Z',
}
SHORT = {'UCF-Crime':'ucf', 'XD-Violence':'xd'}
SPLITS = ROOT/'outputs/results/score_ablations_20261026/ablations.json'
LIMITATION = (
    'Conditional descriptive intervals on already inspected benchmark data. '
    'Source groups, rather than individual frames, are resampled in pairs; '
    'intervals do not account for configuration-search uncertainty and do not '
    'constitute a new independent test.'
)


class SortedCounts:
    """Sort scores once; subsequent paired bootstrap draws change counts only."""
    def __init__(self, scores, positive, negative, groups):
        order=np.argsort(scores,kind='stable')[::-1]
        self.scores=np.asarray(scores)[order]
        self.positive=np.asarray(positive,dtype=float)[order]
        self.negative=np.asarray(negative,dtype=float)[order]
        self.groups=np.asarray(groups,dtype=np.int32)[order]
        self.ends=np.r_[np.flatnonzero(np.diff(self.scores)),len(self.scores)-1]

    def curves(self, group_weights=None):
        if group_weights is None:
            positive,negative=self.positive,self.negative
        else:
            multiplier=group_weights[self.groups]
            positive,negative=self.positive*multiplier,self.negative*multiplier
        tp=np.cumsum(positive)[self.ends]
        fp=np.cumsum(negative)[self.ends]
        if not len(tp) or not tp[-1] or not fp[-1]:
            raise ValueError('A bootstrap sample must contain both frame classes')
        # A resampled group can remove every frame at a particular score.
        # Remove those empty score bins, especially the leading bins, before
        # computing precision so no 0/0 contaminates the AP integration.
        keep=np.r_[True,np.diff(tp+fp)>0]
        keep &= (tp+fp)>0
        tp,fp=tp[keep],fp[keep]
        recall=np.r_[0.,tp/tp[-1]]
        fpr=np.r_[0.,fp/fp[-1]]
        precision=np.r_[1.,tp/(tp+fp)]
        auc=float(np.sum(np.diff(fpr)*(recall[1:]+recall[:-1])*.5))
        ap=float(np.sum(np.diff(recall)*precision[1:]))
        return {'roc_auc':auc,'average_precision':ap},(fpr,recall,precision)


def check_weighted_implementation():
    """Independent toy check against explicitly repeated frames, including ties."""
    from sklearn.metrics import average_precision_score,roc_auc_score
    scores=np.array([.2,.2,.8]);p=np.array([2,0,1]);n=np.array([0,3,2]);g=np.array([0,0,1])
    counter=SortedCounts(scores,p,n,g)
    for weights in (np.array([1,1]),np.array([2,3]),np.array([0,1])):
        x=[];y=[]
        for score,pos,neg,group in zip(scores,p,n,g):
            x.extend([score]*int((pos+neg)*weights[group]))
            y.extend(([1]*int(pos)+[0]*int(neg))*int(weights[group]))
        got,_=counter.curves(weights)
        assert np.isclose(got['roc_auc'],roc_auc_score(y,x),rtol=0,atol=1e-12)
        assert np.isclose(got['average_precision'],average_precision_score(y,x),rtol=0,atol=1e-12)


def compress(scores,labels,group):
    """Collapse adjacent equal score/label runs while retaining exact counts."""
    starts=np.r_[0,1+np.flatnonzero((scores[1:]!=scores[:-1])|(labels[1:]!=labels[:-1]))]
    length=np.diff(np.r_[starts,len(scores)])
    pos=length*labels[starts]
    return scores[starts],pos,length-pos,np.full(len(starts),group,dtype=np.int32)


def load_dataset(dataset,selected_root,splits):
    original=BASELINES[dataset];selected=selected_root/SHORT[dataset]
    baseline_summary=json.loads((original/'summary.json').read_text())
    selected_summary=json.loads((selected/'summary.json').read_text())
    names=sorted(baseline_summary['per_video'])
    if set(names)!=set(selected_summary['per_video']):
        raise ValueError(f'{dataset}: model video sets differ')
    validation=set(splits['diagnostic_validation_videos'])
    evaluation=set(splits['diagnostic_evaluation_videos'])
    if validation & evaluation or validation|evaluation != set(names):
        raise ValueError('Invalid pre-existing validation/evaluation partition')
    group_name=lambda name:name.split('__#')[0]
    validation_groups={group_name(name) for name in validation}
    evaluation_groups={group_name(name) for name in evaluation}
    if validation_groups&evaluation_groups:
        raise ValueError('Source groups cross partitions')
    groups={name:i for i,name in enumerate(sorted(validation_groups|evaluation_groups))}
    evaluation_ids=np.array([groups[name] for name in sorted(evaluation_groups)])
    parts={'baseline':[],'selected':[]};frame_count=positive_count=0
    for name in names:
        with np.load(original/'predictions'/f'{name}.npz') as z:
            labels=np.asarray(z['labels']);old=np.asarray(z['scores'])
        with np.load(selected/'predictions'/f'{name}.npz') as z:
            new_labels=np.asarray(z['labels']);new=np.asarray(z['scores'])
        if (old.ndim!=1 or new.shape!=old.shape or labels.shape!=old.shape
                or not len(old) or not np.array_equal(labels,new_labels)
                or not np.isfinite(old).all() or not np.isfinite(new).all()
                or not np.isin(labels,[0,1]).all()):
            raise ValueError(f'{dataset}/{name}: incompatible frame predictions')
        frame_count+=len(labels);positive_count+=int(labels.sum())
        for key,scores in [('baseline',old),('selected',new)]:
            parts[key].append(compress(scores,labels,groups[group_name(name)]))
    counters={}
    for key,items in parts.items():
        columns=[np.concatenate([row[i] for row in items]) for i in range(4)]
        scores,p,n,g=columns
        mask=np.isin(g,evaluation_ids)
        counters[key]={
            'all_benchmark':SortedCounts(scores,p,n,g),
            'evaluation':SortedCounts(scores[mask],p[mask],n[mask],g[mask]),
        }
    summaries={'baseline':baseline_summary,'selected':selected_summary}
    metrics={}
    for key,partitions in counters.items():
        metrics[key]={part:counter.curves()[0] for part,counter in partitions.items()}
        for metric in ('roc_auc','average_precision'):
            expected=summaries[key]['overall'][metric]
            if not np.isclose(metrics[key]['all_benchmark'][metric],expected,rtol=0,atol=1e-12):
                raise ValueError(f'{dataset}/{key}: pooled {metric} differs from actual run summary')
        # Production metadata's partition numbers provide an extra independent
        # check without changing the pre-existing split used for this report.
        reported=summaries[key].get('partitions',{}).get('evaluation',{})
        reported=reported.get('overall',reported)
        for metric in ('roc_auc','average_precision'):
            if metric in reported and not np.isclose(metrics[key]['evaluation'][metric],reported[metric],rtol=0,atol=1e-12):
                raise ValueError(f'{dataset}/{key}: evaluation {metric} differs from production summary')
    return counters,metrics,evaluation_ids,len(groups),{
        'dataset':dataset,'baseline_run':str(original.resolve()),'selected_run':str(selected.resolve()),
        'baseline_summary_sha256':hashlib.sha256((original/'summary.json').read_bytes()).hexdigest(),
        'selected_summary_sha256':hashlib.sha256((selected/'summary.json').read_bytes()).hexdigest(),
        'frames':frame_count,'anomalous_frames':positive_count,'videos':len(names),
        'evaluation_videos':len(evaluation),'evaluation_source_groups':len(evaluation_groups),
        'source_group_rule':'Video stem before __#; full stem for names without __#.',
        'source_groups_resampled':sorted(evaluation_groups),
        'selected_model_sha256':selected_summary.get('model_sha256'),
        'selected_config_sha256':selected_summary.get('config_sha256'),
    }


def bootstrap(counters,evaluation_ids,group_count,replicates,seed):
    rng=np.random.default_rng(seed);draws=[];discarded=0
    while len(draws)<replicates:
        counts=rng.multinomial(len(evaluation_ids),np.full(len(evaluation_ids),1/len(evaluation_ids)))
        weights=np.zeros(group_count,dtype=np.int32);weights[evaluation_ids]=counts
        try:
            old=counters['baseline']['evaluation'].curves(weights)[0]
            new=counters['selected']['evaluation'].curves(weights)[0]
        except ValueError:
            discarded+=1
            if discarded>replicates:raise RuntimeError('Too many degenerate bootstrap samples')
            continue
        draws.append([old['roc_auc'],new['roc_auc'],old['average_precision'],new['average_precision']])
        if len(draws)%50==0:print(f'  {len(draws)}/{replicates} paired source-group draws',flush=True)
    draws=np.asarray(draws)
    result={}
    for metric,base_column,new_column in [('roc_auc',0,1),('average_precision',2,3)]:
        result[metric]={
            'baseline_interval_95':np.quantile(draws[:,base_column],[.025,.975]).tolist(),
            'selected_interval_95':np.quantile(draws[:,new_column],[.025,.975]).tolist(),
            'gain_interval_95_percentage_points':(100*np.quantile(draws[:,new_column]-draws[:,base_column],[.025,.975])).tolist(),
            'fraction_resamples_with_positive_gain':float(np.mean(draws[:,new_column]>draws[:,base_column])),
        }
    return result,draws,discarded


def draw_curve(dataset,counters,metrics,partition,output):
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    metric='roc_auc' if dataset=='UCF-Crime' else 'average_precision'
    for key,label,color in [('baseline','Original','#6b7280'),('selected','Shared selected','#007f8b')]:
        _,(fpr,recall,precision)=counters[key][partition].curves()
        # Raster plots cannot resolve millions of points. Metrics use every
        # exact score tie; only the rendered path is thinned to 15k points.
        ids=np.unique(np.linspace(0,len(recall)-1,min(len(recall),15000)).astype(int))
        legend=f'{label} ({"AUC" if metric=="roc_auc" else "AP"} {100*metrics[key][partition][metric]:.2f}%)'
        if metric=='roc_auc':ax.plot(fpr[ids],recall[ids],label=legend,color=color,lw=1.8)
        else:ax.step(recall[ids],precision[ids],where='pre',label=legend,color=color,lw=1.6)
    if metric=='roc_auc':
        ax.plot([0,1],[0,1],ls='--',color='#b0b0b0',lw=1)
        ax.set(xlabel='False positive rate',ylabel='True positive rate')
        filename=f'{SHORT[dataset]}_roc_{partition}.png'
    else:
        c=counters['baseline'][partition];prevalence=c.positive.sum()/(c.positive+c.negative).sum()
        ax.axhline(prevalence,ls='--',color='#b0b0b0',lw=1,label=f'Positive prevalence {100*prevalence:.2f}%')
        ax.set(xlabel='Recall',ylabel='Precision')
        filename=f'{SHORT[dataset]}_precision_recall_{partition}.png'
    scope='Evaluation subset' if partition=='evaluation' else 'Full reused benchmark'
    ax.set(title=f'{dataset}: {scope}',xlim=(0,1),ylim=(0,1))
    ax.grid(alpha=.18);ax.legend(loc='lower right' if metric=='roc_auc' else 'upper right',frameon=False)
    fig.savefig(output/filename,dpi=180)
    plt.close(fig)
    return filename


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-root',type=Path,default=ROOT/'outputs/shared_model_evaluation/20261026')
    parser.add_argument('--splits',type=Path,default=SPLITS)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/results/shared_model_comparison_20261026')
    parser.add_argument('--replicates',type=int,default=200)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.replicates<2:parser.error('At least two bootstrap replicates are required')
    check_weighted_implementation();start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    split_data=json.loads(args.splits.read_text());result={};all_draws={};plots=[]
    for dataset in BASELINES:
        print(f'Loading paired native-frame predictions: {dataset}',flush=True)
        split=next(s for s in split_data if s['dataset']==dataset)
        counters,metrics,evaluation_ids,group_count,provenance=load_dataset(dataset,args.selected_root,split)
        intervals,draws,discarded=bootstrap(counters,evaluation_ids,group_count,args.replicates,args.seed)
        for metric in intervals:
            intervals[metric]['gain_percentage_points']=100*(metrics['selected']['evaluation'][metric]-metrics['baseline']['evaluation'][metric])
        result[dataset]={'metrics':metrics,'evaluation_paired_source_bootstrap':intervals,
                         'discarded_degenerate_samples':discarded,'provenance':provenance}
        all_draws[SHORT[dataset]]=draws
        for partition in ('evaluation','all_benchmark'):
            plots.append(draw_curve(dataset,counters,metrics,partition,args.output))
        print(json.dumps({'dataset':dataset,'metrics':metrics,'intervals':intervals}),flush=True)
    payload={
        'replicates':args.replicates,'seed_per_dataset':args.seed,
        'interval_method':'Paired source-group percentile bootstrap, 2.5th and 97.5th percentiles.',
        'bootstrap_scope':'Pre-existing evaluation groups only; draw number of source groups with replacement.',
        'metric_scope':'Pooled native-frame metrics. Entire source-group frame lengths are preserved in each resample.',
        'implementation_checks':'Exact grouped toy metrics matched independently expanded sklearn metrics; full/evaluation point metrics checked against available actual run summaries.',
        'limitation':LIMITATION,'datasets':result,'plots':plots,'elapsed_seconds':time.monotonic()-start,
    }
    (args.output/'comparison.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output/'bootstrap_draws.npz',**all_draws,
        columns=np.array(['baseline_roc_auc','selected_roc_auc','baseline_average_precision','selected_average_precision']))
    lines=['# Actual shared-model comparison','',LIMITATION,'',
           '| Evaluation subset | Original | Shared selected | Gain (pp) | Paired source-group 95% interval for gain (pp) |',
           '|---|---:|---:|---:|---:|']
    for dataset,metric in [('UCF-Crime','roc_auc'),('XD-Violence','average_precision')]:
        d=result[dataset];m=d['metrics'];b=d['evaluation_paired_source_bootstrap'][metric];lo,hi=b['gain_interval_95_percentage_points']
        lines.append(f'| {dataset} {"ROC-AUC" if metric=="roc_auc" else "AP"} | {m["baseline"]["evaluation"][metric]*100:.3f}% | {m["selected"]["evaluation"][metric]*100:.3f}% | {b["gain_percentage_points"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |')
    lines+=['',f'{args.replicates} paired resamples per dataset; seed {args.seed}. Source groups are filename prefixes before `__#`, with full video stems used otherwise. Scores are sorted once per model/partition; each draw reweights the complete source groups.','',
            '| Full reused benchmark | Original | Shared selected |','|---|---:|---:|']
    for dataset,metric in [('UCF-Crime','roc_auc'),('XD-Violence','average_precision')]:
        m=result[dataset]['metrics']
        lines.append(f'| {dataset} {"ROC-AUC" if metric=="roc_auc" else "AP"} | {m["baseline"]["all_benchmark"][metric]*100:.3f}% | {m["selected"]["all_benchmark"][metric]*100:.3f}% |')
    lines+=['','Curves are rendered from actual native-frame predictions; their legends use exact pooled metrics.','']
    for plot in plots:lines.append(f'![{plot}]({plot})')
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Wrote {args.output}/comparison.json ({time.monotonic()-start:.1f}s)',flush=True)


if __name__=='__main__':main()
