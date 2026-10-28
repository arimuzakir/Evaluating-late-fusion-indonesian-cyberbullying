"""Agreement of recorded ratings; this does not certify the human review process."""
from pathlib import Path
import json,itertools,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import cohen_kappa_score

def run(root):
    root=Path(root);label=root/'data/labels/v1_1_GoldAdjudicated'
    master=pd.read_csv(label/'data/dataset_master_13307_labeled.csv')
    raw=pd.read_csv(root/'data/annotation_sources/anotasi_mentah_5_anotator.csv')
    cols=[f'anotator_{i}' for i in range(1,6)]
    assert master.set_index('master_id')[cols].equals(raw.set_index('master_id')[cols])
    a=raw[cols].to_numpy(int);ones=a.sum(1);counts=np.column_stack([5-ones,ones])
    observed=((counts*(counts-1)).sum(1)/20).mean();expected=((counts.sum(0)/(len(a)*5))**2).sum()
    sample=pd.read_csv(label/'annotation/WINDOW_AUDIT_REFERENCE_153.csv')
    changes=pd.read_csv(label/'annotation/WINDOW_LABEL_CHANGELOG_4.csv')
    overrides=master.loc[(master[cols].sum(1)>=3).astype(int)!=master.label_final,'master_id']
    assert set(overrides)==set(changes.master_id)
    e1=pd.read_csv(label/'annotation/source_snapshots/EXPERT_1_INDEPENDENT_WINDOW_REVIEW.csv')
    e2=pd.read_csv(label/'annotation/source_snapshots/EXPERT_2_INDEPENDENT_WINDOW_REVIEW.csv')
    experts=e1.merge(e2,on=['audit_item_id','master_id'],suffixes=('_e1','_e2'),validate='one_to_one')
    assert len(experts)==153
    assert np.array_equal(experts.set_index('master_id').loc[sample.master_id].window_label_e1,sample.window_label_expert1)
    assert np.array_equal(experts.set_index('master_id').loc[sample.master_id].window_label_e2,sample.window_label_expert2)
    summary={'n_recorded_items':len(a),'recorded_vote_cells':int(a.size),
       'fleiss_kappa':float((observed-expected)/(1-expected)),
       'mean_pairwise_cohen_kappa':float(np.mean([cohen_kappa_score(a[:,i],a[:,j]) for i,j in itertools.combinations(range(5),2)])),
       'window_review_n':len(sample),'window_kappa_before_adjudication':float(cohen_kappa_score(sample.window_label_expert1,sample.window_label_expert2)),
       'window_agreement_n':int(sample.window_label_expert1.eq(sample.window_label_expert2).sum()),
       'discordant_window_reviews':int(sample.window_label_expert1.ne(sample.window_label_expert2).sum()),
       'majority_overrides':changes.master_id.tolist(),
       'scope':'recorded ratings only; human independence is researcher-reported; no identity or certification verification',
       'researcher_clarified_cases':['CB_MASTER_00202','CB_MASTER_00111','CB_MASTER_09067'],
       'labels_modified':False,'historical_votes_modified':False}
    out=root/'results/reproduction';out.mkdir(parents=True,exist_ok=True)
    (out/'ANNOTATION_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return summary
