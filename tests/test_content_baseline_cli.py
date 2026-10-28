import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import build_recommender_samples as builder
import evaluate_content_baseline as baseline
from common import sha256
from test_sample_builder import toy_dataset


def test_baseline_uses_explicit_candidates_and_defaults_to_validation_only(toy_dataset,tmp_path):
    ids,lookup,train,val,test=builder.load_catalog()
    evaluation=tmp_path/'eval'
    builder.build_split('validation',ids,lookup,train,val,test,output=evaluation,negatives=3)
    sparse.save_npz(toy_dataset/'content/tfidf_all.npz',sparse.eye(len(ids),format='csr',dtype=np.float32))
    (toy_dataset/'split_manifest.json').write_text('{}')
    output=tmp_path/'baseline'
    command=[sys.executable,str(Path(baseline.__file__)),'--data',str(toy_dataset),
             '--evaluation',str(evaluation),'--output',str(output)]
    done=subprocess.run(command,capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    report=json.loads((output/'metrics.json').read_text())
    assert set(report['splits'])=={'validation'}
    assert report['test_evaluated'] is False
    assert report['sources']['samples_validation']==sha256(evaluation/'samples_validation.npz')
    assert report['splits']['validation']['tfidf_profile']['overall']['samples']==40
    assert subprocess.run(command,capture_output=True).returncode!=0


def test_research_cli_rejects_synthetic_demo_before_evaluation(tmp_path):
    (tmp_path/'demo_manifest.json').write_text('{}')
    output=tmp_path/'results'
    result=subprocess.run([sys.executable,str(Path(baseline.__file__)),'--data',str(tmp_path),
                            '--output',str(output)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Synthetic demo' in result.stderr
    assert not output.exists()
