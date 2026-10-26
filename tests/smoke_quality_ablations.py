"""Actual offline three-fit/reload controller exercise with a tiny native Qwen model."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.runtime import write_json
from tools.run_quality_ablations import run,FORMAT


def main(root):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast,Qwen2Config,Qwen2ForSequenceClassification
    root=Path(root).resolve();root.mkdir(parents=True)
    torch.set_num_threads(1)
    base=root/'base';base.mkdir()
    vocab={'<pad>':0,'<unk>':1,'<eos>':2,'input':3,'target':4}
    for i in range(60):vocab['sample'+str(i)]=len(vocab)
    for i in range(5):vocab['label'+str(i)]=len(vocab)
    backend=Tokenizer(WordLevel(vocab,unk_token='<unk>'));backend.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',unk_token='<unk>',eos_token='<eos>')
    tokenizer.save_pretrained(base)
    torch.manual_seed(42)
    Qwen2ForSequenceClassification(Qwen2Config(vocab_size=len(vocab),hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,
        num_labels=5,pad_token_id=0,eos_token_id=2)).save_pretrained(base)
    quality=[{'id':i,'source':f'input label{i%5} sample{i}','label':i%5,'target':f'target label{i%5} sample{i}'} for i in range(20)]
    organizer=quality[:10]
    dev=[{'id':i,'source':f'input label{i%5} sample{i}','label':i%5,'target':f'target label{i%5} sample{i}'} for i in range(20,30)]
    package=root/'input'/'arbitrary mount'/'nested'/'payload';(package/'data').mkdir(parents=True);(package/'baseline').mkdir()
    for name,rows in [('quality',quality),('organizer',organizer),('validation',dev)]:write_json(package/'data'/(name+'.json'),rows)
    for name in ('qwen7b','sailor7b','mistral7b','correction'):
        rows=[]
        for r in dev:
            rows.append({'id':r['id'],'source':r['source'],'target_class':r['label'],'target_correction':r['target'],
                         'predicted_class':r['label'],'predicted_correction':r['target'],
                         'probabilities':[.8 if i==r['label'] else .05 for i in range(5)]})
        write_json(package/'baseline'/(name+'.json'),rows)
    manifest={'format':FORMAT,'quality_source_commit':'tiny-offline-mechanics-only','counts':{'quality':20,'organizer':10,'validation':10},
              'files':{str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*.json')}}
    write_json(package/'manifest.json',manifest)
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/quality_ablation_classification.json').read_text())
    config['model'].update(name=str(base),revision='tiny-local',compute_dtype='float32',device_map='single')
    config['run']['device']='cpu'
    offline_tracking=os.environ.get('VGEC_SMOKE_OFFLINE_TRACKING')=='1'
    config['tracking'].update(enabled=offline_tracking,mode='offline' if offline_tracking else 'disabled')
    config['training']['max_length']=32;config['training']['effective_batch_size']=2
    config['training']['args'].update(num_train_epochs=5,per_device_train_batch_size=1,per_device_eval_batch_size=1,
                                      gradient_checkpointing=False,warmup_steps=0,logging_steps=1)
    plan={'name':'tiny-quality-ablation','classification_config':config,
          'package_proof':{'manifest_sha256':hashlib.sha256((package/'manifest.json').read_bytes()).hexdigest()},
          'optimizer_steps':6,'evaluation_steps':2,'arms':[
              {'name':'quality_hierarchical','dataset':'quality','objective':'hierarchical'},
              {'name':'organizer_hierarchical','dataset':'organizer','objective':'hierarchical'},
              {'name':'quality_flat','dataset':'quality','objective':'flat'}]}
    result=run(plan,root/'input',root/'runs',source_identity={'commit':'tiny-mechanics-only'},device='cpu')
    summary=json.loads((result/'ablation_summary.json').read_text())
    for name in summary['fresh_training']:
        history=json.loads((result/'arms'/name/'epoch_history.json').read_text())
        assert [r['optimizer_steps'] for r in history]==[2,4,6]
        receipt=json.loads((result/'arms'/name/'run_manifest.json').read_text())
        assert receipt['saved_classifier_reload_verified'] and receipt['optimizer_steps']==6
    assert len(summary['historical_quality_cached'])==8
    assert (result/'ablation_results.zip').is_file()
    print('SMOKE PASSED: three actual fits, three selected-adapter reloads, identical selection schedule, cached votes, summaries and ZIP.')
    print('SMOKE_OUTPUT:'+str(result))


if __name__=='__main__':main(sys.argv[1])
