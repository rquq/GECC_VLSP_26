"""No-download toy corpus pipeline plus real tiny native generator mechanics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from model.runtime import resolve_output, write_json
from model.data import load_training_split
from augmentation.pipeline import run_pipeline
from augmentation.journal import Journal
from augmentation.inference import NativeEngine
from agents.tests.smoke_pipeline import fixture


class ScriptedCaller:
    """Protocol fixture only; its text is never used in an official dataset."""
    def __init__(self): self.calls = []
    def call(self,role,key,prompt,sampling=False):
        self.calls.append((role,key,prompt))
        if role=='translator':
            body=prompt.split('\n\n')[-1]
            text=body if key.endswith(':forward') else 'novel useful clean reference '+body
        elif role=='author':
            applicable='error of type 2:' in prompt
            text=json.dumps({'applicable':applicable,'clean_span':'useful clean','error_span':'clean useful','explanation':'Toy phrase permutation.'})
        elif 'Sentence A:' in prompt:
            text=json.dumps({'parent_label':0,'translation_label':0,'meaning_preserved':True,'names_and_numbers_preserved':True,
                             'confidence':.99,'evidence':'Toy independent equivalence review.'})
        else:
            source=prompt.split('Sentence:\n')[-1]
            text=json.dumps({'source_label':2,'mechanism':'noun phrase modifier order','correction':source.replace('clean useful','useful clean'),
                             'single_error':True,'required_edit':True,'meaning_preserved':True,'other_error_types':[],
                             'confidence':.99,'evidence':'Toy independently inferred phrase permutation.'})
        return {'id':hashlib.sha256(key.encode()).hexdigest(),'text':text}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    root=resolve_output({'run':{'name':'augmentation-mechanics-'+args.device,'output_root':'runs'}}); root.mkdir(mode=0o700)
    data=fixture(root,'cpu',final_lora=True)
    paths={'original':data/'VGEC 2026_v1.json','classification':data/'VGEC_classifier.json','correction':data/'VGEC_balanced.json'}
    recipe=json.loads((ROOT/'configs/augmentation.json').read_text())
    recipe['data']['counts']={role:len(json.loads(path.read_text())['train']) for role,path in paths.items() if role!='original'}
    recipe['data']['sha256']={role:hashlib.sha256(path.read_bytes()).hexdigest() for role,path in paths.items()}
    recipe['parents']['limit']=4
    caller=ScriptedCaller()
    result=run_pipeline(paths,recipe,caller,root/'scripted',{'source_commit':'test_fixture_no_corpus_promotion'})
    assert result['accepted_error_pairs']>0
    for stage in ('classification','correction'):
        path=Path(result['files'][stage]['path']); raw=json.loads(path.read_text()); base=json.loads(paths[stage].read_text())['train']
        assert raw['train'][:len(base)]==base
        train,val,info=load_training_split(path,0,42); assert not val and info['accepted_error_pairs']>0
    with zipfile.ZipFile(result['archive']['path']) as z:
        assert z.testzip() is None and 'VGEC_classifier.json' in z.namelist() and 'public_test.json' not in z.namelist()
    assert all('public_test' not in prompt for _,_,prompt in caller.calls)
    # Actual generation uses a tiny native Qwen checkpoint with deterministic
    # whole-JSON output, not a scripted call. Only placement is adapted for the
    # available single laptop GPU; production still requires two GPUs.
    from transformers import AutoModelForCausalLM
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    config=AutoConfig.from_pretrained(root/'base')
    with init_empty_weights(): model=AutoModelForCausalLM.from_config(config)
    model.tie_weights(); count=sum(p.numel() for p in model.parameters()); del model
    roles={}
    for role in ('translator','author','reviewer'):
        destination=root/('native-'+role); shutil.copytree(root/'base',destination)
        roles[role]={'name':str(destination),'revision':'fixture','parameters':count,'license':'Apache-2.0'}
    native_recipe={**recipe,'models':roles,'run':{**recipe['run'],'device':args.device}}
    native_recipe['generation']['max_new_tokens']=8
    native_recipe['translation']['max_new_tokens']=8
    def single_placement(model,device):
        return {'parameters':sum(p.numel() for p in model.parameters()),'device_map':{'':'cuda:0'},
                'memory_budget':{},'tied_embeddings':model.get_input_embeddings().weight is model.get_output_embeddings().weight}
    from contextlib import nullcontext
    placement=patch('augmentation.inference.layer_placement',side_effect=single_placement) if args.device=='cuda' else nullcontext()
    with placement:
        with Journal(root/'native',native_recipe,{'fixture':True},NativeEngine) as journal:
            generated=journal.call('author','native-author','label0 item1 token .',False)
            assert generated['status']=='completed' and isinstance(json.loads(generated['text']),dict)
            journal.call('reviewer','native-review','label1 item2 token .',False)
        with Journal(root/'native',native_recipe,{'fixture':True},NativeEngine) as journal:
            assert journal.call('author','native-author','label0 item1 token .',False)==generated
    write_json(root/'verification.json',{'status':'passed','scope':'Toy/no-download mechanics only; no real corpus augmentation or score claim',
               'device':args.device,'scripted_accepted_pairs':result['accepted_error_pairs'],'native_parameters':count,
               'native_journal_resume_verified':True,'run':str(root)})
    print(json.dumps(json.loads((root/'verification.json').read_text()),indent=2))


if __name__=='__main__': main()
