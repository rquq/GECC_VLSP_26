"""Controlled quality-data/classifier ablations, using the submitted LoRA trainer."""
from contextlib import ExitStack
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import zipfile

from model.runtime import PROJECT_ROOT, resolve_output, write_json

FORMAT = "vgec-quality-ablations-package-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_rows(quality, organizer, validation):
    from model.data import _validate_records, _id_key, _source_key
    for rows in (quality, organizer, validation):
        _validate_records(rows)
        if set(r['label'] for r in rows) != set(range(5)):
            raise ValueError("Each partition must contain all five classes")
    dev_ids = {_id_key(r['id']) for r in validation}
    dev_text = {_source_key(r[k]) for r in validation for k in ('source', 'target')}
    for rows in (quality, organizer):
        if dev_ids & {_id_key(r['id']) for r in rows}:
            raise ValueError("Development IDs occur in training")
        if dev_text & {_source_key(r[k]) for r in rows for k in ('source', 'target')}:
            raise ValueError("Development source/target text occurs in training")
    quality_by_id = {_id_key(r['id']): r for r in quality}
    if any(quality_by_id.get(_id_key(r['id'])) != r for r in organizer):
        raise ValueError("Organizer-only rows must be unchanged eligible quality originals")


def find_package(input_root, expected_sha256):
    matches = []
    for path in sorted(Path(input_root).rglob('manifest.json')):
        try:
            packet = read(path)
        except (ValueError, OSError):
            continue
        if isinstance(packet,dict) and packet.get('format') == FORMAT and digest(path) == expected_sha256:
            matches.append((path.parent, packet))
    if len(matches) != 1:
        raise ValueError("Attach one extracted VGEC26-quality-ablations-20261027 dataset; "
                         f"found {len(matches)} matching manifests under {input_root}")
    root, manifest = matches[0]
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing or changed package payload: {relative}")
    return root, manifest


def study_config(base, arm, plan, output):
    config = copy.deepcopy(base)
    config.pop('ensemble', None)
    config['run'].update(name=arm['name'], world_size=1, seed=42, output_root=str(output))
    config['oof'] = {'enabled': False}
    config['training']['objective'] = arm['objective']
    config['training']['evaluation_steps'] = plan['evaluation_steps']
    config['training']['args'].update(max_steps=plan['optimizer_steps'])
    config['evaluation'] = {'baseline': False}
    return config


def validate_cached(records, rows):
    from model.data import _id_key
    expected = {_id_key(r['id']): r for r in records}
    if len(rows) != len(records) or {_id_key(r['id']) for r in rows} != set(expected):
        raise ValueError("Cached predictions must match every typed development ID exactly")
    for row in rows:
        gold = expected[_id_key(row['id'])]
        if any(row.get(k) != gold[g] for k, g in [('source','source'),('target_class','label'),('target_correction','target')]):
            raise ValueError("Cached prediction source/label/reference differs from development data")
        if row.get('probabilities') is not None:
            p = row['probabilities']
            if len(p) != 5 or any(not isinstance(x,(float,int)) or not math.isfinite(x) or x < 0 for x in p) or abs(sum(p)-1)>1e-5:
                raise ValueError("Cached probabilities are not a five-class distribution")


def cached_comparisons(package, records):
    from model.classifier_ensemble import vote
    from model.classification import _classification_metrics
    members = []
    reports = {}
    for name in ('qwen7b', 'sailor7b', 'mistral7b'):
        rows = read(package / 'baseline' / (name+'.json'))
        validate_cached(records, rows)
        p = [{'id':r['id'], 'label':r['predicted_class'], 'probabilities':r['probabilities']} for r in rows]
        members.append(p)
        reports[name] = _classification_metrics(records, p)
    for name, weights, phrase in [('equal_plain',[1,1,1],1.), ('equal_phrase',[1,1,1],1.25),
                                  ('quality_112_plain',[1,1,2],1.), ('quality_112_phrase',[1,1,2],1.25),
                                  ('sailor_121_phrase',[1,2,1],1.25)]:
        p = vote(members, {'weights':weights,'temperature':1.,'phrase_scale':phrase})
        reports[name] = _classification_metrics(records, p)
    corrections = read(package/'baseline/correction.json')
    validate_cached(records, corrections)
    from model.metrics import experiment_report
    correction_report = experiment_report(records, [{'id':r['id'], 'label':r['predicted_class'],
                                                       'target':r['predicted_correction']} for r in corrections])
    return reports, correction_report['corrections']


def write_summary(output, fresh, cached, corrections, manifest, plan):
    gleu = corrections['all']['gleu_plus']
    summary = {'scope':'Repeatedly used 78-row development set; not an independent or private-test result.',
               'quality_source_commit':manifest['quality_source_commit'], 'plan':plan,
               'fresh_training':fresh, 'historical_quality_cached':cached,
               'fixed_historical_quality_corrector':corrections,
               'combined_scope':'Illustrative pairing with unchanged historical quality corrector outputs; no corrector retrain or private inference.'}
    write_json(output/'ablation_summary.json', summary)
    fields = ['source','name','macro_f1','micro_f1','phrase_f1','function_word_f1',
              'selected_step','final_step_macro_f1','optimizer_steps','train_rows','examples_seen',
              'fixed_corrector_gleu','illustrative_combined','elapsed_seconds']
    rows = []
    for source, entries in [('fresh',fresh),('cached_338293f',cached)]:
        for name, entry in entries.items():
            m = entry['classification'] if source=='fresh' else entry
            row = {'source':source,'name':name,'macro_f1':m['macro_f1'],'micro_f1':m['micro_f1'],
                   'phrase_f1':m['per_class']['2']['f1'],'function_word_f1':m['per_class']['3']['f1'],
                   'fixed_corrector_gleu':gleu,'illustrative_combined':(m['macro_f1']+gleu)/2}
            if source=='fresh':
                row.update({k:entry[k] for k in ('selected_step','final_step_macro_f1','optimizer_steps','train_rows','examples_seen','elapsed_seconds')})
            rows.append(row)
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fields);writer.writeheader();writer.writerows(rows)
    lines=['# Quality-run development ablations','',summary['scope'],'',
           '| Source | Model / rule | Macro-F1 | Micro-F1 | Phrase F1 | Function-word F1 |',
           '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+row['source']+' | '+row['name']+' | '+' | '.join(f"{100*row[k]:.2f}" for k in ('macro_f1','micro_f1','phrase_f1','function_word_f1'))+' |')
    lines += ['', 'Fresh arms share optimizer-step and evaluation budgets; historical cached fits are a separate source.',
              'Organizer-only reuses fewer distinct rows for more passes; class weights follow the same fit-local formula. Final-step results, actual epochs and exposure estimates are retained in JSON.',
              'No model was trained on development or private-test inputs. No independent/private improvement is established.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with zipfile.ZipFile(output/'ablation_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and (path.name in {'ablation_summary.json','comparison.csv','comparison.md','pipeline_manifest.json'}
                                  or path.name in {'epoch_history.json','trainer_history.json','metrics.json','config.json','run_manifest.json','predictions.json','selected_epoch_predictions.json'}):
                archive.write(path,str(path.relative_to(output)))
    return summary


def prepare_inputs(plan, input_root):
    if (type(plan['optimizer_steps']) is not int or plan['optimizer_steps']<1
            or type(plan['evaluation_steps']) is not int or not 1<=plan['evaluation_steps']<=plan['optimizer_steps']):
        raise ValueError('Invalid shared optimization/evaluation budget')
    arms=plan['arms']
    if (len(arms)!=3 or len({a['name'] for a in arms})!=3
            or {(a['dataset'],a['objective']) for a in arms}!={('quality','hierarchical'),('organizer','hierarchical'),('quality','flat')}):
        raise ValueError('Expected the three predeclared quality ablation arms')
    proof = read(PROJECT_ROOT/plan['package_proof']) if isinstance(plan['package_proof'],str) else plan['package_proof']
    package, packet = find_package(input_root,proof['manifest_sha256'])
    quality=read(package/'data/quality.json');organizer=read(package/'data/organizer.json');validation=read(package/'data/validation.json')
    validate_rows(quality,organizer,validation)
    for key, rows in [('quality',quality),('organizer',organizer),('validation',validation)]:
        if len(rows)!=packet['counts'][key]:raise ValueError('Package row count mismatch: '+key)
    cached,corrections=cached_comparisons(package,validation)
    return package,packet,quality,organizer,validation,cached,corrections


def run(plan, input_root, output_root, *, source_identity=None, device='cuda'):
    from model.tracking import TrackingRun
    from model.reproducibility import process_environment
    from model.two_phase import _run_stage
    from model.weight_cache import temporary_cache_environment, prepare_weight_cache
    package,packet,quality,organizer,validation,cached,corrections=prepare_inputs(plan,input_root)
    proof = read(PROJECT_ROOT/plan['package_proof']) if isinstance(plan['package_proof'],str) else plan['package_proof']
    work=resolve_output({'run':{'name':plan['name'],'output_root':str(output_root)}})
    output=work/'run';output.mkdir(parents=True)
    print('VGEC_RUN_DIR:'+str(output),flush=True)
    base=read(PROJECT_ROOT/plan['classification_config']) if isinstance(plan['classification_config'],str) else plan['classification_config']
    environment={**process_environment(42),'VGEC_CUDA_STAGE':'1' if device=='cuda' else '0',
                 'VGEC_FORWARD_EVENTS':'1','VGEC_PIPELINE_TRACKING':'1','WANDB_RUN_GROUP':work.name}
    fresh={}
    write_json(work/'experiment_manifest.json',{'source':source_identity,'plan':plan,'package':packet,'controller_deadline':None})
    with TrackingRun() as tracking, ExitStack() as resources:
        environment.update(temporary_cache_environment(resources))
        for relative in packet['files']:
            if relative.endswith('.json'):tracking.protect_console_file(package/relative)
        tracking.start(base,output,'pipeline',group=work.name,metadata={'kind':'quality-ablation','historical_source':'338293f','fresh_fits':3})
        for name,metrics in cached.items():
            tracking.evaluation({'classification':metrics},'historical_quality/'+name)
        tracking.evaluation({'corrections':corrections},'fixed_historical_corrector')
        try:
            for arm in plan['arms']:
                root=output/'arms'/arm['name'];root.mkdir(parents=True)
                config=study_config(base,arm,plan,root);config['_tracking_group']=work.name
                config['run']['device']=device
                write_json(root/'config.json',config)
                write_json(root/'train.json',quality if arm['dataset']=='quality' else organizer)
                write_json(root/'validation.json',validation)
                started=time.monotonic()
                for phase in ('fit','verify'):
                    prepare_weight_cache(config['model'],environment)
                    message='Starting '+arm['name']+' / '+phase
                    print(message,flush=True);tracking.console_line(message)
                    _run_stage([sys.executable,'-u','-m','tools.quality_ablation_worker','--output-dir',str(root),
                                '--device',device,'--phase',phase],
                               {**environment,'VGEC_STAGE_NAME':arm['name']+'/'+phase},tracking,timeout_seconds=None)
                metrics=read(root/'metrics.json');receipt=read(root/'training_complete.json');history=read(root/'epoch_history.json')
                steps=[r['optimizer_steps'] for r in history]
                expected=list(range(plan['evaluation_steps'],plan['optimizer_steps']+1,plan['evaluation_steps']))
                if plan['optimizer_steps'] not in expected:expected.append(plan['optimizer_steps'])
                if steps!=expected or receipt['optimizer_steps']!=plan['optimizer_steps']:
                    raise RuntimeError('Training/evaluation budget differs between arms')
                n=receipt['train_count'];batch=receipt['effective_batch_size'];updates=math.ceil(n/batch)
                full,tail=divmod(receipt['optimizer_steps'],updates)
                fresh[arm['name']]={'classification':metrics['classification'],'objective':arm['objective'],
                                   'dataset':arm['dataset'],'train_rows':n,'examples_seen':full*n+min(tail*batch,n),
                                   'optimizer_steps':receipt['optimizer_steps'],'selected_step':receipt['best_optimizer_steps'],
                                   'selected_epoch':receipt['best_epoch'],'final_step_macro_f1':history[-1]['classification']['macro_f1'],
                                   'class_weighting':receipt['class_weighting'],'elapsed_seconds':time.monotonic()-started}
                tracking.evaluation(metrics,'arms/'+arm['name'])
                write_json(output/'partial_results.json',fresh)
            write_json(output/'pipeline_manifest.json',{'mode':'quality_ablations','status':'completed','arms':list(fresh),
                                                        'source':source_identity,'package_manifest_sha256':proof['manifest_sha256'],
                                                        'scope':'Development comparisons only; no private submission generated.'})
            result=write_summary(output,fresh,cached,corrections,packet,plan)
            tracking.summary({'status':'completed','fresh_fits':len(fresh),'cached_comparisons':len(cached)})
            print(json.dumps({'status':'completed','fresh_fits':len(fresh),'results':str(output/'ablation_results.zip')}),flush=True)
        except BaseException:
            tracking.console_line(traceback.format_exc());raise
    return output


def run_from_args(args, source):
    recipe=read(args.recipe);plan=read(PROJECT_ROOT/recipe['plan'])
    kaggle=Path('/kaggle/working').is_dir()
    input_root=Path(args.input_root) if args.input_root else (Path('/kaggle/input') if kaggle else PROJECT_ROOT/'dataset')
    if args.preflight_only:
        prepared=prepare_inputs(plan,input_root)
        print(json.dumps({'status':'preflight_passed','counts':prepared[1]['counts'],'model_fits_started':0}),flush=True)
        return
    return run(plan,input_root,
               Path(args.output_root) if args.output_root else (Path('/kaggle/working/runs') if kaggle else PROJECT_ROOT/'runs'),
               source_identity=source)
