"""Kaggle augmentation entrypoint: bounded generation and dataset export only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from model.runtime import PROJECT_ROOT, resolve_output, write_json
from tools.runtime import source_identity
from tools.run_experiment import resolve_inputs
from tools.experiment_log import record
from augmentation.config import load_recipe
from augmentation.data import baselines
from augmentation.inference import native_audit
from augmentation.worker import ProcessEngine
from augmentation.journal import Journal
from augmentation.pipeline import run_pipeline
from augmentation.recovery import load_payload, resolve_payload, run_recovery
from augmentation.review import digest


def prepare_inputs(recipe, input_root, work):
    # task1 input resolution imports only the three training files. The attached
    # canonical archive may contain public_test.json, but its contents are unread.
    packet = resolve_payload(input_root, work, recipe) if recipe.get('recovery') else None
    # Scope extracted uploads to the verified packet's own folder, so an older
    # attached dataset with the same baseline filenames does not create ambiguity.
    if packet is not None and all((packet.parent / name).is_file() for name in ('VGEC 2026_v1.json', 'VGEC_classifier.json', 'VGEC_balanced.json')):
        input_root = packet.parent
    adapted = {'mode':'task1', 'data_archive':recipe['data']['archive'], 'data_sha256':recipe['data']['sha256']}
    inputs = resolve_inputs(adapted, Path(input_root), Path(work) / 'inputs')
    raw, info = baselines(inputs, recipe)
    from model.data import load_classifier_split
    load_classifier_split(inputs['original'], inputs['classification'], 0, 42)
    if raw['classification']['original_sha256'] != raw['correction']['original_sha256']:
        raise ValueError('Classifier/corrector baselines have different original populations')
    if packet is not None:
        inputs['recovery'] = packet
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'configs/augmentation.json'))
    parser.add_argument('--input-root')
    parser.add_argument('--output-dir')
    parser.add_argument('--resume-run', help='Resume an unchanged private run journal; no notebook switch')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.resume_run and args.output_dir: parser.error('Use either resume-run or output-dir')
    recipe = load_recipe(args.config)
    source = source_identity(PROJECT_ROOT)
    if source['commit'] is None or source['dirty']:
        raise ValueError('Kaggle augmentation requires committed clean source')
    from importlib.metadata import version
    if tuple(int(x) for x in version('transformers').split('.')[:2]) < (5,6):
        raise RuntimeError('Augmentation requires Transformers>=5.6 in its isolated environment')
    kaggle = Path('/kaggle/working').is_dir()
    input_root = Path(args.input_root) if args.input_root else Path('/kaggle/input') if kaggle else PROJECT_ROOT / 'dataset'
    root = Path('/kaggle/working/augmentation-runs') if kaggle else PROJECT_ROOT / recipe['run']['output_root']
    work = Path(args.resume_run).resolve() if args.resume_run else resolve_output({'run':{'name':recipe['name'], 'output_root':str(root)}}, args.output_dir)
    if not args.resume_run and work.exists(): raise ValueError('New augmentation output must not already exist')
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    print(f'VGEC_RUN_DIR:{work}', flush=True)
    if args.resume_run:
        manifest = json.loads((work / 'run_identity.json').read_text())
        if manifest['source'] != source or manifest['recipe'] != recipe:
            raise ValueError('Resume source/recipe differs; start a separately documented run')
        inputs = {role: Path(entry['path']) for role, entry in manifest['inputs'].items()}
        if any(not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != manifest['inputs'][role]['sha256'] for role,p in inputs.items()):
            raise ValueError('Resume baseline inputs are missing or changed')
    else:
        inputs = prepare_inputs(recipe, input_root, work)
        manifest = {'source':source, 'recipe':recipe, 'inputs':{role:{'path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for role,p in inputs.items()}}
        write_json(work / 'run_identity.json', manifest)
        write_json(work / 'config.json', recipe)
    identity = {'source':source, 'inputs':{role:entry['sha256'] for role,entry in manifest['inputs'].items()}}
    payload = load_payload(inputs['recovery'], recipe['recovery']['sha256'], recipe, inputs) if recipe.get('recovery') else None
    if args.preflight_only or not (work / 'pipeline_manifest.json').is_file():
        audits = {role:native_audit(settings, recipe['run']['device']) for role,settings in recipe['models'].items()
                  if payload is None or role == 'reviewer'}
        write_json(work / 'preflight.json', audits)
    if args.preflight_only:
        print('All baseline, native-size and placement checks passed; no weights loaded or data generated', flush=True)
        return
    if (work / 'pipeline_manifest.json').is_file():
        result = json.loads((work / 'pipeline_manifest.json').read_text())
        if result.get('status') in ('completed','completed_no_accepted_augmentation', 'reviewer_check_failed'):
            with Journal(work, recipe, identity, ProcessEngine): pass
            if result['status'] == 'reviewer_check_failed':
                print('Saved reviewer check failed; no inference repeated. Inspect reviewer_check.json.')
                return
            for entry in result['files'].values():
                if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['sha256']:
                    raise ValueError('Completed dataset file changed')
            if hashlib.sha256(Path(result['archive']['path']).read_bytes()).hexdigest() != result['archive']['sha256']:
                raise ValueError('Completed augmentation archive changed')
            print('Completed unchanged run already exported; no generation repeated', flush=True)
            return
    record(work, 'started', experiment=recipe['name'], commit=source['commit'], seed=42)
    try:
        with Journal(work, recipe, identity, ProcessEngine) as caller:
            if payload is None:
                result = run_pipeline(inputs, recipe, caller, work, source)
            else:
                provenance = {'payload_sha256': recipe['recovery']['sha256'], 'source': payload['source'],
                              'source_recipe_sha256': digest(payload['recipe'])}
                result = run_recovery(inputs, recipe, caller, work, source, payload['candidates'], provenance)
        record(work, 'reviewer_check_failed' if result['status'] == 'reviewer_check_failed' else 'completed',
               experiment=recipe['name'], accepted_error_pairs=result['accepted_error_pairs'],
               augmentation_rows=result['new_rows_per_stage'], archive_sha256=result.get('archive', {}).get('sha256'))
    except BaseException as error:
        write_json(work / 'failure.json', {'status':'failed', 'error_type':type(error).__name__, 'error':str(error), 'source':source})
        record(work, 'failed', experiment=recipe['name'], error_type=type(error).__name__)
        raise


if __name__ == '__main__': main()
