"""Build one private Kaggle upload: exact quality rows and cached dev predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from model.data import load_classifier_split
from model.runtime import PROJECT_ROOT, write_json
from model.tracking import records_identity
from tools.run_quality_ablations import FORMAT, validate_rows, cached_comparisons


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(data, saved_run, output):
    data=Path(data);saved_run=Path(saved_run);output=Path(output)
    if output.exists():raise FileExistsError('Use a new package directory to preserve earlier artifacts')
    output.mkdir(parents=True)
    for name in ('data','baseline','provenance'):(output/name).mkdir()
    train,info=load_classifier_split(data/'VGEC 2026_v1.json',data/'VGEC_classifier.json',.2,42)
    dev=info['validation_records'];originals=[r for r in train if type(r['id']) is int]
    validate_rows(train,originals,dev)
    if (len(train),len(originals),len(dev))!=(1128,321,78):raise ValueError('Expected frozen quality training and78dev')
    for name,rows in [('quality',train),('organizer',originals),('validation',dev)]:write_json(output/'data'/(name+'.json'),rows)
    for name in ('qwen7b','sailor7b','mistral7b'):
        rows=json.loads((saved_run/'classification/members'/name/'predictions.json').read_text())
        keys=('id','source','target_class','target_correction','predicted_class','probabilities')
        write_json(output/'baseline'/(name+'.json'),[{k:r[k] for k in keys} for r in rows])
    rows=json.loads((saved_run/'correction/predictions.json').read_text())
    keys=('id','source','target_class','target_correction','predicted_class','predicted_correction')
    write_json(output/'baseline/correction.json',[{k:r[k] for k in keys} for r in rows])
    provenance=output/'provenance'
    for name in ['VGEC 2026_v1.json','VGEC_classifier.json']:shutil.copy2(data/name,provenance/name)
    (output/'NOTICE.txt').write_text('PRIVATE RESEARCH DATA — VGEC 2026 consortium corpus agreement applies. Do not publish this dataset or derivatives. Original labels and targets are unchanged. Quality augmentation includes 32 UD Vietnamese VTB training sentences, CC BY-SA 4.0, https://github.com/UniversalDependencies/UD_Vietnamese-VTB/tree/3c971fd1c5d03561e00b267d3d0057b81a9bea84 . No public/private test inputs, model weights or nested archives are included. Development is repeatedly reused selection data. Cached probabilities/corrections are historical outputs, not fresh-model or private scores.\n')
    reports,corrections=cached_comparisons(output,dev)
    if abs(reports['quality_112_phrase']['macro_f1']-0.5727738927738928)>1e-12:raise ValueError('Frozen quality vote does not reproduce its recorded development score')
    if abs(corrections['all']['gleu_plus']-0.6057518271992086)>1e-12:raise ValueError('Frozen corrector does not reproduce its recorded development score')
    manifest={'format':FORMAT,'quality_source_commit':'338293f90e664803ee3dde12e82dd5d4d5601d43',
              'counts':{'quality':len(train),'organizer':len(originals),'validation':len(dev)},
              'records':{'quality':records_identity(train),'organizer':records_identity(originals),'validation':records_identity(dev)},
              'roles':{'quality':'training-only','organizer':'training-only eligible original subset','validation':'selection/evaluation-only','baseline':'historical development predictions; never training'},
              'files':{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    write_json(output/'manifest.json',manifest)
    archive=output.parent/(output.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.rglob('*')):
            if path.is_file():handle.write(path,str(path.relative_to(output)))
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None and not any(x.endswith(('.zip','.tar','.gz')) for x in handle.namelist())
    proof={'format':FORMAT,'archive_name':archive.name,'manifest_sha256':sha(output/'manifest.json'),
           'archive_sha256_for_download_only':sha(archive),'counts':manifest['counts'],
           'quality_source_commit':manifest['quality_source_commit'],'historical_development_macro_f1':reports['quality_112_phrase']['macro_f1'],
           'historical_development_gleu':corrections['all']['gleu_plus']}
    return archive,proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--saved-run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--proof',default=str(PROJECT_ROOT/'configs/quality_ablation_package.json'))
    args=parser.parse_args();archive,proof=build(args.data,args.saved_run,args.output);write_json(args.proof,proof)
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'counts':proof['counts']}))


if __name__=='__main__':main()
