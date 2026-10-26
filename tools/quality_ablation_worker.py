"""One process per fit or selected-adapter reload; no retained GPU model state."""
import argparse
import json
from pathlib import Path


def main():
    from model.runtime import load_config, write_json
    from model.tracking import TrackingRun
    from model.data import _validate_records
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],required=True)
    parser.add_argument('--phase',choices=['fit','verify'],required=True)
    args=parser.parse_args();root=Path(args.output_dir)
    if args.phase=='verify':
        from model.classifier_finalize import finalize
        with TrackingRun() as tracking:
            cfg=load_config(root/'config.json','classification')
            tracking.start(cfg,root/'verification','classification-verification',group=cfg.get('_tracking_group'))
            report=finalize(root,args.device);tracking.complete('classifier',report)
        return
    config=load_config(root/'config.json','classification')
    train=json.loads((root/'train.json').read_text());validation=json.loads((root/'validation.json').read_text())
    _validate_records(train);_validate_records(validation)
    from model.classification import _fit,_preprocessing
    with TrackingRun() as tracking:
        stats=_fit(config,train,validation,root/'classifier',root,args.device,42,_preprocessing(config),tracking=tracking)
        write_json(root/'training_complete.json',{**stats,'config':config,'train_count':len(train),
                                                 'validation_count':len(validation),'final_fit':False})


if __name__=='__main__':main()
