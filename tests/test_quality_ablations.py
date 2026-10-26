import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import contextlib
import io
import sys
import types

from model.classification import _architecture, _joint_log_probabilities, _weighted_flat_loss
from tools.run_quality_ablations import find_package, validate_rows, study_config, FORMAT


class ObjectiveTests(unittest.TestCase):
    def test_flat_composes_softmax_and_preserves_partial_window_gradients(self):
        import torch
        x = torch.tensor([[.1, .2, -.7, .4, 1.], [-.5, .2, .9, .6, -.1]], requires_grad=True)
        y = torch.tensor([0, 3]); w = torch.tensor([.5, 1., 1.5, 2., 2.5])
        torch.testing.assert_close(_joint_log_probabilities(x, 'flat').exp(), x.softmax(-1))
        loss = _weighted_flat_loss(x, y, w)
        expected = (torch.nn.functional.cross_entropy(x, y, reduction='none') * w[y]).mean()
        torch.testing.assert_close(loss, expected)
        pieces = sum(_weighted_flat_loss(x[i:i+1], y[i:i+1], w, 2) for i in range(2))
        torch.testing.assert_close(torch.autograd.grad(loss, x, retain_graph=True)[0], torch.autograd.grad(pieces, x)[0])

    def test_hierarchy_metadata_default_is_unchanged(self):
        self.assertEqual(_architecture()['subtype_loss_weight'], 2.)
        self.assertEqual(_architecture('flat')['name'], 'flat_error_type')
        with self.assertRaises(ValueError): _architecture('unknown')


class ContractTests(unittest.TestCase):
    def test_launcher_returns_ablation_reports_without_submission_paths(self):
        from tools import notebook
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/'pipeline_manifest.json').write_text(json.dumps({'mode':'quality_ablations'}))
            (root/'ablation_summary.json').write_text(json.dumps({'fresh_training':{'quality_flat':{'classification':{'macro_f1':.4,'micro_f1':.6}}}}))
            class Finished:
                stdout=['VGEC_RUN_DIR:'+str(root)+'\n'];args=[]
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def wait(self):return 0
            shown=[];display=types.ModuleType('IPython.display');display.FileLink=str;display.display=shown.append
            with patch.object(notebook.subprocess,'Popen',return_value=Finished()),patch.dict(sys.modules,{'IPython.display':display}),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(notebook.launch(Path(__file__).resolve().parents[1],sys.executable),root)
            self.assertEqual(len(shown),4)
            self.assertTrue(shown[-1].endswith('ablation_results.zip'))

    def test_preflight_never_starts_training(self):
        from tools import run_quality_ablations as module
        root=Path(__file__).resolve().parents[1]
        args=SimpleNamespace(recipe=str(root/'configs/experiment.json'),input_root='/tmp/input',output_root='/tmp/output',preflight_only=True)
        prepared=(Path('/tmp/package'),{'counts':{'quality':1128,'organizer':321,'validation':78}},[],[],[],{}, {})
        with patch.object(module,'run') as train, patch.object(module,'prepare_inputs',return_value=prepared,create=True):
            module.run_from_args(args,{'commit':'test'})
        train.assert_not_called()

    def test_root_tracking_accepts_live_metrics_for_each_ablation_arm(self):
        from model.tracking import TrackingRun
        tracking=TrackingRun();tracking.run=MagicMock()
        for arm in ['quality_hierarchical','organizer_hierarchical','quality_flat']:
            tracking.progress({'stage':arm+'/fit','prefix':'train','optimizer_step':2,'metrics':{'loss':1.25}})
            self.assertEqual(tracking.run.log.call_args.args[0][arm+'/fit/train/loss'],1.25)
        self.assertEqual(tracking.run.log.call_count,3)

    def test_nested_extracted_dataset_discovery_rejects_changed_payload(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'weird upload name'/'nested'/'package';root.mkdir(parents=True)
            p=root/'data.json';p.write_text('[]')
            m=root/'manifest.json';m.write_text(json.dumps({'format':FORMAT,'files':{'data.json':hashlib.sha256(p.read_bytes()).hexdigest()}}))
            proof=hashlib.sha256(m.read_bytes()).hexdigest()
            self.assertEqual(find_package(Path(t),proof)[0],root)
            p.write_text('[1]')
            with self.assertRaises(ValueError):find_package(Path(t),proof)

    def test_manifest_cannot_escape_dataset_root(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'dataset';root.mkdir();(Path(t)/'outside.json').write_text('[]')
            m=root/'manifest.json';m.write_text(json.dumps({'format':FORMAT,'files':{'../outside.json':hashlib.sha256(b'[]').hexdigest()}}))
            with self.assertRaises(ValueError):find_package(root,hashlib.sha256(m.read_bytes()).hexdigest())

    def test_only_requested_arm_fields_change(self):
        root = Path(__file__).resolve().parents[1]
        base = json.loads((root/'configs/quality_ablation_classification.json').read_text())
        plan = {'optimizer_steps': 705, 'evaluation_steps': 141}
        a = study_config(base, {'name':'quality_hierarchical','objective':'hierarchical','dataset':'quality'}, plan, Path('/tmp/a'))
        b = study_config(base, {'name':'quality_flat','objective':'flat','dataset':'quality'}, plan, Path('/tmp/b'))
        self.assertEqual(a['model'], b['model']); self.assertEqual(a['lora'], b['lora'])
        self.assertFalse(a['model']['load_in_4bit'])
        self.assertEqual(a['training']['args'], b['training']['args'])
        self.assertEqual(a['training']['args']['max_steps'], 705)
        self.assertEqual(a['training']['evaluation_steps'], 141)

    def test_cross_stage_validation_overlap_and_duplicate_ids_fail(self):
        rows=[{'id':i,'source':f'source {i}','label':i,'target':f'target {i}'} for i in range(5)]
        dev=[{**x,'id':x['id']+10,'source':x['source']+' dev','target':x['target']+' dev'} for x in rows]
        validate_rows(rows, rows, dev)
        with self.assertRaises(ValueError): validate_rows(rows, rows, rows)
        with self.assertRaises(ValueError): validate_rows(rows+[rows[0]], rows, dev)


if __name__ == '__main__': unittest.main()
