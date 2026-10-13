"""Reviewed data rejects unsupported edits, holdout changes and shortcut dominance."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from model.data import _counts, _records_from_raw, load_classifier_split, load_records, load_training_split, split_records
from model.reviewed_data import canonical_hash
from tools.build_reviewed_dataset import build
from tools.local_teacher import MODEL, PARAMETERS, REVISION


class ReviewedDataTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.original=self.root/"original.json"
        raw=[dict(ID=c*100+i,Input=f"original source {c} {i}.",Output=dict(Error_Type_Id=c,
              **{"Fixed Sentence":None if c==0 else f"corrected words {c} {i}."})) for c in range(5) for i in range(10)]
        self.original.write_text(json.dumps(raw,ensure_ascii=False))
        train,_=split_records(load_records(self.original),.2,42)
        identifier=next(r["id"] for r in train if r["label"]==2)
        before=next(r for r in raw if r["ID"]==identifier)
        before["Output"]["Fixed Sentence"]="(Câu này đúng)"
        self.original.write_text(json.dumps(raw,ensure_ascii=False))
        repair=dict(id=identifier,original=before,proposed_label=0,proposed_full_correction=before["Input"],
                    method="explicit_original_annotation",evidence=dict(annotation=before["Output"]["Fixed Sentence"]))
        self.write("repairs.json",dict(candidate_changes=[repair]))
        self.pilot=self.root/"pilot";self.pilot.mkdir()
        self.write("pilot/manifest.json",dict(model=MODEL,revision=REVISION,parameters=PARAMETERS,
                  original_sha256=hashlib.sha256(self.original.read_bytes()).hexdigest()))
        verification=dict(source_label=0,target_label=0,meaning_preserved=True,single_error=False,required_edit=False)
        self.write("pilot/calls.jsonl",dict(kind="blind_review",key="repair-"+str(identifier),parsed=verification))
        (self.pilot/"parents.jsonl").write_text("")
        self.write("decisions.json",dict(accepted_keys=[],accepted_repairs=[identifier],
                  repair_evidence={str(identifier):"Original explicit annotation and independent eligible review agree."},synthetic_evidence={}))
        self.write("public_test.json",[])
        self.output=self.root/"new"
        build(self.original,self.root/"repairs.json",self.pilot,self.root/"decisions.json",self.root/"public_test.json",self.output)
        self.path=self.output/"VGEC_classifier.json"
        self.bundle=json.loads(self.path.read_text())

    def write(self,name,value):
        (self.root/name).write_text(json.dumps(value,ensure_ascii=False)+"\n")

    def load(self,bundle):
        self.path.write_text(json.dumps(bundle,ensure_ascii=False))
        return load_classifier_split(self.original,self.path,.2,42)

    def test_under_cap_empty_pool_and_supported_clean_repair_load(self):
        train,info=self.load(self.bundle)
        self.assertEqual(len(train),40)
        self.assertEqual(sum(r["label"]==0 for r in train),9)
        self.assertEqual(info["format"],"vgec-reviewed-v4")
        _,validation,_=load_training_split(self.output/"VGEC_balanced.json",.2,42)
        self.assertEqual(validation,split_records(load_records(self.original),.2,42)[1])

    def test_changed_holdout_or_original_snapshot_is_rejected(self):
        for field in ("validation","snapshot"):
            b=copy.deepcopy(self.bundle)
            if field=="validation": b[field][0]["Input"]+=" altered"
            else:b["provenance"]["original_snapshot"][0]["Input"]+=" altered"
            with self.assertRaises(ValueError):self.load(b)

    def test_unsupported_relabel_or_teacher_disagreement_is_rejected(self):
        for mutation in ("label","verification","source"):
            b=copy.deepcopy(self.bundle);r=b["provenance"]["repairs"][0]
            if mutation=="label":r["after"]["Output"]["Error_Type_Id"]=3
            elif mutation=="verification":r["verification"]["source_label"]=2
            else:r["after"]["Input"]+=" changed"
            with self.assertRaises(ValueError):self.load(b)

    def test_foreign_parent_and_dominant_mechanism_are_rejected(self):
        b=copy.deepcopy(self.bundle);p=b["provenance"]
        cleans=[r for r in b["train"] if r["Output"]["Error_Type_Id"]==0][:5]
        for i,clean in enumerate(cleans):
            identifier=f"synthetic:{i}";target=clean["Input"]
            b["train"].append(dict(ID=identifier,Input=f"synthetic error words {i}.",Output=dict(Error_Type_Id=2,**{"Fixed Sentence":target})))
            p["selected_synthetic_ids"].append(identifier)
            p["train_rows"].append(dict(id=identifier,kind="synthetic",parent_ids=[clean["ID"]],clean_id=clean["ID"],
                teacher_parameters=PARAMETERS,teacher_revision=REVISION,output_sha256="a"*64,review_status="accepted",
                review_evidence="fixture review",family=f"family-{i}",mechanism=f"mechanism-{i}",
                blind_review=dict(source_label=2,target_label=0,single_error=True,meaning_preserved=True,required_edit=True)))
        p["training_class_counts"]=_counts(_records_from_raw(b["train"]))
        p["synthetic_class_counts"]={str(c):(5 if c==2 else 0) for c in range(5)}
        self.load(b)
        wrong=copy.deepcopy(b);wrong["provenance"]["train_rows"][-1]["parent_ids"]=[b["validation"][0]["ID"]]
        with self.assertRaisesRegex(ValueError,"parent/review"):self.load(wrong)
        wrong=copy.deepcopy(b)
        for m in wrong["provenance"]["train_rows"][-5:]:m["mechanism"]="same shortcut"
        with self.assertRaisesRegex(ValueError,"20 percent"):self.load(wrong)
        wrong=copy.deepcopy(b);wrong["provenance"]["train_rows"][-1]["blind_review"]["source_label"]=0
        with self.assertRaisesRegex(ValueError,"blind subtype"):self.load(wrong)


if __name__=="__main__":unittest.main()
