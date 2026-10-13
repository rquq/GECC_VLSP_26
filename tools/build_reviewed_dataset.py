"""Build private v4 bundles from eligible-teacher outputs and explicit review decisions."""
import argparse
import base64
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import zipfile

from model.data import (_counts, _id_key, _records_from_raw, _source_key, declares_correct_sentence,
                        load_classifier_split, load_records, load_training_split, split_records)
from model.reviewed_data import canonical_hash
from model.runtime import write_json
from tools.build_dataset import audit_record
from tools.local_teacher import MODEL, PARAMETERS, REVISION


def build(original, repair_candidates, pilot, decisions, public_test, output, legacy_correction=None):
    if output.exists():
        raise ValueError("Reviewed output must be new; never replace the original/pinned bundle")
    raw = json.loads(original.read_text())
    originals, validation = split_records(load_records(original), .2, 42)
    before = {_id_key(r["ID"]): r for r in raw}
    allowed = {_id_key(r["id"]) for r in originals}
    choice = json.loads(decisions.read_text())
    events = [json.loads(line) for line in (pilot/"calls.jsonl").read_text().splitlines()]
    manifest = json.loads((pilot/"manifest.json").read_text())
    if (manifest["model"] != MODEL or manifest["revision"] != REVISION or manifest["parameters"] != PARAMETERS
            or manifest["original_sha256"] != hashlib.sha256(original.read_bytes()).hexdigest()
            or len(events) > 144):
        raise ValueError("Pilot model/original/budget differs from approved provenance")
    generated = {e["key"]: e for e in events if e["kind"] == "generation"}
    verified = {e["key"]: e for e in events if e["kind"] == "blind_review"}
    parents = {e["key"]: e for e in map(json.loads,(pilot/"parents.jsonl").read_text().splitlines())}
    replacements, repairs = {}, []
    candidates = json.loads(repair_candidates.read_text())["candidate_changes"]
    for r in candidates:
        if r["id"] not in choice["accepted_repairs"]:
            continue
        key = _id_key(r["id"])
        check = verified["repair-"+str(r["id"])]["parsed"]
        after = copy.deepcopy(r["original"])
        after["Output"] = dict(Error_Type_Id=r["proposed_label"], **{"Fixed Sentence":r["proposed_full_correction"]})
        replacements[key] = after
        repairs.append(dict(before=r["original"],after=after,method=r["method"],evidence=r["evidence"],
                            status="accepted",verification=check,teacher_parameters=PARAMETERS,
                            teacher_revision=REVISION,review_evidence=choice["repair_evidence"][str(r["id"])]))
    errors=[]; metadata=[]
    original_clean = {_source_key(r["source"]):r["id"] for r in originals if r["label"] == 0}
    for key in choice["accepted_keys"]:
        event = generated[key]; value = event["parsed"]; check = verified[key]["parsed"]
        context = parents[key]
        if (value["label"] != context["label"] or check.get("source_label") != value["label"]
                or check.get("target_label") != 0 or check.get("single_error") is not True
                or check.get("meaning_preserved") is not True or check.get("required_edit") is not True
                or any(_id_key(p) not in allowed for p in context["parent_ids"])):
            raise ValueError("Accepted synthetic lacks blind subtype/target/meaning agreement")
        clean_id = original_clean.get(_source_key(value["target"]))
        if clean_id is None or before[_id_key(clean_id)]["Input"] != value["target"]:
            raise ValueError("This reviewed pilot requires exact original clean counterparts")
        row = dict(ID=key,Input=value["source"],Output=dict(Error_Type_Id=value["label"],
                                                        **{"Fixed Sentence":value["target"]}))
        audit = audit_record(_records_from_raw([row])[0],row)
        if audit["quarantine_reasons"]:
            raise ValueError("Accepted synthetic violates structural correction checks")
        errors.append(row)
        metadata.append(dict(id=key,kind="synthetic",parent_ids=context["parent_ids"],clean_id=clean_id,
            mechanism=context["mechanism"],family=f"qwen-minimal:{value['label']}:{context['mechanism']}",
            teacher=MODEL,teacher_revision=REVISION,teacher_parameters=PARAMETERS,seed=42,
            output_sha256=event["output_sha256"],review_status="accepted",blind_review=check,
            review_evidence=choice["synthetic_evidence"][key]))
    output.mkdir(parents=True,mode=0o700)
    legacy_rows=[];legacy_metadata=[];legacy_bytes=None
    if legacy_correction:
        _, legacy_validation, legacy_info=load_training_split(legacy_correction,.2,42)
        if legacy_validation != validation or legacy_info["original_sha256"] != hashlib.sha256(original.read_bytes()).hexdigest():
            raise ValueError("Preserved corrector augmentation has a different original/holdout")
        legacy_bytes=legacy_correction.read_bytes()
        legacy=json.loads(legacy_bytes)
        legacy_ids={_id_key(k) for k in legacy_info["provenance"]["selected_synthetic_ids"]}
        legacy_rows=[r for r in legacy["train"] if _id_key(r["ID"]) in legacy_ids]
        legacy_metadata=[r for r in legacy_info["provenance"]["train_rows"] if _id_key(r["id"]) in legacy_ids]
    original_digest=hashlib.sha256(original.read_bytes()).hexdigest()
    summaries={}
    for stage,name in (("classification","VGEC_classifier.json"),("correction","VGEC_balanced.json")):
        retained=[]; omitted=[]; provenance_rows=[]
        for r in originals:
            key=_id_key(r["id"]); row=copy.deepcopy(replacements.get(key,before[key]))
            record=_records_from_raw([row])[0]; audit=audit_record(record,row,original=True)
            reject=(stage=="correction" and bool(audit["quarantine_reasons"]) or
                    stage=="classification" and record["label"]!=0 and declares_correct_sentence(record["target"]))
            if reject:
                omitted.append(r["id"])
            else:
                retained.append(row)
                provenance_rows.append(dict(id=r["id"],kind="original",original_id=r["id"],audit_flags=audit["flags"]))
        stage_errors=copy.deepcopy(errors)+(copy.deepcopy(legacy_rows) if stage=="correction" else [])
        stage_metadata=metadata+(legacy_metadata if stage=="correction" else [])
        training=retained+stage_errors
        random.Random(42).shuffle(training)
        provenance=dict(stage=stage,original_snapshot=raw,original_snapshot_sha256=canonical_hash(raw),repairs=repairs,
            original_train_ids=[r["id"] for r in originals],retained_original_train_ids=[r["ID"] for r in retained],
            quarantined_ids=omitted,selected_synthetic_ids=[r["ID"] for r in stage_errors],train_rows=provenance_rows+stage_metadata,
            training_class_counts=_counts(_records_from_raw(training)),synthetic_class_counts=_counts(_records_from_raw(stage_errors)),
            balance_policy="reviewed_under_cap_2x_matched_controls",teacher_manifest=manifest,
            review_decisions_sha256=hashlib.sha256(decisions.read_bytes()).hexdigest(),validation_raw_unchanged=True)
        if stage=="correction" and legacy_bytes:
            provenance.update(preserved_legacy_bundle=base64.b64encode(legacy_bytes).decode(),
                              preserved_legacy_sha256=hashlib.sha256(legacy_bytes).hexdigest())
        bundle=dict(format="vgec-reviewed-v4",seed=42,validation_fraction=.2,original_sha256=original_digest,
                    train=training,validation=[before[_id_key(r["id"])] for r in validation],provenance=provenance)
        write_json(output/name,bundle)
        load_training_split(output/name,.2,42)
        if stage=="classification":
            load_classifier_split(original,output/name,.2,42)
        summaries[stage]=dict(rows=len(training),originals=len(retained),synthetics=len(stage_errors),
            quarantined=len(omitted),counts=provenance["training_class_counts"],sha256=hashlib.sha256((output/name).read_bytes()).hexdigest())
    (output/"VGEC 2026_v1.json").write_bytes(original.read_bytes())
    (output/"public_test.json").write_bytes(public_test.read_bytes())
    summary=dict(format="vgec-reviewed-manifest-v4",original_sha256=original_digest,holdout_rows=len(validation),
                 accepted_repairs=len(repairs),accepted_pairs=len(errors),bundles=summaries,
                 teacher=MODEL,revision=REVISION,parameters=PARAMETERS,private=True,redistribution_allowed=False,
                 mechanism_counts=dict(Counter(f"{r['blind_review']['source_label']}:{r['mechanism']}" for r in metadata)))
    write_json(output/"manifest.json",summary)
    archive=output/"VGEC26-reviewed-v4.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for name in ("VGEC 2026_v1.json","VGEC_classifier.json","VGEC_balanced.json","public_test.json","manifest.json"):
            z.write(output/name,name)
    for p in output.iterdir():
        p.chmod(0o600)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("original","repair-candidates","pilot","decisions","public-test","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--legacy-correction",type=Path,help="Preserve exact pinned v3 corrector augmentation")
    a=parser.parse_args()
    print(json.dumps(build(a.original,a.repair_candidates,a.pilot,a.decisions,a.public_test,a.output,a.legacy_correction),indent=2))


if __name__=="__main__":
    main()
