"""Local experiment events with scalar provenance; corpus text never enters this log."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path



def record(work, status, **fields):
    event = {"at": datetime.now(timezone.utc).isoformat(), "status": status, **fields}
    event["work"] = str(work)
    paths = [Path(work) / "experiment_events.jsonl"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            fcntl.flock(output, fcntl.LOCK_EX)
            output.write(json.dumps(event, ensure_ascii=False) + "\n")
            fcntl.flock(output, fcntl.LOCK_UN)
    return event
