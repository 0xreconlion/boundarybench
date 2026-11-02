from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path


def _load_reconlab():
    checkout = os.environ.get("BOUNDARYBENCH_LAB_CONSOLE")
    if not checkout:
        raise RuntimeError("BOUNDARYBENCH_LAB_CONSOLE must point to the private lab-console checkout")
    path = Path(checkout).resolve()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from reconlab.config import DEFAULT_ROOT
    from reconlab.jobqueue import JobQueue
    from reconlab.labdb import LabDB
    from reconlab.ollama import loaded
    return DEFAULT_ROOT, JobQueue, LabDB, loaded


def enqueue(model: str, suite: str, out_name: str, num_predict: int = 256) -> dict:
    """Enqueue one bounded run.

    `num_predict` defaults to 256, the original fixed value, so existing
    behaviour is unchanged unless a caller asks otherwise.

    Why it is overridable at all: measured 2026-11-02, a reasoning model spends
    the whole 256-token budget on its preamble and never emits the closing JSON,
    so every case scores `invalid` and the run reads as a total contract failure.
    `heretic:4b` returned output_tokens 256/256 on 48 of 48 cases and
    `qwen3.5:9b` on 40 of 48, while `llama3.1:8b` never hit the cap at all
    (median 37). Attributing that to the model rather than the budget would be
    blaming the subject for a harness limit, so the budget has to be a variable
    the operator can set and the run can record.
    """
    # Validate caller input before touching the environment or the queue, so a
    # bad budget is rejected on its own terms rather than behind an unrelated
    # RuntimeError about operator configuration.
    if not isinstance(num_predict, int) or isinstance(num_predict, bool):
        raise ValueError("num_predict must be an integer")
    if not 1 <= num_predict <= 8192:
        raise ValueError("num_predict must be between 1 and 8192")
    default_root, JobQueue, _, _ = _load_reconlab()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/+:\-]{0,199}", model):
        raise ValueError("model identity is invalid")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", suite):
        raise ValueError("suite must be one committed suite id")
    requested = Path(out_name)
    if requested.is_absolute():
        resolved = requested.resolve()
        if resolved.parent != Path(default_root).resolve() / "boundarybench":
            raise ValueError("absolute out must be directly inside the lab BoundaryBench result root")
        run_id = resolved.name
    else:
        if len(requested.parts) != 1:
            raise ValueError("relative out must be one run directory name")
        run_id = requested.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", run_id):
        raise ValueError("out must name one bounded run directory")
    spec = {"model": model, "suite": suite, "run_id": run_id,
            "sampling": {"temperature": 0, "seed": 42, "num_predict": num_predict}}
    return JobQueue(default_root).enqueue(spec, kind="boundarybench", planned_units=48)


def enqueue_weekly(model: str, suite: str) -> dict:
    default_root, JobQueue, LabDB, loaded = _load_reconlab()
    queue = JobQueue(default_root)
    if queue.list(states=("queued", "waiting_resource", "running", "cancel_requested")):
        return {"status": "skipped", "reason": "queue-busy"}
    db = LabDB(default_root)
    try:
        if db.row("SELECT slot FROM gpu_lease WHERE slot=1"):
            return {"status": "skipped", "reason": "gpu-leased"}
    finally:
        db.close()
    if loaded():
        return {"status": "skipped", "reason": "gpu-resident-model"}
    run_id = time.strftime("weekly-%Y%m%dT%H%M%SZ", time.gmtime())
    result = enqueue(model, suite, run_id)
    return {"status": "queued", "job_id": result["job_id"], "run_id": run_id}


def retain_latest(root: Path, keep: int = 8) -> list[str]:
    # This helper returns candidates only. Deletion remains an operator action.
    runs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in runs[keep:]]

