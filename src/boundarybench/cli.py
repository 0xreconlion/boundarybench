from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import ValidationError, build_card, verify_run
from .queue import enqueue, enqueue_weekly


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="boundarybench")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="enqueue a bounded run on the private lab-console GPU queue")
    run.add_argument("--model", required=True)
    run.add_argument("--suite", required=True)
    run.add_argument("--out", required=True)
    verify = commands.add_parser("verify", help="deterministically verify a run directory")
    verify.add_argument("run_dir", type=Path)
    card = commands.add_parser("card", help="create a sanitized card requiring human review")
    card.add_argument("run_dir", type=Path)
    card.add_argument("--out", required=True, type=Path)
    weekly = commands.add_parser("queue-weekly", help="enqueue only when the private GPU is idle")
    weekly.add_argument("--model", required=True)
    weekly.add_argument("--suite", default="core-v1")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            result = enqueue(args.model, args.suite, args.out)
        elif args.command == "verify":
            result = verify_run(args.run_dir)
        elif args.command == "card":
            result = build_card(args.run_dir, args.out)
        else:
            result = enqueue_weekly(args.model, args.suite)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
        print(f"boundarybench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

