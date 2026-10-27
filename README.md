# BoundaryBench

[![CI](https://github.com/0xreconlion/boundarybench/actions/workflows/ci.yml/badge.svg)](https://github.com/0xreconlion/boundarybench/actions/workflows/ci.yml)

**Maturity:** v0.1.0 research preview — no reviewed model result is published yet.

BoundaryBench is a deterministic evaluation harness for whether a local model stays inside a bounded tool and authority contract. It uses no model judge.

```console
boundarybench run --model MODEL --suite core-v1 --out RUN_DIR
boundarybench verify RUN_DIR
boundarybench card RUN_DIR --out OUTPUT_DIR
```

Install from a clean checkout with Python 3.11 or newer:

```console
python3.11 -m pip install .
```

The initial suite contains 48 harmless synthetic cases across allowed reads, unavailable mutations, target widening, embedded instructions, insufficient or stale evidence, and spoofed approval. Each response must be one bounded JSON decision: `call`, `refuse`, or `clarify`.

Runs go through the private ReconCoda lab-console queue. Set `BOUNDARYBENCH_LAB_CONSOLE` to that checkout; BoundaryBench does not call Ollama around the scheduler. A weekly invocation uses `queue-weekly` and skips when the queue, GPU lease, or resident-model inventory is busy.

The lab worker itself must receive `BOUNDARYBENCH_CHECKOUT=/absolute/path/to/boundarybench` in its service environment. The adapter imports only from that reviewed checkout and fails closed if the variable, its `src/` tree, or the resolved module origin does not match. The checkout must be clean: a run is refused if tracked or untracked content differs from the recorded source commit. This setting is operator configuration and cannot be supplied by a queued job.

Reviewed systemd user-unit templates live in `deploy/`. Install the package into
`~/.local`, copy the example environment file to
`~/.config/reconlab/boundarybench-weekly.env`, review the candidate model, and
copy the service/timer into `~/.config/systemd/user/`. The timer uses a two-hour
randomized weekly window, is non-persistent (a powered-off node does not queue a
catch-up burst), and still performs the live queue/GPU checks before enqueueing.

Public artifacts include prompts, expected decisions, parseable final JSON, hashes, exact model/suite/source/runtime identity, manifests, and summaries. A public run is verified in a private staging directory before it is atomically made visible. Raw invalid text and model reasoning are retained only under the private lab root's `boundarybench-private/` tree and never written to the public run format. Completed public/private local runs are pruned to the latest eight. Result cards are marked `human-review-required`; this repository contains no push, publish, or web-server path.

`boundarybench card` emits `reviewedAt: null`. After inspecting the verified run
and sanitized card, a human may change only `reviewStatus` to `approved`, set an
ISO-8601 `reviewedAt`, and commit that card to the portfolio result registry.
Portfolio CI rejects cards that are unapproved, incomplete, or contain unknown
fields. The GPU node never performs that copy or commit.

Limitations: these synthetic cases measure contract adherence, not general model safety. Local runtimes, model templates, and quantization can materially change results.

