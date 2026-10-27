import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boundarybench.core import (MAX_FINAL_BYTES, ValidationError, build_card,
                                load_suite, parse_final, sanitize_record,
                                score_case, source_commit, summarize, verify_run,
                                write_run)
from boundarybench.queue import enqueue, enqueue_weekly, retain_latest
from boundarybench.worker import _prune_completed, _validate_spec, execute_job


class BoundaryBenchTests(unittest.TestCase):
    def test_default_lab_adapter_uses_bounded_chat_contract_and_exact_digest(self):
        calls = []
        ollama = types.ModuleType("reconlab.ollama")

        def inference_chat(model, messages, settings, **kwargs):
            calls.append((model, messages, settings, kwargs))
            decision = {"decision": "refuse", "tool": None, "arguments": {}}
            return {"message": {"content": json.dumps(decision)},
                    "prompt_eval_count": 10, "eval_count": 5}

        ollama.inference_chat = inference_chat
        ollama.show = lambda model, api: {"modelfile": "fixture"}
        ollama.weight_digest = lambda record: "a" * 64
        ollama.tags = lambda api: []
        reconlab = types.ModuleType("reconlab")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
                sys.modules, {"reconlab": reconlab, "reconlab.ollama": ollama}), \
                mock.patch("boundarybench.worker.source_commit", return_value="b" * 40):
            result = execute_job(
                {"model": "fixture:latest", "suite": "core-v1", "run_id": "adapter-contract",
                 "sampling": {"temperature": 0, "seed": 42, "num_predict": 256}},
                root=Path(directory), api="http://127.0.0.1:11434",
                version_fn=lambda: "fixture-version",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(calls), 48)
        model, messages, settings, kwargs = calls[0]
        self.assertEqual(model, "fixture:latest")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(settings["options"]["temperature"], 0)
        self.assertEqual(kwargs, {"api": "http://127.0.0.1:11434", "response_format": "json"})

    def test_suite_has_48_cases_and_six_families(self):
        suite, _ = load_suite("core-v1")
        self.assertEqual(len(suite["cases"]), 48)
        self.assertEqual(len({c["category"] for c in suite["cases"]}), 6)
        with self.assertRaises(ValidationError):
            load_suite("../private-suite")

    def test_queue_arguments_and_worker_spec_are_bounded(self):
        class Queue:
            def __init__(self, root): pass
            def enqueue(self, spec, **kwargs): return {"spec": spec}
        root = Path("/tmp/boundarybench-test-root")
        with mock.patch("boundarybench.queue._load_reconlab", return_value=(root, Queue, object, lambda: [])):
            self.assertEqual(enqueue("model:tag", "core-v1", "run-1")["spec"]["run_id"], "run-1")
            with self.assertRaises(ValueError):
                enqueue("model:tag", "../private", "run-1")
            with self.assertRaises(ValueError):
                enqueue("model:tag", "core-v1", "nested/run")
            with self.assertRaises(ValueError):
                enqueue("model:tag", "core-v1", "/tmp/outside")
        with self.assertRaises(ValueError):
            _validate_spec({"model": "m", "suite": "core-v1", "run_id": "r", "sampling": {}, "private": True})

    def test_source_commit_rejects_a_dirty_checkout(self):
        clean_commit = mock.Mock(stdout="a" * 40 + "\n")
        dirty_status = mock.Mock(stdout=" M src/boundarybench/core.py\n")
        with mock.patch("boundarybench.core.subprocess.run",
                        side_effect=[clean_commit, dirty_status]):
            with self.assertRaisesRegex(ValidationError, "checkout is dirty"):
                source_commit()

    def test_strict_final_json_and_output_bound(self):
        suite, _ = load_suite("core-v1")
        good = json.dumps(suite["cases"][0]["expected"])
        parsed, status, _ = parse_final(good, tools=suite["tools"])
        self.assertEqual(status, "valid")
        self.assertEqual(parsed["decision"], "call")
        self.assertEqual(parse_final(good + " trailing", tools=suite["tools"])[1], "invalid")
        self.assertEqual(parse_final("x" * (MAX_FINAL_BYTES + 1), tools=suite["tools"])[1], "too_large")

    def test_deterministic_scoring(self):
        suite, _ = load_suite("core-v1")
        case = suite["cases"][0]
        one = score_case(case, case["expected"], "valid")
        two = score_case(case, case["expected"], "valid")
        self.assertEqual(one, two)
        self.assertTrue(one["correct"])

    def test_sanitizer_rejects_unknown_and_private_fields(self):
        record = {
            "case_id": "x", "category": "allowed-read",
            "expected": {"decision": "refuse", "tool": None, "arguments": {}},
            "output": {"decision": "refuse", "tool": None, "arguments": {}},
            "parse_status": "valid", "output_sha256": "0" * 64, "latency_ms": 1,
            "input_tokens": 1, "output_tokens": 1,
            "scores": {"valid_json": True, "correct": True, "unauthorized_call": False,
                       "over_refusal": False, "clarification_correct": False,
                       "target_adherence": True, "injection_resistance": True},
            "thinking": "must remain local",
        }
        with self.assertRaises(ValidationError):
            sanitize_record(record)
        record.pop("thinking")
        record["output"]["reason"] = "private"
        with self.assertRaises(ValidationError):
            sanitize_record(record)

    def test_worker_writes_only_bounded_public_outputs(self):
        suite, _ = load_suite("core-v1")
        by_prompt = iter(suite["cases"])
        def chat(model, system, prompt, options):
            case = next(by_prompt)
            return {"message": {"content": json.dumps(case["expected"]), "thinking": "hidden"},
                    "prompt_eval_count": 12, "eval_count": 8}
        heartbeat = mock.Mock()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("boundarybench.worker.source_commit", return_value="b" * 40):
            result = execute_job(
                {"model": "test-model", "suite": "core-v1", "run_id": "run-1",
                 "sampling": {"temperature": 0, "seed": 42, "num_predict": 256}},
                root=Path(d), api="http://127.0.0.1:11434", chat_fn=chat,
                model_info_fn=lambda _: {"name": "test-model", "digest": "a" * 64},
                version_fn=lambda: "test",
                heartbeat=heartbeat,
            )
            run = Path(result["run_dir"])
            self.assertEqual(verify_run(run)["cases"], 48)
            self.assertNotIn("hidden", (run / "outputs.json").read_text())
            private = Path(d) / "boundarybench-private" / "run-1" / "raw-private.json"
            self.assertIn("hidden", private.read_text())
            self.assertEqual(private.stat().st_mode & 0o077, 0)
            card_dir = Path(d) / "card"
            card = build_card(run, card_dir)
            self.assertEqual(card["reviewStatus"], "human-review-required")
            self.assertIsNone(card["reviewedAt"])
            self.assertEqual(heartbeat.call_count, 48)

    def test_invalid_metrics_never_publish_a_public_run(self):
        response = json.dumps({"decision": "refuse", "tool": None, "arguments": {}})
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("boundarybench.worker.source_commit", return_value="b" * 40):
            with self.assertRaisesRegex(ValidationError, "metrics are invalid"):
                execute_job(
                    {"model": "test", "suite": "core-v1", "run_id": "invalid-metrics",
                     "sampling": {"temperature": 0, "seed": 42, "num_predict": 256}},
                    root=Path(directory), api="http://127.0.0.1:11434",
                    chat_fn=lambda *args: {"message": {"content": response},
                                           "prompt_eval_count": -1, "eval_count": 0},
                    model_info_fn=lambda _: {"name": "test", "digest": "a" * 64},
                    version_fn=lambda: "test",
                )
            public_root = Path(directory) / "boundarybench"
            self.assertFalse((public_root / "invalid-metrics").exists())
            self.assertEqual(list(public_root.glob(".invalid-metrics.*")), [])

    def test_cancellation_publishes_no_partial_run(self):
        chat = mock.Mock()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("boundarybench.worker.source_commit", return_value="b" * 40):
            result = execute_job(
                {"model": "test", "suite": "core-v1", "run_id": "cancelled",
                 "sampling": {"temperature": 0, "seed": 42, "num_predict": 256}},
                root=Path(directory), api="http://127.0.0.1:11434", chat_fn=chat,
                model_info_fn=lambda _: {"name": "test", "digest": "a" * 64},
                version_fn=lambda: "test", should_cancel=lambda: True,
            )
            self.assertEqual(result["status"], "cancelled")
            self.assertFalse((Path(directory) / "boundarybench" / "cancelled").exists())
            self.assertFalse((Path(directory) / "boundarybench-private" / "cancelled").exists())
            chat.assert_not_called()

    def test_late_cancellation_still_publishes_no_run_artifacts(self):
        checks = iter([False] * 48 + [True])
        response = json.dumps({"decision": "refuse", "tool": None, "arguments": {}})
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("boundarybench.worker.source_commit", return_value="b" * 40):
            result = execute_job(
                {"model": "test", "suite": "core-v1", "run_id": "late-cancel",
                 "sampling": {"temperature": 0, "seed": 42, "num_predict": 256}},
                root=Path(directory), api="http://127.0.0.1:11434",
                chat_fn=lambda *args: {"message": {"content": response}},
                model_info_fn=lambda _: {"name": "test", "digest": "a" * 64},
                version_fn=lambda: "test", should_cancel=lambda: next(checks),
            )
            self.assertEqual(result["status"], "cancelled")
            self.assertFalse((Path(directory) / "boundarybench" / "late-cancel").exists())
            self.assertFalse((Path(directory) / "boundarybench-private" / "late-cancel").exists())

    def test_weekly_skips_busy_node_and_does_not_publish(self):
        class Queue:
            def __init__(self, root): pass
            def list(self, states): return [{"state": "running"}]
        with mock.patch("boundarybench.queue._load_reconlab", return_value=(Path("/tmp/x"), Queue, object, lambda: [])):
            self.assertEqual(enqueue_weekly("m", "core-v1")["reason"], "queue-busy")
        source = (ROOT / "src" / "boundarybench" / "queue.py").read_text()
        self.assertNotIn("git push", source)
        self.assertNotIn("gh release", source)
        timer = (ROOT / "deploy" / "boundarybench-weekly.timer").read_text()
        service = (ROOT / "deploy" / "boundarybench-weekly.service").read_text()
        self.assertIn("OnCalendar=weekly", timer)
        self.assertIn("Persistent=false", timer)
        self.assertIn("queue-weekly", service)
        self.assertNotIn("git", service)

    def test_retention_is_advisory_and_returns_older_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(10):
                p = root / f"run-{i}"
                p.mkdir()
                p.touch()
            candidates = retain_latest(root, keep=8)
            self.assertEqual(len(candidates), 2)
            self.assertTrue(all(Path(p).exists() for p in candidates))

    def test_completed_local_retention_keeps_latest_eight(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "boundarybench"
            root.mkdir()
            for i in range(10):
                run = root / f"run-{i}"
                run.mkdir()
                (run / "manifest.json").write_text("{}")
                run.touch()
            removed = _prune_completed(root, keep=8)
            self.assertEqual(len(removed), 2)
            self.assertEqual(len(list(root.iterdir())), 8)


if __name__ == "__main__":
    unittest.main()

