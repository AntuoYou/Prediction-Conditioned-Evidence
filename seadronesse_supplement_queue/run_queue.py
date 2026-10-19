#!/usr/bin/env python3
"""Run a JSON experiment queue with bounded GPU/service-slot concurrency.

Commands are argv arrays (never shell strings). Each task receives a dedicated
resource slot and its own log directory. This is an orchestration layer; it
does not implement a model runner or infer missing protocol parameters.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def endpoint_ready(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def render_argv(argv: list[str], values: dict[str, str]) -> list[str]:
    return [part.format_map(values) for part in argv]


def run_task(task: dict, slot: dict, output_root: pathlib.Path,
             repo_root: pathlib.Path, poll_seconds: float) -> dict:
    task_id = task["id"]
    task_dir = output_root / task_id
    task_dir.mkdir(parents=True, exist_ok=False)
    endpoint = slot.get("health_url", "")
    if endpoint:
        deadline = time.monotonic() + float(slot.get("ready_timeout_seconds", 0))
        while not endpoint_ready(endpoint):
            if time.monotonic() >= deadline:
                result = {"task_id": task_id, "status": "blocked_endpoint_not_ready",
                          "slot": slot["name"], "health_url": endpoint,
                          "finished_utc": utc_now()}
                (task_dir / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result
            time.sleep(poll_seconds)

    values = {
        "repo": str(repo_root),
        "task_dir": str(task_dir),
        "task_id": task_id,
        "run_id": task.get("run_id", task_id),
        "slot": slot["name"],
        "device": str(slot.get("device", "")),
        "url": str(slot.get("url", "")),
    }
    values.update({str(k): str(v).format_map(values)
                   for k, v in task.get("params", {}).items()})
    argv = render_argv(task["command"], values)
    env = os.environ.copy()
    env.update({str(k): str(v).format_map(values) for k, v in task.get("env", {}).items()})
    for k, v in slot.get("env", {}).items():
        env[str(k)] = str(v).format_map(values)
    started = utc_now()
    with (task_dir / "stdout.log").open("w", encoding="utf-8") as out, \
         (task_dir / "stderr.log").open("w", encoding="utf-8") as err:
        proc = subprocess.run(argv, cwd=repo_root, env=env, stdout=out,
                              stderr=err, check=False)
    result = {"task_id": task_id, "run_id": values["run_id"],
              "status": "completed" if proc.returncode == 0 else "failed",
              "return_code": proc.returncode, "slot": slot["name"],
              "device": values["device"], "url": values["url"],
              "argv": argv, "started_utc": started, "finished_utc": utc_now()}
    (task_dir / "status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=pathlib.Path, required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=1,
                        help="Cap concurrent tasks; set to verified free server slots.")
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")
    queue_path = args.queue.resolve()
    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        parser.error(f"repo root does not exist: {repo_root}")
    spec = json.loads(queue_path.read_text(encoding="utf-8"))
    slots = spec.get("slots", [])
    enabled = [task for task in spec.get("tasks", []) if task.get("enabled", False)]
    if not slots:
        parser.error("queue has no resource slots; add verified GPU/NPU service slots")
    if not enabled:
        print("No enabled tasks. Review queue.json and enable only after command/config review.")
        return 2
    if any(not task.get("command") for task in enabled):
        parser.error("every enabled task needs a non-empty argv command")
    ids = [task["id"] for task in enabled]
    if len(ids) != len(set(ids)):
        parser.error("enabled task ids must be unique")

    args.output.mkdir(parents=True, exist_ok=True)
    run_dir = args.output / dt.datetime.now().strftime("queue_%Y%m%d_%H%M%S")
    run_dir.mkdir()
    (run_dir / "queue_snapshot.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    results: list[dict] = []
    pending = iter(enabled)
    active: dict[cf.Future, tuple[dict, dict]] = {}
    available = list(slots[:args.max_parallel])
    stop = False
    with cf.ThreadPoolExecutor(max_workers=min(args.max_parallel, len(slots))) as pool:
        while not stop:
            while available and len(active) < args.max_parallel:
                try:
                    task = next(pending)
                except StopIteration:
                    break
                slot = available.pop(0)
                future = pool.submit(run_task, task, slot, run_dir, repo_root, args.poll_seconds)
                active[future] = (task, slot)
                print(f"START {task['id']} on {slot['name']}", flush=True)
            if not active:
                break
            done, _ = cf.wait(active, return_when=cf.FIRST_COMPLETED)
            for future in done:
                task, slot = active.pop(future)
                available.append(slot)
                result = future.result()
                results.append(result)
                print(f"{result['status'].upper()} {task['id']} ({slot['name']})", flush=True)
                if args.stop_on_error and result["status"] != "completed":
                    stop = True
        if stop:
            for task in pending:
                results.append({"task_id": task["id"], "status": "not_started_stop_on_error"})

    summary = {"created_utc": utc_now(), "repo_root": str(repo_root),
               "enabled_tasks": len(enabled), "results": results}
    (run_dir / "queue_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [r for r in results if r["status"] != "completed"]
    print(f"QUEUE_DONE completed={len(results)-len(failed)} noncompleted={len(failed)} output={run_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
