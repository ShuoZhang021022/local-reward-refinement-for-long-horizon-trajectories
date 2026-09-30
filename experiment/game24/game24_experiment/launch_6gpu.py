"""Run the six planned Game24 arm/seed jobs on six separate A100 GPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import ExperimentConfig
from .data import load_puzzles, split_puzzles
from .train import _config_from_plan, _source_hashes


MIN_MEMORY_MIB = 38 * 1024  # A100 40GB reports usable capacity below its nominal size.


def visible_gpu_ids(value: str | None) -> list[str]:
    ids = [part.strip() for part in value.split(",")] if value is not None else [str(i) for i in range(6)]
    if len(ids) != 6 or any(not part for part in ids) or len(set(ids)) != 6:
        raise ValueError("Exactly six distinct GPUs must be visible in CUDA_VISIBLE_DEVICES")
    return ids


def planned_jobs(config: ExperimentConfig, gpu_ids: list[str]) -> list[tuple[str, int, str]]:
    pairs = [(arm, seed) for seed in config.training_seeds for arm in config.arms]
    if len(pairs) != 6 or len(gpu_ids) != 6:
        raise ValueError("This launcher requires exactly six planned arm/seed runs and six GPUs")
    return [(arm, seed, gpu_id) for (arm, seed), gpu_id in zip(pairs, gpu_ids)]


def worker_cpu_threads(job_count: int) -> int:
    """Keep six model processes from each claiming the full CPU thread pool."""
    available = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                 else os.cpu_count() or 1)
    return max(1, available // job_count)


def check_gpu(gpu_id: str) -> tuple[str, int, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", gpu_id, "--query-gpu=name,memory.total,uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot inspect GPU {gpu_id} with nvidia-smi") from exc
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one physical GPU for identifier {gpu_id}: {rows}")
    try:
        name, memory, uuid = (part.strip() for part in rows[0].rsplit(",", 2))
        memory_mib = int(memory)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Cannot parse GPU {gpu_id} details: {rows[0]}") from exc
    if "A100" not in name or memory_mib < MIN_MEMORY_MIB or not uuid.startswith("GPU-"):
        raise RuntimeError(
            f"GPU {gpu_id} is {name} with {memory_mib} MiB, UUID {uuid}; "
            "expected a full A100 with at least 38 GiB"
        )
    return name, memory_mib, uuid


def require_distinct_physical_gpus(infos: list[tuple[str, int, str]]) -> None:
    if len({uuid for _, _, uuid in infos}) != len(infos):
        raise ValueError("Selected GPU identifiers resolve to the same physical GPU")


def launch(plan_path: Path, output_root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config = _config_from_plan(plan)
    if plan["source_hashes"] != _source_hashes():
        raise ValueError("Source files changed after this plan was locked; prepare a new plan")
    puzzles, data_hash = load_puzzles(repo_root / config.dataset_csv)
    if plan["split"] != split_puzzles(puzzles, data_hash, seed=config.split_seed):
        raise ValueError("Dataset or split does not match the locked plan")
    gpu_ids = visible_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES"))
    jobs = planned_jobs(config, gpu_ids)
    cpu_threads = worker_cpu_threads(len(jobs))
    gpu_infos = [check_gpu(gpu_id) for _, _, gpu_id in jobs]
    require_distinct_physical_gpus(gpu_infos)
    for (arm, seed, gpu_id), (name, memory_mib, uuid) in zip(jobs, gpu_infos):
        print(f"{arm} seed={seed}: GPU {gpu_id} "
              f"({name}, {memory_mib} MiB, {uuid})", flush=True)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir()

    processes: list[tuple[str, int, subprocess.Popen]] = []
    try:
        for arm, seed, gpu_id in jobs:
            label = f"{arm.replace('_', '-')}-{seed}"
            run_dir = output_root / label
            log_path = output_root / "logs" / f"{label}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("OMP_NUM_THREADS", str(cpu_threads))
            env.setdefault("MKL_NUM_THREADS", str(cpu_threads))
            command = [
                sys.executable, "-m", "game24_experiment.train", "run",
                "--plan", str(plan_path), "--output", str(run_dir.resolve()),
                "--arm", arm, "--seed", str(seed),
            ]
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=repo_root, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
            processes.append((arm, seed, process))
            print(f"started {label}: PID {process.pid}; CPU threads "
                  f"OMP={env['OMP_NUM_THREADS']} MKL={env['MKL_NUM_THREADS']}; "
                  f"log {log_path}", flush=True)

        pending = {(arm, seed): process for arm, seed, process in processes}
        failed = []
        while pending:
            for key, process in list(pending.items()):
                status = process.poll()
                if status is None:
                    continue
                del pending[key]
                if status:
                    failed.append((key, status))
                print(f"finished {key[0]} seed={key[1]}: exit={status}", flush=True)
            if pending:
                time.sleep(5)
        if failed:
            raise RuntimeError(f"Game24 runs failed (no automatic retry): {failed}")
    except BaseException:
        for _, _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, process in processes:
            if process.poll() is None:
                process.wait()
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    launch(args.plan, args.output_root)


if __name__ == "__main__":
    main()
