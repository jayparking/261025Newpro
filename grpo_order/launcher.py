import os
from pathlib import Path
import subprocess
import sys
import time

from .core import atomic_json, code_hash, digest, packages, read_json
from .data import load_prepared
from .freeze import verify_freeze
from .integrity import validate_completion
from .storage import run_path


def launch(config, config_path, mode, resume_failed=False, seed=1, stop_after=None):
    if stop_after is not None and mode != "smoke":
        raise ValueError("--stop-after is only supported for smoke")
    if mode == "full":
        verify_freeze(config)
    # Probe in a child that exits before training; a CUDA-probing parent would keep
    # contexts resident and appear as an unexpected GPU process to the workers.
    stamp = time.time_ns()
    report_path = Path(config["output_dir"]) / "preflight" / f"{stamp}.json"
    probe_config = report_path.with_name(f"config-{stamp}.json")
    atomic_json(probe_config, config)
    probe_env = os.environ.copy()
    slots = config["runtime"]["allocated_gpu_slots"]
    single_gpu = slots == [0]
    probe_env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, slots)))
    probe_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    probe = subprocess.run([sys.executable, "-m", "grpo_order", "preflight", "--config", str(probe_config),
                            "--report", str(report_path)], env=probe_env)
    if not report_path.exists():
        raise RuntimeError(f"Preflight subprocess failed (exit {probe.returncode}) without report")
    report = read_json(report_path)
    if report["errors"] or probe.returncode:
        raise RuntimeError(f"Preflight failed: {report['errors']}; report: {report_path}")
    if mode == "full":
        verify_freeze(config, report["models"])
    mapping = report["hardware"]["uuid_mapping"]
    if [m["allocated_slot"] for m in mapping] != slots:
        raise ValueError(f"Preflight must resolve allocated CUDA slots {slots} to UUIDs")
    seeds = [seed] if mode in ("pilot", "smoke") else config["seeds"]["training"]
    failed = []
    for s in seeds:
        for role in ("base", "rl"):
            processes, handles = [], []
            try:
                for slot, order in ((0, "random"), (0 if single_gpu else 1, "curriculum")):
                    root = run_path(config, mode, role, order, s)
                    if (root / "complete.json").exists() and resume_failed:
                        # Finished jobs are never retrained. Refuse changed identity even here.
                        metadata = read_json(root / "metadata.json")
                        expected_updates = 300 if mode == "full" else (3 if mode == "smoke" else config["pilot"]["updates"])
                        validate_completion(metadata, read_json(root / "complete.json"), expected_updates)
                        manifest, _ = load_prepared(config)
                        contract = metadata["contract"]
                        if (contract["config"] != digest(config) or contract["data"] != manifest["hash"]
                            or contract["model"] != report["models"][role]["content_hash"]
                            or contract["code"] != code_hash() or contract["packages"] != packages()):
                            raise ValueError(f"Completed run identity changed: {root}")
                        print(f"Skip completed: {root}", flush=True)
                        continue
                    if root.exists() and not resume_failed:
                        raise FileExistsError(f"Existing run {root}; use --resume-failed")
                    log_dir = Path(config["output_dir"]) / "launcher_logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log = (log_dir / f"{mode}_{role}_{s}_{order}_{time.time_ns()}.log").open("w")
                    handles.append(log)
                    env = os.environ.copy()
                    uuid = mapping[slot]["uuid"]
                    env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                               GRPO_ALLOCATED_SLOT=str(slot), GRPO_ASSIGNED_GPU_UUID=uuid, TOKENIZERS_PARALLELISM="false",
                               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
                    cmd = [sys.executable, "-m", "grpo_order", "train", "--config", str(config_path),
                           "--model", role, "--order", order, "--seed", str(s), "--mode", mode]
                    if resume_failed:
                        cmd.append("--resume")
                    if stop_after is not None:
                        cmd.extend(["--stop-after", str(stop_after)])
                    print(f"Start {role}/{order}/seed{s} on {uuid}; log: {log.name}", flush=True)
                    # One GPU: wait and release its exclusive lock before starting the next run.
                    # Stream the active worker so evaluation/training progress is visible.
                    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE if single_gpu else log,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
                    processes.append((root, p))
                    if single_gpu:
                        try:
                            for line in p.stdout:
                                log.write(line)
                                log.flush()
                                print(line, end="", flush=True)
                        finally:
                            p.stdout.close()
                        code = p.wait()
                        if code:
                            failed.append({"run": str(root), "returncode": code})
                            break
                if not single_gpu:
                    for root, p in processes:
                        code = p.wait()
                        if code:
                            failed.append({"run": str(root), "returncode": code})
            except BaseException:
                for _, p in processes:
                    if p.poll() is None:
                        p.terminate()
                for _, p in processes:
                    p.wait()
                raise
            finally:
                for handle in handles:
                    handle.close()
            if failed:
                break
        if failed:
            break
    if failed:
        raise RuntimeError(f"Stopped after failed run; use --resume-failed: {failed}")
