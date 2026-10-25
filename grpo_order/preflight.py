import importlib.util
import os
from pathlib import Path
import platform
import re
import subprocess

from .assets import inventory, safetensor_dtypes, validate_layout, validate_receipt
from .core import digest, file_hash, packages, read_json
from .data import load_prepared
from .grading import check_grader, gold_parse
from .model import load_tokenizer, validate_context


def command(args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


def hardware():
    out = {"platform": platform.platform(), "python": platform.python_version(),
           "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
           "cuda_device_order": os.getenv("CUDA_DEVICE_ORDER"), "packages": packages(),
           "nvidia_smi": command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"]),
           "processes": command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])}
    if importlib.util.find_spec("torch"):
        import torch
        out.update(torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                   torch_arch_list=torch.cuda.get_arch_list())
        out["devices"] = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free, total = torch.cuda.mem_get_info()
                item = {"logical_index": i, "name": torch.cuda.get_device_name(i),
                        "uuid": str(getattr(torch.cuda.get_device_properties(i), "uuid", "")),
                        "capability": torch.cuda.get_device_capability(i), "total_bytes": total,
                        "free_bytes": free, "bf16": torch.cuda.is_bf16_supported()}
                try:
                    # Actual kernel/backward test catches unsupported architectures/drivers.
                    x = torch.randn(16, 16, device=f"cuda:{i}", dtype=torch.bfloat16, requires_grad=True)
                    (x @ x).float().square().mean().backward()
                    torch.cuda.synchronize(i)
                    item["bf16_kernel_backward"] = "passed"
                    del x
                    torch.cuda.empty_cache()
                except Exception as e:
                    item["bf16_kernel_backward"] = repr(e)
                out["devices"].append(item)
    return out


def uuid_mapping(h):
    """Join CUDA to nvidia-smi by full UUID, never by the two index spaces."""
    # torch 2.7.1 CUuuid.__str__ emits the UUID body, nvidia-smi emits GPU-<body>.
    # Keep the nvidia-smi spelling for CUDA_VISIBLE_DEVICES; normalize only the join key.
    def key(value):
        return value.strip().lower().removeprefix("gpu-")
    smi = {}
    for line in h["nvidia_smi"].get("stdout", "").splitlines():
        fields = [v.strip() for v in line.split(",")]
        if len(fields) >= 2:
            smi[key(fields[1])] = {"nvidia_smi_index": fields[0], "uuid": fields[1]}
    mapping = []
    for device in h.get("devices", []):
        uuid = key(device.get("uuid", ""))
        if not uuid or uuid not in smi:
            raise ValueError(f"Cannot match CUDA logical {device['logical_index']} UUID to nvidia-smi: {uuid!r}")
        mapping.append({"allocated_slot": device["logical_index"], "cuda_uuid": device["uuid"], **smi[uuid]})
    if len({m["uuid"].lower() for m in mapping}) != len(mapping):
        raise ValueError("Duplicate GPU UUID assignment")
    return mapping


def gpu_errors(h, config, single=False):
    errors = []
    devices = h.get("devices", [])
    expected = 1 if single else 2
    if len(devices) != expected:
        errors.append(f"Expected {expected} visible CUDA device(s), found {len(devices)}")
    if h["nvidia_smi"].get("returncode") != 0:
        errors.append("nvidia-smi unavailable or failed")
    try:
        mapping = uuid_mapping(h)
        h["uuid_mapping"] = mapping
        expected_uuid = os.getenv("GRPO_ASSIGNED_GPU_UUID")
        if single and expected_uuid and [m["uuid"].lower() for m in mapping] != [expected_uuid.lower()]:
            errors.append("Worker CUDA UUID differs from launcher-assigned GPU UUID")
    except ValueError as exc:
        errors.append(str(exc))
        mapping = []
    for d in devices:
        if not d["bf16"] or d["bf16_kernel_backward"] != "passed":
            errors.append(f"CUDA/BF16 kernel test failed: {d}")
        if d["free_bytes"] < config["runtime"]["min_free_vram_gib"] * 1024**3:
            errors.append(f"Insufficient configured free VRAM threshold: {d}")
    query = h["processes"]
    if query.get("returncode") != 0:
        errors.append("Unable to inspect GPU compute processes")
    elif not config["runtime"]["allow_other_gpu_processes"]:
        uuids = {r["uuid"].lower() for r in mapping}
        own_pids = {str(os.getpid())}
        status = Path("/proc/self/status")
        if status.is_file():
            for line in status.read_text().splitlines():
                if line.startswith("NSpid:"):
                    own_pids.update(line.split()[1:])
        other = [line for line in query.get("stdout", "").splitlines()
                 if line.split(",")[0].strip().lower() in uuids and line.split(",")[1].strip() not in own_pids]
        if other:
            errors.append("Existing compute processes or unresolved PID namespace on assigned UUID(s): " + str(other))
    return errors


def dependency_errors():
    import importlib.metadata
    errors = []
    root = Path(__file__).resolve().parents[1]
    lines = dict.fromkeys(line for filename in ("requirements.txt", "requirements-assets.txt")
                         for line in (root / filename).read_text().splitlines())
    for line in lines:
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        name = name.split("[")[0]
        try:
            actual = importlib.metadata.version(name).split("+")[0]
        except importlib.metadata.PackageNotFoundError:
            actual = "MISSING"
        if actual != expected:
            errors.append(f"Package {name}: required {expected}, found {actual}")
    return errors


def resolve_model(spec):
    if not spec.get("path"):
        raise ValueError("Model path/ID unset; fill models.base.path and models.rl.path")
    path = Path(spec["path"]).expanduser()
    receipt = None
    if spec.get("hub_id"):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing local {spec['hub_id']} at {path}; run bash scripts/run.sh download --only models (does not train)")
        receipt = validate_receipt(path, spec["hub_id"], spec.get("revision"))
        validate_layout(path, spec["hub_id"], "model")
        if spec.get("tokenizer_revision") and spec["tokenizer_revision"] != receipt["tokenizer_revision"]:
            raise ValueError("Requested tokenizer revision differs from downloaded model snapshot")
    if not path.is_dir():
        if not spec.get("revision") or not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
            raise ValueError("Cached Hub model requires immutable 40-character commit revision")
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(spec["path"], revision=spec["revision"], local_files_only=True))
    if not (path / "config.json").is_file():
        raise ValueError(f"Not a complete original model directory: {path}")
    if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
        raise ValueError(f"No model weights in {path}; adapter-only checkpoints cannot be starting models")
    # Hash weights, tokenizer, config and model-card bytes, not only mutable path strings.
    files = inventory(path)
    card = Path(spec["model_card"]).expanduser() if spec.get("model_card") else path / "README.md"
    card_text = card.read_text() if card.is_file() else None
    meta = {"configured": spec, "resolved_path": str(path.resolve()), "files": files,
            "content_hash": digest(files), "model_card_text": card_text,
            "model_card_hash": file_hash(card) if card.is_file() else None,
            "lineage_status": spec["lineage"], "history_status": spec["training_history"],
            "hf_config": read_json(path / "config.json"), "asset_receipt": receipt,
            "resolved_model_revision": receipt["revision"] if receipt else spec.get("revision"),
            "resolved_tokenizer_revision": receipt["tokenizer_revision"] if receipt else None,
            "lineage_sources": spec.get("lineage_sources", [])}
    meta["actual_stored_parameter_dtypes"] = safetensor_dtypes(path)
    if card.is_file():
        try:
            from huggingface_hub import ModelCard
            meta["model_card_metadata"] = ModelCard.load(str(card)).data.to_dict()
        except Exception as e:
            meta["model_card_metadata_error"] = str(e)
    return path, meta


def check(config, *, environment_only=False):
    h = hardware()
    errors = gpu_errors(h, config) + dependency_errors()
    report = {"hardware": h, "errors": errors, "models": {}, "warnings": []}
    if environment_only:
        return report
    for role, spec in config["models"].items():
        try:
            path, meta = resolve_model(spec)
            if Path(config["output_dir"]).resolve().is_relative_to(path.resolve()):
                raise ValueError("output_dir must be outside the original model directory")
            report["models"][role] = meta
            if not spec["role_verified"]:
                errors.append(f"models.{role}.role_verified is false: manually verify Base / public RL role from model card")
            if not meta["model_card_text"]:
                report["warnings"].append(f"{role}: no substantive model card; use recorded official repo/paper evidence; unreported history remains unknown")
        except Exception as e:
            errors.append(f"{role}: {e}")
    if len(report["models"]) == 2:
        if report["models"]["base"]["content_hash"] == report["models"]["rl"]["content_hash"]:
            errors.append("Base and RL resolve to identical original model contents; verify model roles")
        report["relationship_evidence"] = {role: report["models"][role].get("model_card_metadata", {}).get("base_model") for role in ("base", "rl")}
        report["warnings"].append("Model-card base_model is a documented claim; equal size or architecture does not establish lineage")
    try:
        report["grader"] = check_grader()
        manifest, splits = load_prepared(config)
        report["manifest_hash"] = manifest["hash"]
        for rows in splits.values():
            for row in rows:
                gold_parse(row["answer"])
        for role, meta in report["models"].items():
            tok, context = load_tokenizer(Path(meta["resolved_path"]), config["models"][role])
            report["models"][role]["eos"] = tok.grpo_eos
            report["models"][role]["prompt_lengths"] = validate_context(tok, context, config, splits)
    except Exception as e:
        errors.append(f"data/tokenizer/grader: {e}")
    return report
