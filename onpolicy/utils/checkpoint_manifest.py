import hashlib
import json
import os
from pathlib import Path


CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
CHECKPOINT_MANIFEST_FORMAT = 1
OPTIONAL_CHECKPOINT_NAMES = ("md_gru_shared.pt",)


def expected_checkpoint_names(num_agents, extra_names=()):
    names = []
    for agent_id in range(int(num_agents)):
        names.extend(
            (
                f"actor_agent{agent_id}.pt",
                f"critic_agent{agent_id}.pt",
                f"normer{agent_id}.pkl",
            )
        )
    names.extend(str(name) for name in extra_names)
    return tuple(names)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {
        "bytes": int(stat.st_size),
        "sha256": sha256(path),
    }


def build_checkpoint_manifest(
    model_dir,
    num_agents,
    episode_index,
    session_total_num_steps,
    source_total_num_steps,
    save_interval_episodes,
    cumulative_step_provenance,
    args_path=None,
    extra_checkpoint_names=(),
):
    model_dir = Path(model_dir)
    unknown_extra_names = set(extra_checkpoint_names) - set(
        OPTIONAL_CHECKPOINT_NAMES
    )
    if unknown_extra_names:
        raise ValueError(
            f"unsupported extra checkpoint files: {sorted(unknown_extra_names)}"
        )
    session_steps = int(session_total_num_steps)
    source_steps = int(source_total_num_steps)
    manifest = {
        "format_version": CHECKPOINT_MANIFEST_FORMAT,
        "episode_index": int(episode_index),
        "session_total_num_steps": session_steps,
        "source_total_num_steps": source_steps,
        "total_num_steps": source_steps + session_steps,
        "cumulative_step_provenance": str(cumulative_step_provenance),
        "save_interval_episodes": int(save_interval_episodes),
        "num_agents": int(num_agents),
        "files": {
            name: file_fingerprint(model_dir / name)
            for name in expected_checkpoint_names(
                num_agents, extra_checkpoint_names
            )
        },
    }
    if args_path is not None:
        manifest["args_json"] = file_fingerprint(args_path)
    return manifest


def write_checkpoint_manifest_atomic(model_dir, manifest):
    model_dir = Path(model_dir)
    manifest_path = model_dir / CHECKPOINT_MANIFEST_NAME
    temporary_path = model_dir / (CHECKPOINT_MANIFEST_NAME + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, manifest_path)


def read_checkpoint_manifest(
    model_dir, num_agents, args_path=None, extra_checkpoint_names=None
):
    model_dir = Path(model_dir)
    manifest_path = model_dir / CHECKPOINT_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("format_version") != CHECKPOINT_MANIFEST_FORMAT:
        raise ValueError("unsupported checkpoint manifest format")
    if int(manifest.get("num_agents", -1)) != int(num_agents):
        raise ValueError("checkpoint manifest num_agents mismatch")

    files = manifest.get("files")
    actual_names = set(files) if isinstance(files, dict) else set()
    if extra_checkpoint_names is None:
        extra_checkpoint_names = tuple(
            name for name in OPTIONAL_CHECKPOINT_NAMES if name in actual_names
        )
    unknown_extra_names = set(extra_checkpoint_names) - set(
        OPTIONAL_CHECKPOINT_NAMES
    )
    if unknown_extra_names:
        raise ValueError(
            f"unsupported extra checkpoint files: {sorted(unknown_extra_names)}"
        )
    expected_file_names = expected_checkpoint_names(
        num_agents, extra_checkpoint_names
    )
    expected_names = set(expected_file_names)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            "checkpoint manifest file set mismatch: "
            f"missing={missing}, extra={extra}"
        )
    for name in expected_file_names:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"manifest checkpoint file is missing: {path}")
        actual = file_fingerprint(path)
        if actual != files[name]:
            raise RuntimeError(
                f"checkpoint file does not match manifest: {name}: "
                f"expected {files[name]}, got {actual}"
            )

    session_steps = manifest.get("session_total_num_steps")
    source_steps = manifest.get("source_total_num_steps")
    total_steps = manifest.get("total_num_steps")
    if not all(isinstance(value, int) and value >= 0 for value in (
        session_steps, source_steps, total_steps
    )):
        raise ValueError("checkpoint manifest has invalid training steps")
    if source_steps + session_steps != total_steps:
        raise ValueError("checkpoint manifest cumulative step mismatch")
    if manifest.get("cumulative_step_provenance") not in {
        "verified_chain",
        "legacy_declared_base",
    }:
        raise ValueError("checkpoint manifest has invalid step provenance")

    args_fingerprint = manifest.get("args_json")
    if args_path is not None:
        if not isinstance(args_fingerprint, dict):
            raise ValueError("checkpoint manifest does not bind args.json")
        actual_args = file_fingerprint(args_path)
        if actual_args != args_fingerprint:
            raise RuntimeError(
                "args.json does not match checkpoint manifest: "
                f"expected {args_fingerprint}, got {actual_args}"
            )
    return manifest
