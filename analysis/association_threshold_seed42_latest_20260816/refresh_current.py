"""Read-only refresh of the three current seed-42 remote runs.

The remote event files and checkpoints are copied only when their remote
metadata stays unchanged during each copy.  Checkpoint file hashes are
verified locally before the snapshot is considered usable.
"""

from __future__ import annotations

import datetime as dt
import getpass
import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path, PurePosixPath

import paramiko


HERE = Path(__file__).resolve().parent
REMOTE_HOST = "114.212.117.24"
REMOTE_USER = "test"
REMOTE_ROOT = PurePosixPath(
    "/home/test/wyj/Projects/on-policy-last-obs-clean-5UAV"
)
RUNS = {
    "psi0p1": "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p1_nofilter_seed42_60m_20260816",
    "psi0p3": "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p3_nofilter_seed42_60m_20260816",
    "psi0p5": "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_vmax30_psi0p5_nofilter_seed42_60m_20260816",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stat_signature(sftp: paramiko.SFTPClient, remote: PurePosixPath) -> tuple[int, int]:
    item = sftp.stat(str(remote))
    return int(item.st_size), int(item.st_mtime)


def get_stable(
    sftp: paramiko.SFTPClient,
    remote: PurePosixPath,
    local: Path,
    attempts: int = 8,
    pause_seconds: float = 1.0,
) -> dict[str, int]:
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_name(local.name + ".part")
    for attempt in range(1, attempts + 1):
        before = stat_signature(sftp, remote)
        temporary.unlink(missing_ok=True)
        sftp.get(str(remote), str(temporary))
        after = stat_signature(sftp, remote)
        if before == after and temporary.stat().st_size == before[0]:
            temporary.replace(local)
            os.utime(local, (after[1], after[1]))
            return {"bytes": int(after[0]), "mtime": int(after[1]), "attempt": attempt}
        temporary.unlink(missing_ok=True)
        time.sleep(pause_seconds)
    raise RuntimeError(f"Remote file kept changing while copying: {remote}")


def read_remote_json(sftp: paramiko.SFTPClient, remote: PurePosixPath) -> dict:
    with sftp.open(str(remote), "r") as handle:
        return json.loads(handle.read().decode("utf-8"))


def fetch_run(sftp: paramiko.SFTPClient, key: str, run_name: str) -> dict:
    remote_run = REMOTE_ROOT / "onpolicy/scripts/results/mec/mappo" / run_name / "run1"
    remote_log_dir = remote_run / "logs"
    remote_model_dir = remote_run / "models"
    event_items = [
        item
        for item in sftp.listdir_attr(str(remote_log_dir))
        if stat.S_ISREG(item.st_mode) and item.filename.startswith("events.out.tfevents.")
    ]
    if not event_items:
        raise FileNotFoundError(f"No TensorBoard event file under {remote_log_dir}")
    latest_event = max(event_items, key=lambda item: (item.st_mtime, item.filename))

    destination_event = HERE / "data" / "remote" / f"remote_{key}.tfevents"
    event_meta = get_stable(
        sftp,
        remote_log_dir / latest_event.filename,
        destination_event,
    )

    remote_manifest = remote_model_dir / "checkpoint_manifest.json"
    remote_args = remote_run / "args.json"
    manifest = read_remote_json(sftp, remote_manifest)
    expected_files = sorted(manifest["files"])
    remote_files = [remote_args, remote_manifest] + [
        remote_model_dir / name for name in expected_files
    ]
    signatures_before = {str(path): stat_signature(sftp, path) for path in remote_files}

    destination_run = HERE / "remote_snapshots" / f"remote_{key}" / "checkpoint_snapshot"
    destination_model = destination_run / "models"
    destination_run.mkdir(parents=True, exist_ok=True)
    destination_model.mkdir(parents=True, exist_ok=True)
    copied = {
        "args.json": get_stable(sftp, remote_args, destination_run / "args.json"),
        "checkpoint_manifest.json": get_stable(
            sftp,
            remote_manifest,
            destination_model / "checkpoint_manifest.json",
        ),
    }
    for name in expected_files:
        copied[name] = get_stable(
            sftp,
            remote_model_dir / name,
            destination_model / name,
        )

    signatures_after = {str(path): stat_signature(sftp, path) for path in remote_files}
    if signatures_before != signatures_after:
        raise RuntimeError(f"Remote checkpoint changed during snapshot: {run_name}")

    local_manifest = json.loads(
        (destination_model / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    if local_manifest != manifest:
        raise RuntimeError(f"Manifest mismatch after snapshot: {run_name}")
    for name, record in manifest["files"].items():
        local_file = destination_model / name
        if local_file.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"Size mismatch for {run_name}/{name}")
        if sha256(local_file) != record["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {run_name}/{name}")

    args = json.loads((destination_run / "args.json").read_text(encoding="utf-8"))
    return {
        "key": key,
        "run_name": run_name,
        "remote_run": str(remote_run),
        "event_remote_name": latest_event.filename,
        "event": event_meta,
        "checkpoint_step": int(manifest["session_total_num_steps"]),
        "episode_index": int(manifest["episode_index"]),
        "checkpoint_dir": str(destination_model),
        "args": {
            name: args.get(name)
            for name in (
                "seed",
                "association_threshold",
                "offload_deadline_filter",
                "num_env_steps",
                "n_rollout_threads",
                "neighbor_distance",
                "neighbor_R",
                "hotspot_layout_mode",
                "md_lifetime_min",
                "md_lifetime_max",
                "mean_velocity",
                "noise_scale",
                "actor_message_mode",
                "experiment_name",
            )
        },
        "copied_files": len(copied),
    }


def main() -> None:
    password = os.environ.get("DCPPO_REMOTE_PASSWORD") or getpass.getpass(
        f"Password for {REMOTE_USER}@{REMOTE_HOST}: "
    )
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(REMOTE_HOST, username=REMOTE_USER, password=password, timeout=20)
    try:
        with client.open_sftp() as sftp:
            refreshed = [fetch_run(sftp, key, name) for key, name in RUNS.items()]
    finally:
        client.close()

    payload = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "remote_host": f"{REMOTE_HOST}:22",
        "read_only_remote": True,
        "runs": refreshed,
    }
    output = HERE / "data" / "refresh_metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
