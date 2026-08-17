"""Read-only refresh of the three remote curriculum checkpoints.

The remote trainer remains untouched.  A checkpoint is accepted only when
the manifest and every copied file have the same remote size/mtime before and
after transfer.  The resulting snapshot has the same flat layout as the
local evaluator's checkpoint_snapshot directory.
"""

from __future__ import annotations

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
SNAPSHOT_SUBDIR = os.environ.get("MEC_REFRESH_SUBDIR", "remote_snapshots")
REMOTE_HOST = "114.212.117.24"
REMOTE_USER = "test"
REMOTE_ROOT = PurePosixPath(
    "/home/test/wyj/Projects/on-policy-last-obs-clean-5UAV"
)
REMOTE_RUNS = {
    "psi0p5": "dcppoR520_fixed600_200_noactor_psi0p5_nofilter_curr_p07_10m25m_seed2_60m_20260816",
    "psi0p7": "dcppoR520_fixed600_200_noactor_psi0p7_nofilter_curr_p07_10m25m_seed2_60m_20260816",
    "psi0p9": "dcppoR520_fixed600_200_noactor_psi0p9_nofilter_curr_p07_10m25m_seed2_60m_20260816",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_signature(sftp, path: PurePosixPath) -> tuple[int, int]:
    item = sftp.stat(str(path))
    return int(item.st_size), int(item.st_mtime)


def copy_stable(sftp, remote: PurePosixPath, local: Path) -> dict[str, int]:
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_name(local.name + ".part")
    for attempt in range(1, 9):
        before = remote_signature(sftp, remote)
        temporary.unlink(missing_ok=True)
        sftp.get(str(remote), str(temporary))
        after = remote_signature(sftp, remote)
        if before == after and temporary.stat().st_size == before[0]:
            temporary.replace(local)
            os.utime(local, (after[1], after[1]))
            return {"bytes": after[0], "mtime": after[1], "attempt": attempt}
        temporary.unlink(missing_ok=True)
        time.sleep(1.0)
    raise RuntimeError(f"Remote file kept changing while copying: {remote}")


def remote_json(sftp, path: PurePosixPath) -> dict:
    with sftp.open(str(path), "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def manifest_names(manifest: dict) -> list[str]:
    files = manifest.get("files", [])
    if isinstance(files, dict):
        return sorted(files)
    return sorted(str(item) for item in files)


def refresh_one(sftp, key: str, run_name: str) -> dict:
    remote_run = REMOTE_ROOT / "onpolicy/scripts/results/mec/mappo" / run_name / "run1"
    remote_models = remote_run / "models"
    remote_manifest_path = remote_models / "checkpoint_manifest.json"
    remote_args_path = remote_run / "args.json"
    manifest = remote_json(sftp, remote_manifest_path)
    model_names = manifest_names(manifest)
    files = [remote_args_path, remote_manifest_path] + [remote_models / name for name in model_names]
    before = {str(path): remote_signature(sftp, path) for path in files}

    destination = HERE / SNAPSHOT_SUBDIR / f"remote_{key}" / "checkpoint_snapshot"
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    copied["args.json"] = copy_stable(sftp, remote_args_path, destination / "args.json")
    copied["checkpoint_manifest.json"] = copy_stable(
        sftp, remote_manifest_path, destination / "checkpoint_manifest.json"
    )
    for name in model_names:
        copied[name] = copy_stable(sftp, remote_models / name, destination / name)

    after = {str(path): remote_signature(sftp, path) for path in files}
    if before != after:
        raise RuntimeError(f"Remote checkpoint changed during snapshot: {run_name}")
    local_manifest = json.loads((destination / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if local_manifest != manifest:
        raise RuntimeError(f"Manifest mismatch after snapshot: {run_name}")

    event_items = [
        item for item in sftp.listdir_attr(str(remote_run / "logs"))
        if stat.S_ISREG(item.st_mode) and item.filename.startswith("events.out.tfevents.")
    ]
    event_meta = None
    if event_items:
        event = max(event_items, key=lambda item: (item.st_mtime, item.filename))
        event_meta = copy_stable(
            sftp,
            remote_run / "logs" / event.filename,
            HERE / SNAPSHOT_SUBDIR / "data" / f"remote_{key}.tfevents",
        )

    refresh = {
        "remote_run": str(remote_run),
        "key": key,
        "manifest": manifest,
        "checkpoint_dir": str(destination),
        "files": copied,
        "sha256": {name: sha256(destination / name) for name in copied},
        "event": event_meta,
    }
    (destination.parent / "refresh.json").write_text(
        json.dumps(refresh, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return refresh


def main() -> None:
    password = os.environ.get("MEC_REMOTE_PASSWORD") or getpass.getpass(
        f"Password for {REMOTE_USER}@{REMOTE_HOST}: "
    )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(REMOTE_HOST, 22, REMOTE_USER, password, timeout=20)
    sftp = client.open_sftp()
    try:
        results = [refresh_one(sftp, key, name) for key, name in REMOTE_RUNS.items()]
    finally:
        sftp.close()
        client.close()
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
