"""Read-only refresh of the remote psi=0.3, seed=32 run.

The event file and checkpoint are copied only after their remote size/mtime
metadata is unchanged across a download.  The checkpoint manifest hashes are
also verified locally.  Nothing is modified on the 3090.
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
RUN_NAME = (
    "dcppoR520_fixed600_200_layoutctx_noactor_peragentnoise_s3p0_md12_"
    "vmax30_psi0p3_nofilter_seed32_60m_20260815"
)
REMOTE_RUN = REMOTE_ROOT / "onpolicy/scripts/results/mec/mappo" / RUN_NAME / "run1"
REMOTE_LOG_DIR = REMOTE_RUN / "logs"
REMOTE_MODEL_DIR = REMOTE_RUN / "models"
DEST_EVENT = HERE / "data" / "remote" / "remote_psi0p3_seed32.tfevents"
DEST_RUN = HERE / "remote_checkpoint" / "run1"
DEST_MODEL_DIR = DEST_RUN / "models"
DEST_META = HERE / "data" / "remote" / "refresh_metadata.json"


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
    attempts: int = 6,
    pause_seconds: float = 1.0,
) -> dict[str, int]:
    local.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        before = stat_signature(sftp, remote)
        temporary = local.with_name(local.name + ".part")
        temporary.unlink(missing_ok=True)
        sftp.get(str(remote), str(temporary))
        after = stat_signature(sftp, remote)
        if before == after and temporary.stat().st_size == before[0]:
            shutil.move(str(temporary), str(local))
            os.utime(local, (after[1], after[1]))
            return {
                "bytes": int(after[0]),
                "mtime": int(after[1]),
                "attempt": attempt,
            }
        temporary.unlink(missing_ok=True)
        time.sleep(pause_seconds)
    raise RuntimeError(f"Remote file kept changing while copying: {remote}")


def read_remote_json(sftp: paramiko.SFTPClient, remote: PurePosixPath) -> dict:
    with sftp.open(str(remote), "r") as handle:
        return json.loads(handle.read().decode("utf-8"))


def fetch_checkpoint(sftp: paramiko.SFTPClient) -> dict:
    manifest_remote = REMOTE_MODEL_DIR / "checkpoint_manifest.json"
    manifest = read_remote_json(sftp, manifest_remote)
    expected_files = sorted(manifest["files"])
    remote_files = [manifest_remote, REMOTE_RUN / "args.json"] + [
        REMOTE_MODEL_DIR / name for name in expected_files
    ]
    signatures_before = {str(path): stat_signature(sftp, path) for path in remote_files}

    DEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict[str, int]] = {}
    get_stable(sftp, REMOTE_RUN / "args.json", DEST_RUN / "args.json")
    get_stable(sftp, manifest_remote, DEST_MODEL_DIR / "checkpoint_manifest.json")
    for name in expected_files:
        copied[name] = get_stable(sftp, REMOTE_MODEL_DIR / name, DEST_MODEL_DIR / name)

    signatures_after = {str(path): stat_signature(sftp, path) for path in remote_files}
    if signatures_before != signatures_after:
        raise RuntimeError("Remote checkpoint changed during the snapshot; rerun")

    local_manifest = json.loads(
        (DEST_MODEL_DIR / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    if local_manifest != manifest:
        raise RuntimeError("Downloaded checkpoint manifest differs from remote")
    for name, record in manifest["files"].items():
        local_file = DEST_MODEL_DIR / name
        if local_file.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"Size mismatch for checkpoint file: {name}")
        if sha256(local_file) != record["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for checkpoint file: {name}")

    return {
        "checkpoint_step": int(manifest["session_total_num_steps"]),
        "episode_index": int(manifest["episode_index"]),
        "files": len(expected_files) + 1,
        "remote_signatures": {key: list(value) for key, value in signatures_after.items()},
        "copied": copied,
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
            events = [
                item
                for item in sftp.listdir_attr(str(REMOTE_LOG_DIR))
                if stat.S_ISREG(item.st_mode) and item.filename.startswith("events.out.tfevents.")
            ]
            if not events:
                raise FileNotFoundError(f"No TensorBoard event file under {REMOTE_LOG_DIR}")
            latest_event = max(events, key=lambda item: (item.st_mtime, item.filename))
            event_meta = get_stable(
                sftp,
                REMOTE_LOG_DIR / latest_event.filename,
                DEST_EVENT,
            )
            checkpoint_meta = fetch_checkpoint(sftp)
            metadata = {
                "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "remote_host": f"{REMOTE_HOST}:22",
                "remote_run": str(REMOTE_RUN),
                "run_name": RUN_NAME,
                "event_remote_name": latest_event.filename,
                "event": event_meta,
                "checkpoint": checkpoint_meta,
                "read_only_remote": True,
            }
            DEST_META.parent.mkdir(parents=True, exist_ok=True)
            DEST_META.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
