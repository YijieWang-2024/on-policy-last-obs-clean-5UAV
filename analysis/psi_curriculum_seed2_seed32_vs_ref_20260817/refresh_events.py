"""Refresh only the three remote TensorBoard event files; never touch checkpoints."""

from __future__ import annotations

import os
import time
from pathlib import Path, PurePosixPath

import paramiko


HERE = Path(__file__).resolve().parent
REMOTE_ROOT = PurePosixPath("/home/test/wyj/Projects/on-policy-last-obs-clean-5UAV")
RUNS = {
    "psi0p5": "dcppoR520_fixed600_200_noactor_psi0p5_nofilter_curr_p07_10m25m_seed2_60m_20260816",
    "psi0p7": "dcppoR520_fixed600_200_noactor_psi0p7_nofilter_curr_p07_10m25m_seed2_60m_20260816",
    "psi0p9": "dcppoR520_fixed600_200_noactor_psi0p9_nofilter_curr_p07_10m25m_seed2_60m_20260816",
}


def signature(sftp, remote):
    item = sftp.stat(str(remote))
    return int(item.st_size), int(item.st_mtime)


def fetch_stable(sftp, remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_name(local.name + ".part")
    for _ in range(8):
        before = signature(sftp, remote)
        temporary.unlink(missing_ok=True)
        sftp.get(str(remote), str(temporary))
        after = signature(sftp, remote)
        if before == after and temporary.stat().st_size == before[0]:
            temporary.replace(local)
            os.utime(local, (after[1], after[1]))
            return {"remote": str(remote), "bytes": after[0], "mtime": after[1]}
        temporary.unlink(missing_ok=True)
        time.sleep(0.5)
    raise RuntimeError(f"remote event file kept changing: {remote}")


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "114.212.117.24",
        22,
        "test",
        os.environ["DCPPO_REMOTE_PASSWORD"],
        timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    results = []
    try:
        with client.open_sftp() as sftp:
            for key, name in RUNS.items():
                log_dir = REMOTE_ROOT / "onpolicy/scripts/results/mec/mappo" / name / "run1" / "logs"
                items = [item for item in sftp.listdir_attr(str(log_dir)) if item.filename.startswith("events.out.tfevents.")]
                latest = max(items, key=lambda item: (item.st_mtime, item.filename))
                results.append(fetch_stable(sftp, log_dir / latest.filename, HERE / "data" / "remote" / f"{key}.tfevents"))
    finally:
        client.close()
    print(results)


if __name__ == "__main__":
    main()
