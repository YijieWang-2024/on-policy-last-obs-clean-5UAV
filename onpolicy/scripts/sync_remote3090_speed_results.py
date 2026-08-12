"""Incrementally archive the six formal remote speed runs beside local runs."""

from __future__ import annotations

import argparse
import getpass
import os
import stat
from pathlib import Path, PurePosixPath

import paramiko


HOST = "114.212.117.24"
USER = "test"
REMOTE_ROOT = PurePosixPath(
    "/home/test/wyj/Projects/on-policy-last-obs-clean-5UAV/"
    "onpolicy/scripts/results/mec/mappo"
)
LOCAL_ROOT = Path(__file__).resolve().parent / "results" / "mec" / "mappo"
REMOTE_LOG_ROOT = REMOTE_ROOT.parents[4] / "training_logs"
LOCAL_LOG_ROOT = Path(__file__).resolve().parents[2] / "training_logs"
EXPERIMENTS = (
    "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax10_seed32_60m_20260811_retry1",
    "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax20_seed32_60m_20260811_retry1",
    "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax30_seed32_60m_20260811_retry2",
    "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed2_60m_20260812",
    "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed32_60m_20260812",
    "dcppoR520_fixed600_200_layoutctx_inputv2_peragentnoise_s3p0_md12_vmax40_seed42_60m_20260812",
)
TRAINING_LOGS = tuple(f"{experiment}.log" for experiment in EXPERIMENTS)


def sync_file(
    sftp: paramiko.SFTPClient, remote: PurePosixPath, local: Path
) -> tuple[bool, int]:
    before = sftp.stat(str(remote))
    if (
        local.is_file()
        and local.stat().st_size == before.st_size
        and int(local.stat().st_mtime) == int(before.st_mtime)
    ):
        return False, 0
    local.parent.mkdir(parents=True, exist_ok=True)
    temp = local.with_name(local.name + ".part")
    sftp.get(str(remote), str(temp))
    after = sftp.stat(str(remote))
    if (
        temp.stat().st_size != before.st_size
        or before.st_size != after.st_size
        or int(before.st_mtime) != int(after.st_mtime)
    ):
        temp.unlink(missing_ok=True)
        raise IOError(f"Remote file changed during download: {remote}; rerun sync")
    os.utime(temp, (after.st_atime, after.st_mtime))
    temp.replace(local)
    return True, after.st_size


def sync_tree(sftp: paramiko.SFTPClient, remote: PurePosixPath, local: Path) -> tuple[int, int]:
    files = bytes_downloaded = 0
    local.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(str(remote)):
        remote_child = remote / entry.filename
        local_child = local / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            child_files, child_bytes = sync_tree(sftp, remote_child, local_child)
            files += child_files
            bytes_downloaded += child_bytes
        elif stat.S_ISREG(entry.st_mode):
            changed, size = sync_file(sftp, remote_child, local_child)
            files += int(changed)
            bytes_downloaded += size
    return files, bytes_downloaded


def remote_manifest(sftp: paramiko.SFTPClient, root: PurePosixPath) -> dict[str, int]:
    manifest: dict[str, int] = {}

    def walk(remote: PurePosixPath, relative: PurePosixPath) -> None:
        for entry in sftp.listdir_attr(str(remote)):
            remote_child = remote / entry.filename
            relative_child = relative / entry.filename
            if stat.S_ISDIR(entry.st_mode):
                walk(remote_child, relative_child)
            elif stat.S_ISREG(entry.st_mode):
                manifest[str(relative_child)] = entry.st_size

    walk(root, PurePosixPath())
    return manifest


def local_manifest(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith(".part")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=USER)
    args = parser.parse_args()
    password = os.environ.get("DCPPO_REMOTE_PASSWORD") or getpass.getpass(
        f"Password for {args.user}@{args.host}: "
    )

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=password, timeout=15)
    try:
        with client.open_sftp() as sftp:
            for experiment in EXPERIMENTS:
                files, size = sync_tree(
                    sftp, REMOTE_ROOT / experiment, LOCAL_ROOT / experiment
                )
                remote_files = remote_manifest(sftp, REMOTE_ROOT / experiment)
                local_files = local_manifest(LOCAL_ROOT / experiment)
                if remote_files != local_files:
                    print(f"{experiment}: remote changed during verification; rerun sync")
                    continue
                print(
                    f"{experiment}: downloaded {files} files, {size} bytes; "
                    f"verified {len(remote_files)} files"
                )
            for log_name in TRAINING_LOGS:
                changed, size = sync_file(
                    sftp, REMOTE_LOG_ROOT / log_name, LOCAL_LOG_ROOT / log_name
                )
                print(f"training_logs/{log_name}: downloaded {int(changed)} file, {size} bytes")
    finally:
        client.close()


if __name__ == "__main__":
    main()
