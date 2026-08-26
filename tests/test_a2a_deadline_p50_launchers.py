import os
import subprocess
from pathlib import Path

import pytest
from scipy.special import gammainc


REPO = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = REPO / "onpolicy" / "scripts" / "train"
STATE_DEADLINE_MS = 10.175730095548
ADVANTAGE_DEADLINE_MS = 18.175730095548
WINDOWS_ONLY = pytest.mark.skipif(os.name != "nt", reason="PowerShell launcher test")


def _captured_training_command(script, tmp_path, *arguments):
    capture = tmp_path / "python-arguments.txt"
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        '@echo off\r\necho %*>>"%CAPTURE_ARGS%"\r\nexit /b 0\r\n',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["MARL_PYTHON"] = str(fake_python)
    environment["CAPTURE_ARGS"] = str(capture)
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *map(str, arguments),
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return capture.read_text(encoding="utf-8").splitlines()[-1]


@WINDOWS_ONLY
def test_fixed_unreliable_launcher_forwards_a2a_deadlines(tmp_path):
    command = _captured_training_command(
        TRAIN_SCRIPTS / "run_fixed600_200_unreliable_dataplane.ps1",
        tmp_path,
        "-NumEnvSteps",
        1,
        "-RolloutThreads",
        1,
        "-StateReconstruction",
        "last_obs",
        "-StateDeadlineMs",
        10.125,
        "-AdvantageDeadlineMs",
        18.125,
        "-Python",
        tmp_path / "fake-python.cmd",
        "-ExperimentName",
        "deadline-forwarding-test",
    )
    assert "--state_deadline_ms 10.125" in command
    assert "--advantage_deadline_ms 18.125" in command


@pytest.mark.parametrize("seed", [1, 11, 21])
@WINDOWS_ONLY
def test_local_last_obs_a2a_p50_launcher_emits_matched_deadlines(seed, tmp_path):
    script = TRAIN_SCRIPTS / (
        "run_local_f600_b0_unreldp_unreliable_last_obs_"
        f"deadlineon_a2atimelyp50_seed{seed}_cuda.ps1"
    )
    command = _captured_training_command(script, tmp_path)

    assert f"--seed {seed}" in command
    assert "--state_reconstruction last_obs" in command
    assert f"--state_deadline_ms {STATE_DEADLINE_MS}" in command
    assert f"--advantage_deadline_ms {ADVANTAGE_DEADLINE_MS}" in command
    assert "--disable_offload_deadline_filter" not in command
    assert "--uav_reset_curriculum" not in command


def test_a2a_p50_deadlines_leave_half_the_gamma_latency_mass():
    state_transmission_ms = 8.0
    advantage_transmission_ms = 16.0
    for deadline_ms, transmission_ms in (
        (STATE_DEADLINE_MS, state_transmission_ms),
        (ADVANTAGE_DEADLINE_MS, advantage_transmission_ms),
    ):
        probability = gammainc(2.5, deadline_ms - transmission_ms)
        assert probability == pytest.approx(0.5, abs=1e-12)


def test_linux_seed_runner_accepts_a2a_deadline_overrides(tmp_path):
    bash = (
        Path(r"C:\Program Files\Git\bin\bash.exe")
        if os.name == "nt"
        else Path("/bin/bash")
    )
    capture = tmp_path / "python-arguments.txt"
    fake_python = tmp_path / "fake-python.sh"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > \"$CAPTURE_ARGS\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["MARL_PYTHON"] = fake_python.as_posix()
    environment["CAPTURE_ARGS"] = capture.as_posix()
    result = subprocess.run(
        [
            str(bash),
            (TRAIN_SCRIPTS / "run_fixed600_200_seed.sh").as_posix(),
            "unreliable",
            "md_gru",
            "1",
            "linux-a2a-p50-test",
            "off",
            "enabled",
            str(STATE_DEADLINE_MS),
            str(ADVANTAGE_DEADLINE_MS),
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    command = capture.read_text(encoding="utf-8")
    assert f"--state_deadline_ms {STATE_DEADLINE_MS}" in command
    assert f"--advantage_deadline_ms {ADVANTAGE_DEADLINE_MS}" in command
