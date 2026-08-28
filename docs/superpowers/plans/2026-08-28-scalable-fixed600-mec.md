# Scalable Fixed600 MEC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parameterize the existing Fixed600 launch path for 7-UAV/60-MD and 5-UAV/80-MD experiments while preserving the current 5-UAV/60-MD command, model shapes, and random streams.

**Architecture:** Keep the MEC population and learning implementations unchanged. Extend the existing PowerShell wrappers with validated scale arguments and Fixed600-specific start resolution, then add environment-side deterministic coordinate validation and CPU regression tests for launch commands, shapes, communication/reconstruction tensors, and the historical random stream.

**Tech Stack:** Python 3, NumPy, PyTorch, pytest, Windows PowerShell 5/7, separated MAPPO/PPO.

**Spec:** `docs/superpowers/specs/2026-08-28-scalable-fixed600-mec-design.md`

## Global Constraints

- Current Fixed600 protocol remains 5 UAVs, 60 MD slots, strict 1+4 candidate arrivals, lifetime 12 in formal launchers, and `max_GUs_in_range=20`.
- Seven-UAV starts remain ordered as `(110,180),(220,180),(330,180),(440,180),(400,400),(550,180),(200,400)`.
- Five-UAV/80-MD means `n_GUs=80` and lifetime 16; active MD count remains admission- and coverage-dependent.
- No environment, reward, PPO, communication, GRU loss, or RNG draw-order changes.
- Existing wrapper names, current default experiment names, and the reliable low-level wrapper's historical lifetime-10 default remain compatible.
- All implementation edits use test-driven development; no 60M training run is required.

---

### Task 1: Expose and validate scale arguments in the launch chain

**Files:**
- Modify: `tests/test_a2a_deadline_p50_launchers.py`
- Modify: `onpolicy/scripts/train/run_dynamic_5uav.ps1`
- Modify: `onpolicy/scripts/train/run_fixed600_200_unreliable_dataplane.ps1`
- Modify: `onpolicy/scripts/train/run_fixed2hotspot_per_agent_noise.ps1`

**Interfaces:**
- Consumes: existing PowerShell parameters `NumUAVs`, `NumGUs`, `MDArrivalsPerRegion`, `MDLifetime`, and `UAVStartPositions` in `run_dynamic_5uav.ps1`.
- Produces: `MaxGUsInRange` forwarding; scale parameters on both Fixed600 wrappers; deterministic five-/seven-UAV start resolution; unchanged default Python arguments.

- [ ] **Step 1: Add failing unreliable-launcher tests**

Add Windows-only tests that invoke the unreliable wrapper through
`_captured_training_command`:

```python
@WINDOWS_ONLY
def test_fixed_unreliable_launcher_supports_seven_uavs(tmp_path):
    command = _captured_training_command(
        TRAIN_SCRIPTS / "run_fixed600_200_unreliable_dataplane.ps1",
        tmp_path,
        "-NumEnvSteps", 1,
        "-RolloutThreads", 1,
        "-NumUAVs", 7,
        "-NumGUs", 60,
        "-MDLifetime", 12,
        "-MaxGUsInRange", 20,
        "-StateReconstruction", "last_obs",
        "-Python", tmp_path / "fake-python.cmd",
        "-ExperimentName", "scaled-seven-uav-test",
    )
    assert "--n_UAVs 7" in command
    assert "--n_GUs 60" in command
    assert "--max_GUs_in_range 20" in command
    assert "--md_lifetime_min 12 --md_lifetime_max 12" in command
    assert (
        "--uav_start_positions 110 180 220 180 330 180 440 180 "
        "400 400 550 180 200 400"
    ) in command
```

Add a default-contract test asserting `--n_UAVs 5`, `--n_GUs 60`,
`--max_GUs_in_range 20`, lifetime 12, and the historical five starts.

- [ ] **Step 2: Add failing reliable-launcher tests**

Add a Windows-only test for the reliable wrapper:

```python
@WINDOWS_ONLY
def test_fixed_reliable_launcher_supports_eighty_md_slots(tmp_path):
    command = _captured_training_command(
        TRAIN_SCRIPTS / "run_fixed2hotspot_per_agent_noise.ps1",
        tmp_path,
        "-NoiseScale", 3,
        "-NumEnvSteps", 1,
        "-RolloutThreads", 1,
        "-NumUAVs", 5,
        "-NumGUs", 80,
        "-MdLifetime", 16,
        "-MaxGUsInRange", 20,
        "-HotspotLayoutMode", "episode_template4_600_200",
        "-ExperimentName", "scaled-eighty-md-test",
    )
    assert "--n_UAVs 5" in command
    assert "--n_GUs 80" in command
    assert "--max_GUs_in_range 20" in command
    assert "--md_lifetime_min 16 --md_lifetime_max 16" in command
    assert "--md_arrivals_per_region 1 4" in command
```

Add rejection tests for `NumGUs < sum(arrivals) * MDLifetime`, a coordinate
count different from `2 * NumUAVs`, and `MaxGUsInRange > NumGUs`.

- [ ] **Step 3: Run the launcher tests and verify RED**

Run:

```powershell
& 'C:\Users\wyj2\.conda\envs\marl\python.exe' -m pytest tests/test_a2a_deadline_p50_launchers.py -q
```

Expected: new tests fail because the Fixed600 wrappers do not recognize the
scale parameters and the common launcher still writes `20` directly.

- [ ] **Step 4: Implement the common launcher contract**

In `run_dynamic_5uav.ps1`, add:

```powershell
[ValidateRange(1, 1000)]
[int]$MaxGUsInRange = 20,
```

Validate it against `NumGUs`, validate every explicit coordinate with
`[double]::IsNaN`, `[double]::IsInfinity`, and the `[0, MapSize]` bounds, and
replace:

```powershell
'--max_GUs_in_range', '20',
```

with:

```powershell
'--max_GUs_in_range', $MaxGUsInRange,
```

- [ ] **Step 5: Implement Fixed600 start resolution and forwarding**

Add scale parameters to both Fixed600 wrappers. Resolve omitted starts without
randomness:

```powershell
if ($UAVStartPositions.Count -eq 0) {
    if ($NumUAVs -eq 5) {
        $UAVStartPositions = @(110, 180, 220, 180, 330, 180, 440, 180, 400, 400)
    } elseif ($NumUAVs -eq 7) {
        $UAVStartPositions = @(
            110, 180, 220, 180, 330, 180, 440, 180,
            400, 400, 550, 180, 200, 400
        )
    } else {
        throw "Fixed600 requires explicit UAVStartPositions for $NumUAVs UAVs."
    }
}
```

Forward `NumUAVs`, `NumGUs`, `MaxGUsInRange`, `MDArrivalsPerRegion`,
`MDLifetime`, and resolved starts. Keep the exact current automatic experiment
name for the historical scale; append `_uav${NumUAVs}_md${NumGUs}_life${MDLifetime}`
only for non-default scales.

- [ ] **Step 6: Run launcher tests and verify GREEN**

Run the Task 1 pytest command again. Expected: all launcher tests pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add -- tests/test_a2a_deadline_p50_launchers.py onpolicy/scripts/train/run_dynamic_5uav.ps1 onpolicy/scripts/train/run_fixed600_200_unreliable_dataplane.ps1 onpolicy/scripts/train/run_fixed2hotspot_per_agent_noise.ps1
git diff --cached --check
git commit -m "feat: parameterize Fixed600 experiment scale"
```

---

### Task 2: Validate explicit starts and lock environment scale contracts

**Files:**
- Modify: `tests/test_dynamic_md.py`
- Modify: `onpolicy/envs/mec/mec.py`

**Interfaces:**
- Consumes: `args.n_UAVs`, `args.n_GUs`, `args.max_GUs_in_range`, map bounds, dynamic-arrival and lifetime arguments.
- Produces: deterministic explicit-start validation and tested 7/60 and 5/80 observation/state/action contracts.

- [ ] **Step 1: Add a failing coordinate-validation test**

```python
def test_explicit_uav_starts_must_be_finite_and_inside_map(self):
    common = dict(
        n_UAVs=5,
        uav_start_positions=[110, 180, 220, 180, 330, 180, 440, 180, 400, 400],
    )
    for invalid in (np.nan, np.inf, -1.0, 601.0):
        positions = list(common["uav_start_positions"])
        positions[0] = invalid
        with self.assertRaises(ValueError):
            MEC(self.make_args(**common, uav_start_positions=positions))
```

- [ ] **Step 2: Add 7/60 and 5/80 scale-contract tests**

Construct Fixed600 environments with layout context, spatial Cartesian actor,
completion-priority ordering, strict `[1, 4]` arrivals, and explicit starts.
Assert:

```python
assert seven.obs_dim == 211
assert seven.state_dim == 1477
assert seven.action_space.spaces[1].shape == (20,)
assert seven.attention_active_mask.shape == (7, 7)
np.testing.assert_array_equal(seven.uav_positions[:, :2], SEVEN_UAV_STARTS)

assert eighty.obs_dim == 211
assert eighty.state_dim == 1055
assert eighty.n_GUs == 80
assert eighty.md_lifetime_max == 16
assert eighty.action_space.spaces[1].shape == (20,)
assert eighty.type_s_schema_bits == 7774
assert eighty.type_s_schema_bits <= eighty.args.state_payload_bits
```

Step each environment once with zero Cartesian flight and finite resource
actions, then assert finite observations, states, and rewards.

- [ ] **Step 3: Run tests and verify RED**

Run:

```powershell
& 'C:\Users\wyj2\.conda\envs\marl\python.exe' -m pytest tests/test_dynamic_md.py -q
```

Expected: coordinate-validation test fails because `MEC` currently accepts
non-finite and out-of-map explicit coordinates. Scale-construction assertions
may already pass and serve as regression locks.

- [ ] **Step 4: Implement minimal deterministic validation**

Immediately after reshaping explicit starts in `MEC.__init__`, reject
non-finite values and coordinates outside `args.x_min_uav`, `args.x_max_uav`,
`args.y_min_uav`, and `args.y_max_uav`. Do not call any RNG or alter reset.

- [ ] **Step 5: Run tests and verify GREEN**

Run the Task 2 pytest command again. Expected: all dynamic-MD tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- tests/test_dynamic_md.py onpolicy/envs/mec/mec.py
git diff --cached --check
git commit -m "test: validate scaled Fixed600 environments"
```

---

### Task 3: Verify communication, GRU, network, and RNG compatibility

**Files:**
- Modify: `tests/test_unreliable_communication.py`
- Modify: `tests/test_md_state_reconstruction.py`
- Modify: `tests/test_spatial_flight_actor.py`

**Interfaces:**
- Consumes: scale-aware `MEC`, `MDStateReconstructor`, spatial actor, ego-query attention critic, Type-S masks, and current seed partitioning.
- Produces: CPU evidence that reliable/unreliable modes and MD-GRU work at both new scales without changing the current model/random contract.

- [ ] **Step 1: Add communication scale tests**

For 7/60, verify Type-S reception and geometric masks are `(7, 7)`, diagonal
self-reception remains true, and critic state reshapes to `(7, 7, 211)`. For
5/80, verify state packets retain 20 MD record slots and Type-S schema fits the
8000-bit payload.

- [ ] **Step 2: Add reconstruction capacity tests**

Construct CPU `MDStateReconstructor` instances and assert:

```python
assert seven_reconstructor.n_uavs == 7
assert seven_reconstructor.capacity == 60
assert len(seven_reconstructor.predictors) == 7

assert eighty_reconstructor.n_uavs == 5
assert eighty_reconstructor.capacity == 80
assert eighty_reconstructor.packet_capacity == 20
assert eighty_reconstructor.session_buffers[0].max_events == 16
```

Use the existing synthetic Type-S packet helper to run one reconstruction for
`last_obs` and `md_gru`; outputs and metadata must be finite.

- [ ] **Step 3: Add network-shape and initialization tests**

With actor-message disabled and `max_GUs_in_range=20`, instantiate the spatial
actor and attention critic for current, 7/60, and 5/80 protocols. Assert the
per-policy actor parameter count is equal across all three, the per-policy
critic parameter count is equal across all three, and the 5/80 state-dict
tensor shapes equal 5/60. Assert only the number of separated policies changes
from five to seven.

- [ ] **Step 4: Add the historical RNG regression**

For two independently constructed 5/60 environments with seed 2 and the exact
current Fixed600 arguments, assert exact equality of reset observations,
states, candidate positions, session IDs, and one deterministic transition.
Also assert the resolved five starts and dimensions `(5, 211)` and `(5, 1055)`.
This comparison detects accidental new RNG consumption without relying on a
platform-specific binary hash.

- [ ] **Step 5: Run focused compatibility tests**

```powershell
& 'C:\Users\wyj2\.conda\envs\marl\python.exe' -m pytest tests/test_unreliable_communication.py tests/test_md_state_reconstruction.py tests/test_spatial_flight_actor.py -q
```

Expected: all focused tests pass. If a new assertion exposes a real genericity
bug, add a separate failing regression test before changing production code.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- tests/test_unreliable_communication.py tests/test_md_state_reconstruction.py tests/test_spatial_flight_actor.py
git diff --cached --check
git commit -m "test: cover scaled communication and reconstruction"
```

---

### Task 4: Full verification and handoff

**Files:**
- Modify only if verification reveals a scoped defect: files already listed in Tasks 1-3.

**Interfaces:**
- Consumes: all implementation commits and the approved design.
- Produces: fresh test evidence, clean tracked diff, and exact future launch examples.

- [ ] **Step 1: Run the complete targeted CPU suite**

```powershell
& 'C:\Users\wyj2\.conda\envs\marl\python.exe' -m pytest tests/test_a2a_deadline_p50_launchers.py tests/test_dynamic_md.py tests/test_cartesian_flight.py tests/test_spatial_flight_actor.py tests/test_unreliable_communication.py tests/test_md_state_reconstruction.py tests/test_reliable_consensus.py -q
```

- [ ] **Step 2: Check formatting and repository scope**

```powershell
git diff --check HEAD~3..HEAD
git status --short --untracked-files=no
git log -4 --oneline
```

Confirm only approved tracked files changed. Existing untracked experiment
outputs remain untouched.

- [ ] **Step 3: Record exact future commands**

Prepare concise PowerShell examples for:

```powershell
# 7 UAV / 60 MD
-NumUAVs 7 -NumGUs 60 -MDLifetime 12 -MaxGUsInRange 20

# 5 UAV / 80 MD
-NumUAVs 5 -NumGUs 80 -MDLifetime 16 -MaxGUsInRange 20
```

State that full 60M experiments still require matched seeds and algorithms and
that no training was launched during implementation.
