# Scalable Fixed600 MEC Experiment Design

**Date:** 2026-08-28

## Goal

Extend the existing Fixed600-200 experiment launch path so it can run the
current 5-UAV/60-MD protocol, a 7-UAV/60-MD protocol, and a 5-UAV/80-MD
protocol without changing the behavior, random streams, or policy shapes of
the current default experiment.

## Scope

This change parameterizes the existing launch path and adds validation and
regression coverage. It does not change the dynamic-MD population model,
communication model, reward, PPO implementation, GRU loss, or environment
random-number generation.

The supported scale contracts are:

| Protocol | UAVs | MD slots | Arrivals per slot | Lifetime | Per-UAV MD cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current default | 5 | 60 | 1+4 | 12 | 20 |
| UAV scale | 7 | 60 | 1+4 | 12 | 20 |
| MD scale | 5 | 80 | 1+4 | 16 | 20 |

Here, "80 MD" means a nominal dynamic-population capacity of 80 slots. The
number of active MDs remains determined by geometric admission, lifetime, and
early departure from all UAV coverage. The environment must not force exactly
80 active MDs in every slot.

## Fixed600-200 Start Layouts

The existing five-UAV start order remains:

```text
(110, 180), (220, 180), (330, 180), (440, 180), (400, 400)
```

The seven-UAV protocol uses this exact ordered list:

```text
(110, 180), (220, 180), (330, 180), (440, 180),
(400, 400), (550, 180), (200, 400)
```

Order is part of the experiment contract because separated MAPPO constructs
one policy per UAV ID. Launchers must not sort or shuffle explicit starts.
These Fixed600-specific defaults belong in experiment launch code, not in the
generic MEC environment.

## Launcher Architecture

`onpolicy/scripts/train/run_dynamic_5uav.ps1` remains the common launcher and
keeps its historical filename for compatibility. It already accepts
`NumUAVs`, `NumGUs`, `MDArrivalsPerRegion`, `MDLifetime`, and explicit UAV
starts. It will additionally expose `MaxGUsInRange`, with a default of 20,
instead of writing 20 directly into the Python command.

The Fixed600-200 reliable and unreliable wrappers will expose:

- `NumUAVs`, default 5;
- `NumGUs`, default 60;
- `MDLifetime`; the unreliable Fixed600 wrapper defaults to 12, while the
  lower-level reliable wrapper retains its historical default of 10 for
  backward compatibility. Formal Fixed600 scale launchers pass 12 or 16
  explicitly;
- `MaxGUsInRange`, default 20;
- `MDArrivalsPerRegion`, default `(1, 4)`;
- `UAVStartPositions`, resolved to the ordered five- or seven-UAV Fixed600
  layout when omitted.

An unknown UAV count without explicit start positions is rejected. Custom
positions remain supported when exactly `2 * NumUAVs` finite coordinates are
provided inside the configured UAV map bounds.

Automatic experiment names include the UAV count, MD capacity, and lifetime
for non-default scales so results cannot silently share a directory. The
historical default name remains unchanged to preserve current scripts and
result discovery.

## Validation Contract

Launchers reject invalid configurations before starting Python:

```text
NumUAVs >= 2
1 <= MaxGUsInRange <= NumGUs
NumGUs >= sum(MDArrivalsPerRegion) * MDLifetime
len(UAVStartPositions) == 2 * NumUAVs
all start coordinates are finite and within the UAV map bounds
```

The existing Python-side dynamic-MD capacity check remains as a second line of
defense. MEC environment validation will also reject non-finite or out-of-map
explicit starts. No validation path may consume random numbers.

## Environment and Model Data Flow

The common launcher passes scale values to `train_mec.py`. `MEC` allocates
UAV-major arrays from `n_UAVs` and MD-slot arrays from `n_GUs`. With
`all_uav_k_hops`, `max_UAVs_obs_concat` and `max_UAVs_in_neighbor` equal
`n_UAVs`.

The local actor input and action contract use `max_GUs_in_range=20` in all
three protocols. Consequently:

- actor observation width remains 211 for the selected Fixed600 spatial
  actor configuration;
- the per-UAV action width remains 62;
- 5-UAV/80-MD has the same actor and critic network shapes as 5-UAV/60-MD;
- 7-UAV/60-MD increases attention tokens from 5 to 7 and creates seven
  separated policies, while the parameter count of each attention critic and
  actor remains unchanged.

For unreliable communication, Type-S and Type-A tensors already derive their
UAV axes from `n_UAVs`. `MDStateReconstructor` already creates one predictor
and optimizer per receiver UAV. Its online capacity is
`max(n_GUs, arrivals * lifetime)`, so it becomes 80 for the MD-scale protocol;
its per-session event capacity follows `md_lifetime_max`, becoming 16. The
8000-bit Type-S payload remains large enough: the audited schema budgets are
7773 bits for 5/60 and 7/60, and 7774 bits for 5/80.

## Randomness and Backward Compatibility

The change must not add, remove, or reorder random draws in `MEC`. In
particular, start-layout resolution and validation are deterministic launcher
operations.

For the default 5-UAV/60-MD protocol, the resolved Python argument values and
their effective defaults remain unchanged. A fixed-seed regression test locks
the initial UAV positions, candidate MD positions, observations, state, and a
deterministic one-step transition.

Expected experiment-variable divergence is not a regression:

- 7 UAVs change coverage admission and the communication graph immediately;
- lifetime 16 retains sessions beyond slot 12, so 5/80 diverges from 5/60 as
  the population histories separate.

Existing 5-UAV/60-MD checkpoints remain valid for the existing protocol.
Five-UAV/80-MD has compatible neural tensor shapes but should train from
scratch for a clean scale comparison unless warm-starting is explicitly part
of a later protocol. Seven-UAV checkpoints are not interchangeable with
five-UAV separated-policy checkpoints because the number of policy files and
agent identities differs.

## Tests

Implementation follows test-driven development.

1. Launcher tests first demonstrate that scale parameters are not yet exposed
   by the Fixed600 wrappers, then verify the resolved commands for 7/60 and
   5/80.
2. A default launcher regression verifies that 5/60 still emits the historical
   scale, start positions, lifetime, per-UAV MD cap, PPO, and algorithm flags.
3. Environment tests verify exact seven-UAV start order, observation/state/
   action shapes, attention-mask shape, communication-mask shape, and a finite
   one-step transition.
4. Environment tests verify 5/80 capacity, lifetime 16, unchanged local actor
   and action widths, and sufficient Type-S schema budget.
5. Reconstruction tests construct last-observation and MD-GRU paths for both
   added scales on CPU and verify receiver/session capacities and finite
   outputs.
6. The related pytest suite and PowerShell dry-run launch tests run before the
   change is declared complete.

No 60M training run is part of this implementation. Full training starts only
after the CPU construction and regression gates pass.
