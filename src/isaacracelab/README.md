# Isaac Lab track (opt-in)

GPU-parallel PhysX training via **Isaac Lab 2.3.x** (built on our pinned Isaac Sim 5.1).
This is a *second track* alongside the framework-free core — it trains on the **classical
(PhysX-native) dynamics**, the formulation Isaac Lab's GPU pipeline requires. Kept out of
the core deps so the MWE stays framework-free.

## Status: canonical vs exploratory

**The state-based GPU training stack is validated (canonical).** The torch ports are
parity-verified against the NumPy core (`test_parity.py`); the classical-plant racer trains
to **51 gates / 97% survival at ~257k steps/s** — the *same order of magnitude* as the
analytic NumPy twin (~375k steps/s), where single-env Isaac was ~1500× slower. Headline:
training on the real PhysX engine is now computationally competitive, not just a validation
step. Files (this package, `src/isaacracelab/`): `racer_env.py`, `dynamics_torch.py`,
`course_torch.py`, `test_parity.py`, `train.py`, `eval_compare.py`, `smoke.py`. The frame
conversions are shared (not duplicated): the backend-agnostic `isaacrace.conversions` serves
both NumPy and torch. Run the scripts as modules, e.g. `python -m isaacracelab.train`.

**Vision-in-the-loop (UVP) is under active exploration — not yet validated.** End-to-end
pixels→motors RL is unstable and survive-not-race; the current direction is a **modular
perception→control** architecture (a supervised image→state net feeding the state racer).
The perception half works (5–10 cm gate accuracy); the perception–control *integration* is
the open problem. Files live in the top-level `experimental/` (scripts, run by path —
`python experimental/vision_train.py`): `vision_*.py`, `perception_*.py`, `camera_smoke.py`.
They import `isaacracelab` (the racer) and `isaacrace` (the core); they graduate into this
package when validated.

## Install

Isaac Lab is an **opt-in dependency group** (`isaaclab` in `pyproject.toml`). It resolved
clean against the existing stack — `torch 2.7.0`, `isaacsim 5.1.0.0`, `sb3 2.8.0` all
unchanged — and pulls Isaac Lab 2.3.2 + `rsl-rl-lib`, `skrl`, `warp-lang`, onnx, etc.

```bash
uv sync --group isaaclab
```

> NVIDIA validates the Isaac Sim 5.1 / Isaac Lab 2.3 stack with **torch cu128**; uv resolved
> **cu126** here (forward-compatible on a 12.8 driver). If you hit CUDA-runtime oddities,
> align torch to cu128 — but note that does **not** fix the `cuInit` issue below (that's a
> host fault, below torch).

## Prerequisite: working CUDA runtime

Isaac Lab's GPU pipeline needs a working CUDA **driver API** (`cuInit`). The core project
never needed this (it runs torch on CPU and single-env Isaac PhysX on CPU), so verify it:

```bash
python3 -c "import ctypes; print('cuInit ->', ctypes.CDLL('libcuda.so.1').cuInit(0))"   # 0 = OK
```

If this returns `999` (CUDA_ERROR_UNKNOWN) while `nvidia-smi` works and `/dev/nvidia-uvm`
exists and the kernel-module / userspace / `libcuda` versions all match, the GPU runtime is
in a bad UVM state. The fix that worked here (no reboot needed) is **reloading the UVM
kernel module**:

```bash
sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm
```

Failing that, reboot, or `sudo nvidia-smi -pm 1`. *This is a host issue, not the install.*

## Runner

The pip-only Isaac Lab package ships **no** training/play scripts, so we provide our own.

- **`smoke.py`** — boot Isaac Lab, make a GPU-parallel stock task, step it. Verifies the
  whole pipeline before we write a custom drone task. (It already boots, registers tasks,
  finds `Isaac-Quadcopter-Direct-v0`, and parses its cfg here — it only stops at the host
  `cuInit` fault above.)

```bash
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab \
    python -m isaacracelab.smoke --task Isaac-Quadcopter-Direct-v0 --num_envs 64 --headless
# results -> /tmp/isaaclab_smoke.txt  (Isaac swallows stdout)
```

## DirectRLEnv spike (classical plant, GPU-parallel)

A custom GPU-parallel racing task reusing isaacrace's classical plant + course rules. The
math layer is ported NumPy→torch and **verified bit-for-bit against the NumPy originals**;
the Isaac Lab env/runner are structurally complete but **not yet run** (host `cuInit`).

| file | what | status |
|------|------|--------|
| `dynamics_torch.py` | OQCRL motor lag + per-rotor thrust + drag/torque | ✅ parity-verified |
| `course_torch.py` | `RaceCourseTorch`: build_obs / gate / reward / spawn | ✅ parity-verified |
| `isaacrace.conversions` (shared) | ENU↔NED, FLU↔FRD, quat↔euler, quat-rotate — backend-agnostic | ✅ parity-verified |
| `test_parity.py` | torch-vs-NumPy CPU parity (18 checks) | ✅ runs on CPU |
| `racer_env.py` | `RacerEnv(DirectRLEnv)` + `RacerEnvCfg` (classical plant) | ✅ validated (trains to flight) |
| `train.py` | rsl_rl PPO runner | ✅ runs end-to-end |
| `eval_compare.py` | baseline-vs-GPU policy comparison (gates/survival) | ✅ runs |
| `camera_smoke.py` | TiledCamera rendering-throughput feasibility | ✅ runs |
| `vision_env.py` | `VisionRacerEnv`: image+proprio obs, rendered gates | ✅ verified |
| `vision_check.py` | vision obs/render sanity check | ✅ runs |
| `vision_train.py` | sb3 PPO + CNN+MLP extractor (image+proprio) | ✅ trains (learning) |

The classical plant is **pre-summed to one body wrench**: per-rotor thrusts at the arms
become a net FLU force + `r×F` torque (frame-linear, so identical to applying 4 forces) +
the yaw reaction couple + body drag — PhysX still integrates rotation through the real
inertia tensor + gyroscopic coupling. `# VERIFY` marks the spots in `racer_env.py` needing
on-GPU confirmation of the exact Isaac Lab 2.3 API / frame conventions (wrench frame,
`root_lin_vel_b`, `find_bodies`, mass read).

Run the parity test now (no GPU needed):

```bash
env -u PYTHONPATH uv run --group isaaclab python -m isaacracelab.test_parity
```

Train (headless + **UNBUFFERED** — Isaac's hard exit at `app.close()` discards Python's
buffered stdout, so without `-u` the training logs vanish and it looks like a hang):

```bash
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 uv run --group isaaclab \
    python -u -m isaacracelab.train --num_envs 4096 --headless
```

Gotchas found during bring-up:
- **No `--headless`** → a GUI window whose unresponsive first-frame triggers the WM
  "application not responding — kill?" dialog. Always train headless.
- **Buffered stdout** is lost on the hard exit — use `PYTHONUNBUFFERED=1 python -u`.
- racer.usd is a **0-DOF articulation** (props are fixed joints), so the cfg sets empty
  `init_state.joint_pos`/`joint_vel` to skip Isaac Lab's default `.*` joint matching.

**Validated (2026-06-24, RTX 3080).** PPO over 200 iterations at 2048 envs (~257k
steps/s, 9.8M steps in ~41 s) learns stable flight: mean episode length **11 → 1155 / 1200
steps**, reward **−0.25 → +18**. That curve validates the GPU classical plant end-to-end
(body-local wrench, `root_lin_vel_b`, obs/reward/reset) — a wrong frame can't learn to fly.
Reference: `kousheekc/isaac_drone_racer`.

**Baseline-vs-GPU comparison** (`eval_compare.py`, 256 random spawns on the GPU classical
plant — same robustness metric as the single-env study):

| policy | gates | survival | ep_len |
|--------|------:|---------:|-------:|
| random | 0.2 | 0% | 80 |
| OQCRL baseline (sb3, effectiveness-model) | 22.4 | 21% | 688 |
| single-env classical fine-tune (sb3) | 26.1 | 64% | 792 |
| GPU survivor (rsl_rl, 200 it, progress-only) | 10.4 | 95% | 1144 |
| **GPU racer (rsl_rl, 800 it + gate bonus)** | **51.0** | **97%** | **1165** |

Takeaways: (1) **env parity** — the single-env-fine-tuned classical policy scores ~64%
survival here vs ~65% on the single-env classical plant, so the GPU port faithfully
reproduces it (independent of the learning-curve check). (2) **Reward shaping + budget
matter, not the plant** — the 200-it progress-only policy was a cautious survivor (95%
survival, only 10.4 gates: a "survive-and-creep" optimum). Adding a **+10 gate-pass bonus**,
trimming the rate penalty (0.001→0.0005), and training to 800 iters (cheap — ~3 min at 257k
steps/s) turns it into the **best policy on both axes**: 51 gates (~2× the single-env
fine-tune) at 97% survival. `train.py --gate_bonus`/`--rate_penalty`/`--run_name` control it.

The GPU racer is a *classical-plant* policy (the GPU track's bet), distinct from the
effectiveness-model policy the NumPy twin trains. (Spawns are unseeded, so the sb3 rows carry
a few-% run-to-run noise; the ordering is robust.)

## Vision-in-the-loop feasibility (`camera_smoke.py`)

Rendering throughput with an FPV TiledCamera (RGB) on the racer, RTX 3080 / 10 GB:

| config | envs | steps/s | GPU mem |
|--------|-----:|--------:|--------:|
| physics-only | 128 | 25,091 | 4.9 GB |
| RGB 64×64 | 128 | 7,911 | 6.1 GB |
| RGB 64×64 | 256 | 12,881 | 6.4 GB |

**Verdict: feasible — go with TiledCamera RGB.** No Blackwell hang (Ampere is clear); RGB
renders `(N, 64, 64, 3)`. The rendering tax is **~3×** (not the order-of-magnitude feared),
it **scales** (128→256 envs: 7.9k→12.9k steps/s — tiled rendering amortizes), and **VRAM is
mostly fixed** (6.1→6.4 GB), leaving headroom for ~512 envs on 10 GB. At 12.9k steps/s a
vision policy (~50–100M steps) trains in ~1–2 h. RGB beats depth here because the gates are
**color-coded** (green start, orange rest). The `RayCasterCamera` (Warp depth) fallback is
only worth it past ~512 envs or if RGB throughput became the wall — neither applies, so it
wasn't measured. `racer_env` now takes an optional `tiled_camera` cfg (default off; the
state-based path is untouched).

## Vision-in-the-loop (built)

`VisionRacerEnv` (`vision_env.py`) races from **pixels + proprioception**: a 64×64 FPV RGB
image concatenated with a 13-dim ego-state vector (body lin vel, body rates, projected
gravity, motor states). Two things make this real vision, not state-in-disguise: the **gates
are rendered** colored frames (so the camera has something to fly toward — the state env
spawns none), and the **gate-relative features are omitted** (the policy must read the gates
from the image). Plant/reward/reset are inherited from `RacerEnv`.

`vision_train.py` trains it with **sb3 PPO + a custom CNN(image)+MLP(proprio) extractor**
(sb3 for the familiar extractor API; rendering caps throughput anyway — skrl is the
GPU-resident option for big runs). It **learns from pixels**: over 10M steps (256 envs) the
mean episode length reached **1120/1200** (reward −0.8 — nearly the full course), proving the
policy reads the gates from the image. But the PPO is **unstable** — it oscillates (ep_len
647→631→366→960→310) and the *final* checkpoint caught a collapse. Fixes are in: `target_kl`,
longer rollouts (`n_steps=32`), and a **best-ep-length checkpoint** (`race_vision_best`) so
the saved policy is the peak, not the final. A flight-grade racer is a multi-hour run; the
pipeline is proven and clearly learning.

```bash
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
    experimental/vision_check.py --num_envs 16 --headless --enable_cameras   # verify obs
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
    experimental/vision_train.py --num_envs 256 --headless --enable_cameras  # train
```
