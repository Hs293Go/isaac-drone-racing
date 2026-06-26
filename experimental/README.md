# experimental/ — UVP (vision-in-the-loop), under active exploration

In-flight, **not-yet-validated** work toward racing from pixels. Plain scripts (run by path,
not a package) so in-progress code carries no packaging ceremony. They build on the validated
stack: `import isaacracelab` (the GPU racer) + `import isaacrace` (the framework-free core).

When a piece races reliably it graduates into `src/isaacracelab/` (with parity/eval coverage);
the canonical docs are in [`../src/isaacracelab/README.md`](../src/isaacracelab/README.md).

## Current direction — modular perception→control

End-to-end pixels→motors RL was unstable (survive-not-race). The live approach decomposes it:
a supervised CNN predicts the gate-relative **state** from the FPV image (free sim labels),
then the existing state racer controls. Perception works (~5–10 cm gate accuracy); the
**perception–control integration** (covariate shift + a controller too brittle for estimated
state) is the open problem — see the parent README's status section.

## Files

- `vision_env.py` — `VisionRacerEnv` (image+proprio obs); `camera_smoke.py` — render throughput.
- `vision_train.py` / `vision_eval.py` / `vision_check.py` / `vision_probe.py` — end-to-end RL + probes.
- `perception_net.py` — shared CNN (image→state); `perception_train.py` (supervised + DAgger);
  `perception_eval.py` (composed perception→control, with `--oracle`).

Run with the same prefix as the canonical track:

    env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES uv run --group isaaclab python -u \
        experimental/<script>.py --headless --enable_cameras
