# Isaac Race

This is a minimum working example of using Isaac Sim to simulate a racing drone
flying through a racecourse using an RL-trained policy, and to train a fresh
policy from scratch on the simulated drone.

![demo](demo.png)

It combined valuable lessons learned from two battle-tested codebases:

- [Pegasus Simulator](https://github.com/PegasusSimulator/PegasusSimulator/tree/main),
  a framework for simulating and controlling multirotors in Isaac Sim.
- The RL-based controller for racing drones that underlies
  [**One Net to Rule Them All: Domain Randomization in Quadcopter Racing Across Different Platforms**](https://github.com/tudelft/optimal_quad_control_RL)

## Running the code

### Dependencies

We use `uv` to manage the Python environment and dependencies. On Ubuntu, these
commands set up `uv` for you:

```bash
sudo apt-get install pipx
pipx ensurepath
pipx install uv
```

### System Prerequisites

`uv` ensures that a compatible Python 3.11 is installed. However, we currently
use Isaac Sim 5.1, which requires an NVIDIA GPU and a compatible driver.
Notably, a driver version of less than 580 is necessary to run Isaac Sim 5.1.

```bash
sudo apt-get install nvidia-driver-580
```

### Running the demo

After you cloned this repo and run the flight demo (using a pre-trained policy)
by running:

```bash
UV_HTTP_TIMEOUT=300 uv run examples/demo.py session.headless=false session.fpv.enabled=true
```

where `session.headless=false` opens a window to watch the flight, and
`session.fpv.enabled=true` adds a second viewport showing the FPV camera view.

..note:: The first run will pull ~GBs of Isaac Sim wheels, so the
`UV_HTTP_TIMEOUT=300` is recommended to avoid a timeout on slower connections.

### Training a new policy

To train a new policy from scratch, run:

```bash
uv run examples/train.py session.train.domain_randomization=true
```

where `session.train.domain_randomization=true` turns on domain randomization.

And run the trained policy in the demo with, for example if the trained model is
at `./train_out/race_ppo_800000_steps.zip`,

```bash
uv run examples/demo.py session.headless=false session.demo.model=./train_out/race_ppo_800000_steps.zip
```

## Design

### Modeling the drone dynamics

The drone dynamics come from the TU Delft 5-inch-quad model. This model
contributes to _racing across different platforms_ by **NOT** using the
classical rotational dynamics formulation, which depends on a inertia tensor
that is hard to measure well, but models the moments acting on the drone as a
function of motor speeds, motor accelerations, and learned parameters.

To reproduce that model in Isaac SIM, our `RacingDrone` treats translation and
rotation separately:

- **Translation** is left to PhysX: each step we apply the model's body-frame
  thrust + drag as a force and let PhysX add gravity and integrate
  position/velocity.
- **Rotation** is integrated **kinematically**: we compute the _exact_ angular
  acceleration using the motor effectiveness formulation, advance the body rates
  one step, and write them directly to the rigid body, bypassing the inertia
  tensor and any gyroscopic effects.

### Handling frame conventions

We and Pegasus Simulator both encountered the challenge that autopilot
frameworks that Pegasus Simulator supports (e.g. PX4, Ardupilot) and the TU
Delft model descends from (Betaflight) and Isaac Sim use different coordinate
conventions:

| Frame | PX4/Ardupilot/Betaflight | Isaac Sim |
| ----- | ------------------------ | --------- |
| World | NED                      | ENU       |
| Body  | FRD                      | FLU       |

We implemented `isaacrace/conversions.py` to handle this by using axes
permutations and sign flips, as opposed to more expensive rotation matrix
multiplications.

### Managing the Isaac Sim dependency

We consume Isaac Sim 5.1 as the pip `isaacsim` package (cp311) from NVIDIA's
index, as opposed to the classic approach of downloading a .zip installation of
the entire Isaac Sim app and patching `PYTHONPATH` to use it. This makes it
possible to control the Isaac Sim version on a per-venv basis. This is only
possible on modern Isaac, since it now ships as ABI-tagged wheels (built for
stock CPython 3.11) on `pypi.nvidia.com`, so they install into your own venv.

## Notes

- `dynamics.py`, `course.py`, `config.py`, `state.py`, `sensors/sensor.py` are
  pure Python (no isaacsim) and safe to import anywhere; `quadrotor.py` and
  `env.py` import `isaacsim`/`pxr`, so a `SimulationApp` must exist **before**
  importing them. `demo.py`/`train.py` are `@hydra.main` entry points that boot
  the app inside `main()`, then import `RaceEnv` — follow that order in any new
  entry point.
- Config is hydra + plain dataclasses: `conf/config.yaml` composes a `track`
  group (the racecourse) and a `session` group (sim + PPO settings);
  `RaceCourse(track_cfg)` turns the track into gate geometry. Swap
  `conf/track/*.yaml` to race a different course.
