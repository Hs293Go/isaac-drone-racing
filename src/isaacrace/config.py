"""Config schemas for the racecourse (`track` group) and run session.

Plain dataclasses — hydra composes conf/*.yaml into a DictConfig; ``from_dict`` builds
these from it (via ``OmegaConf.to_container``). These configurations objects are
Stdlib-only and import-light (no numpy/isaacsim), so configs load without booting Isaac.
"""

from dataclasses import dataclass, field
import math


@dataclass
class RaceTrackConfig:
    """A racecourse consisting of uniformly sized gates.

    The gates are specified by their center locations in NED coordinates and a yaw angle
    (heading) that defines the gate plane's normal vector.
    """

    gate_pos: list[list[float]]  # [N][3] NED gate centres
    gate_yaw: list[float]  # [N] gate headings (units per gate_yaw_unit)
    start_pos: list[float]  # [3] NED start position
    gate_yaw_unit: str = "multiples_pi_2"  # "multiples_pi_2" or "radians"
    gate_size: float = 1.5  # gate window side length (m)
    # Horizontal (x/y) distance from the center beyond which the drone is out of bounds
    bound_xy: float = 5.0
    # Vertical (z) distance above which the drone is out of bounds
    bound_z: float = 7.0

    @classmethod
    def from_dict(cls, d: dict) -> "RaceTrackConfig":
        """Build a ``RaceTrackConfig`` from a plain dict."""
        return cls(**d)


@dataclass
class RandomSpawnConfig:
    """Parameters to randomize a drone's initial pose relative to a gate."""

    dist_back: float = 1.0  # m before the gate along -through
    vel_bounds: float = 0.5  # uniform +/- on each ENU velocity component
    tilt_bounds: float = math.pi / 9  # uniform +/- on roll, pitch
    rate_bounds: float = 0.1  # uniform +/- on each body rate


@dataclass
class DemoConfig:
    """Demo-entrypoint settings (which model to roll out, for how many steps)."""

    model: str | None = None  # None -> the packaged models/race_ppo.zip
    steps: int = 1200
    capture: bool = False  # record the drone's FPV feed to an mp4 (needs fpv.enabled)
    capture_path: str = "fpv.mp4"
    capture_fps: int = 50


@dataclass
class TrainConfig:
    """PPO training hyperparameters for the race environment."""

    total_timesteps: int = 3_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    n_steps: int = 4096
    batch_size: int = 512
    n_epochs: int = 10
    pi: list[int] = field(default_factory=lambda: [128, 128])  # policy net arch
    vf: list[int] = field(default_factory=lambda: [128, 128])  # value net arch
    log_std_init: float = 0.0
    save_freq: int = 200_000
    # domain randomization: resample drone params +-pct each episode (OQCRL-style)
    domain_randomization: bool = False
    param_dr_pct: float = 0.10
    # perception shaping: reward keeping the target gate in the camera FOV (0 = off).
    perception_weight: float = 0.0
    # sim backend: "batched" (vectorized pure-NumPy BatchedRaceEnv, ~100x faster, no
    # Isaac) or "isaac" (single PhysX World, on-rig fidelity). num_envs -> batched.
    backend: str = "batched"
    num_envs: int = 512
    # output basename under train_out/ (a profile can set its own, e.g. via an
    # experiment config, so it doesn't overwrite the baseline model)
    run_name: str = "race_ppo"


@dataclass
class FpvConfig:
    """Front-mounted FPV camera (USD Camera on the body, shown in a viewport)."""

    enabled: bool = False
    resolution: list[int] = field(default_factory=lambda: [480, 320])  # [width, height]
    fov_deg: float = 90.0  # horizontal field of view
    mount: list[float] = field(
        default_factory=lambda: [0.12, 0.0, 0.03]
    )  # FLU body offset (m)
    tilt_deg: float = 25.0  # up-tilt (FPV rigs angle up)


@dataclass
class SessionConfig:
    """Shared sim settings + per-entrypoint demo/train blocks."""

    seed: int = 0
    headless: bool = True
    max_steps: int = 1200
    demo: DemoConfig = field(default_factory=DemoConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fpv: FpvConfig = field(default_factory=FpvConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionConfig":
        """Build a ``SessionConfig`` (with nested demo/train/fpv) from a dict."""
        return cls(
            seed=d.get("seed", 0),
            headless=d.get("headless", True),
            max_steps=d.get("max_steps", 1200),
            demo=DemoConfig(**d.get("demo", {})),
            train=TrainConfig(**d.get("train", {})),
            fpv=FpvConfig(**d.get("fpv", {})),
        )
