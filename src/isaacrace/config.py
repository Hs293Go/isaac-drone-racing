"""Config schemas for the racecourse (`track` group) and run session.

Plain dataclasses — hydra composes conf/*.yaml into a DictConfig; ``from_dict``
builds these from it (via ``OmegaConf.to_container``). Stdlib-only and
import-light (no numpy/isaacsim), so configs load without booting Isaac. (OQCRL
uses pydantic here; for this MWE, validating a few trusted config files doesn't
warrant the heavier dependency — omegaconf already coerces types.)
"""

from dataclasses import dataclass, field


@dataclass
class RaceTrackConfig:
    """A gate racecourse in the OQCRL NED frame (z down; gates at z=-1.5 => 1.5m up)."""

    gate_pos: list[list[float]]  # [N][3] NED gate centres
    gate_yaw: list[float]  # [N] gate headings (units per gate_yaw_unit)
    start_pos: list[float]  # [3] NED start position
    gate_yaw_unit: str = "multiples_pi_2"  # "multiples_pi_2" or "radians"
    gate_size: float = 1.5  # gate window side length (m)

    @classmethod
    def from_dict(cls, d: dict) -> "RaceTrackConfig":
        """Build a ``RaceTrackConfig`` from a plain dict."""
        return cls(**d)


@dataclass
class DemoConfig:
    """Demo-entrypoint settings (which model to roll out, for how many steps)."""

    model: str | None = None  # None -> the packaged models/race_ppo.zip
    steps: int = 1200


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
