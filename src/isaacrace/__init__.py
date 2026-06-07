"""isaacrace — a minimal Isaac Sim (PhysX) drone RL package (figure-8 race).

Two modules:
  - ``isaacrace.dynamics``  pure-NumPy OQCRL 5-inch quad math + the F8 gate course
                            (safe to import anywhere).
  - ``isaacrace.env``       ``RaceEnv(gym.Env)`` — imports isaacsim, so a
                            ``SimulationApp`` must be constructed FIRST. Always
                            reached via ``isaacrace.demo`` / ``isaacrace.train``,
                            which boot the app before importing the env.

Do NOT ``from isaacrace import env`` at package import time — that would trigger
the isaacsim import before the app exists. ``__init__`` therefore exposes only
``dynamics``.
"""

__version__ = "0.1.0"

from isaacrace import (
    dynamics,
)
