"""isaacracelab — GPU-parallel PhysX drone-racing training on Isaac Lab.

The validated second track, sibling to the framework-free `isaacrace` core: a
`DirectRLEnv` classical-plant racer plus parity-verified torch ports of the core math.
Depends on `isaacrace` and the optional `isaaclab` group (`uv sync --group isaaclab`).

Submodules that import Isaac Lab (e.g. `racer_env`) must be imported AFTER an
AppLauncher/SimulationApp boots — so this `__init__` stays import-light on purpose.
"""
