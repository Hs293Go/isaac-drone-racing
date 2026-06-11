"""Enforce the import contract: the core modules must not pull in Isaac Sim.

The README promises these modules are safe to import anywhere (no SimulationApp
required). A stray top-level isaacsim/omni/pxr/carb import would only blow up at
demo time, so each module is imported in a fresh subprocess and the forbidden
namespaces are asserted absent from sys.modules.
"""

import subprocess  # noqa: S404  # fixed argv: sys.executable + a module name from PURE_MODULES
import sys

import pytest

PURE_MODULES = [
    "isaacrace.config",
    "isaacrace.conversions",
    "isaacrace.state",
    "isaacrace.dynamics",
    "isaacrace.perception",
    "isaacrace.course",
    "isaacrace.env_batched",
]

FORBIDDEN = {"isaacsim", "omni", "pxr", "carb"}

_CHECK = f"""
import importlib, sys
importlib.import_module(sys.argv[1])
hit = sorted({{m.split(".")[0] for m in sys.modules}} & {FORBIDDEN!r})
assert not hit, f"{{sys.argv[1]}} transitively imported {{hit}}"
"""


@pytest.mark.parametrize("module", PURE_MODULES)
def test_module_is_isaac_free(module):
    subprocess.run(  # noqa: S603  # argv is a fixed list; `module` comes from PURE_MODULES above
        [sys.executable, "-c", _CHECK, module],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
