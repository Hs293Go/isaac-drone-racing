# Permissive stub for the compiled pybind binding
# omni/isaac/dynamic_control/_dynamic_control*.so, which ty cannot introspect.
# gen_isaac_lsp.py copies this beside the .so each run (a separate path can't
# override a regular package). Everything it exposes is typed as Any; flesh out
# real signatures here if you want them checked.
from typing import Any

def __getattr__(name: str) -> Any: ...
