from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
KERNEL_ROOT = THIS_DIR.parent
if str(KERNEL_ROOT) not in sys.path:
    sys.path.append(str(KERNEL_ROOT))

try:
    from attn_kernel.attn_kernel_v1210_fused_bsz_fp8fp8 import attn_forward_decode_fp8fp8 as _kernel
except Exception as exc:  # pragma: no cover - used for runtime diagnostics
    _IMPORT_ERROR = exc

    def attn_forward_decode_fp8fp8(*args: Any, **kwargs: Any):
        raise RuntimeError(
            "Failed to import fp8fp8 decode kernel. "
            "Ensure Triton and kernel dependencies are available."
        ) from _IMPORT_ERROR
else:

    def attn_forward_decode_fp8fp8(*args: Any, **kwargs: Any):
        return _kernel(*args, **kwargs)
