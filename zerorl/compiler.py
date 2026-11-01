"""Optional torch.compile wrapper. No-ops when no C++ compiler / Triton is available."""

import sys
import shutil
import torch
from typing import Callable, Any, TypeVar


F = TypeVar("F", bound=Callable[..., Any])

def _cxx_compiler_available() -> bool:
    """True if torch inductor can find a C++ compiler."""
    if sys.platform == "win32":
        return shutil.which("cl") is not None
    return (shutil.which("g++") is not None or
            shutil.which("c++") is not None or
            shutil.which("clang++") is not None)


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _triton_device_supported() -> bool:
    """Triton's GPU backend requires compute capability >= 7.0.

    Triton may be importable yet unusable for the CURRENT device (e.g.
    Pascal sm_61); torch.compile would then raise GPUTooOldForTriton at
    first call, so fast_compile must no-op instead — matching this
    module's documented contract.
    """
    if not torch.cuda.is_available():
        return True  # CPU path; capability check is irrelevant
    try:
        cap = torch.cuda.get_device_capability(torch.cuda.current_device())
        return cap >= (7, 0)
    except Exception:
        return False


def _torch_compile_usable() -> bool:
    """torch.compile CUDA inductor needs Triton on a supported device."""
    if not _cxx_compiler_available():
        return False
    # win32 CPU inductor (MSVC) is unreliable; require CUDA + Triton.
    if sys.platform == "win32":
        return bool(torch.cuda.is_available() and _triton_available())
    if torch.cuda.is_available():
        if not _triton_available():
            return False
        if not _triton_device_supported():
            return False
    return True


def fast_compile(fn: F | None = None,  debug: bool = False, **kwargs) -> F | Callable:
    """Like torch.compile; no-op without a C++ compiler or (on CUDA) without Triton."""
    use_compile = _torch_compile_usable()
    def wrap(f: F) -> F:
        if not use_compile or debug:
            return f
        return torch.compile(f, **kwargs)  # type: ignore[return-value]

    if fn is not None:
        return wrap(fn)
    return wrap
