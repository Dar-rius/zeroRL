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


def fast_compile(fn: F | None = None,  debug: bool = False, **kwargs) -> F | Callable:
    """Like torch.compile; no-op when a C++ compiler is not on PATH."""
    use_compile = _cxx_compiler_available()
    def wrap(f: F) -> F:
        if not use_compile or debug:
            return f
        return torch.compile(f, **kwargs)  # type: ignore[return-value]

    if fn is not None:
        return wrap(fn)
    return wrap
