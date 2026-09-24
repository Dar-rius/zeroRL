"""Shared fixtures for env tests."""

import os
import sys

import pytest


def _mujoco_gl_unavailable() -> bool:
    backend = os.environ.get("MUJOCO_GL", "").lower()
    if backend in {"osmesa", "egl"}:
        return False
    return sys.platform.startswith("linux") and not os.environ.get("DISPLAY")


@pytest.fixture
def mujoco_gl() -> None:
    if _mujoco_gl_unavailable():
        pytest.skip(
            "MuJoCo OpenGL renderer unavailable (headless; set MUJOCO_GL=osmesa|egl)"
        )
