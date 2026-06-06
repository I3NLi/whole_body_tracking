"""Top-level package for whole_body_tracking.

Keep this module side-effect free so utility scripts can import lightweight
submodules such as ``whole_body_tracking.robots.g1`` without triggering the
full IsaacLab task registry walk.
"""

from importlib import import_module


def register_tasks():
    """Import task modules and trigger environment registration explicitly."""
    return import_module(".tasks", __name__)


__all__ = ["register_tasks"]
