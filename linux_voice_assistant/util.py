"""Utility methods."""

import uuid
import logging
from collections.abc import Callable
from typing import Optional
import shlex
import subprocess
import os

_LOGGER = logging.getLogger(__name__)


def get_mac() -> str:
    mac = uuid.getnode()
    mac_str = ":".join(f"{(mac >> i) & 0xff:02x}" for i in range(40, -1, -8))
    return mac_str


def call_all(*callables: Optional[Callable[[], None]]) -> None:
    for item in filter(None, callables):
        item()


def run_command(command: Optional[str], env: Optional[dict] = None) -> None:
    """Run a shell-like command in a non-blocking way."""
    if not command:
        return
    _LOGGER.debug("Running %s", command)
    try:
        merged_env = None
        if env:
            merged_env = dict(os.environ)
            merged_env.update(env)

        subprocess.Popen(
            shlex.split(command),
            close_fds=True,
            env=merged_env,
        )
    except Exception:
        _LOGGER.exception("Failed to run command: %s", command)
