"""Preserve 'python -m droidrun.macro' entrypoint."""
import warnings

warnings.warn(
    "Use 'python -m mobilerun.macro' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from mobilerun.macro.cli import macro_cli

macro_cli()
