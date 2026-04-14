"""Compatibility shim: droidrun -> mobilerun.

Uses a PEP 451 meta-path finder (find_spec) to lazily alias droidrun.*
imports to mobilerun.* on demand. Compatible with Python 3.11-3.13+.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings

warnings.warn(
    "The 'droidrun' package has been renamed to 'mobilerun'. "
    "Please update your imports. This compatibility shim will be "
    "removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

# DO NOT import mobilerun here — that would eagerly trigger the full
# package initialization (log handlers, agent/config/macro/tool imports).
# Instead, use lazy __getattr__ for top-level symbols and a PEP 451
# meta-path finder for submodule aliasing.


class _DroidrunAliasLoader(importlib.abc.Loader):
    """PEP 451 loader: returns the real mobilerun.* module object."""

    def __init__(self, real_name):
        self._real_name = real_name

    def create_module(self, spec):
        # Import the real module and return it directly.
        # sys.modules[droidrun.X] will point to the SAME object.
        real_mod = importlib.import_module(self._real_name)
        sys.modules[spec.name] = real_mod
        return real_mod

    def exec_module(self, module):
        pass  # already fully initialized


class _DroidrunAliasFinder(importlib.abc.MetaPathFinder):
    """PEP 451 finder: lazily alias droidrun.* -> mobilerun.*.

    Uses find_spec (not find_module) — required for Python 3.12+/3.13.
    Physical compat files (__main__.py, macro/) are excluded.
    """

    _active = False  # re-entrancy guard

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith("droidrun."):
            return None
        if fullname in sys.modules:
            return None
        if self._active:
            return None

        # Check if this is a physical file in the compat tree
        self._active = True
        try:
            for finder in sys.meta_path:
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(fullname, path, target)
                if spec is not None:
                    return None  # physical file exists — don't intercept
        finally:
            self._active = False

        # Map to mobilerun.* and verify it exists
        new_name = "mobilerun" + fullname[len("droidrun"):]
        self._active = True
        try:
            real_spec = importlib.util.find_spec(new_name)
        except (ImportError, ValueError):
            real_spec = None
        finally:
            self._active = False

        if real_spec is None:
            return None

        # Build alias spec with correct package metadata
        alias_spec = importlib.machinery.ModuleSpec(
            fullname,
            _DroidrunAliasLoader(new_name),
            origin=real_spec.origin,
            is_package=real_spec.submodule_search_locations is not None,
        )
        if real_spec.submodule_search_locations is not None:
            alias_spec.submodule_search_locations = list(
                real_spec.submodule_search_locations
            )
        return alias_spec


sys.meta_path.append(_DroidrunAliasFinder())


# Lazy top-level symbol forwarding — no `import mobilerun` at module scope.
# `import droidrun` alone does NOT trigger mobilerun's full init.
# Only `from droidrun import DroidAgent` (or similar) triggers it.
def __getattr__(name):
    """Forward any attribute access to mobilerun (loaded lazily)."""
    import mobilerun as _real

    return getattr(_real, name)
