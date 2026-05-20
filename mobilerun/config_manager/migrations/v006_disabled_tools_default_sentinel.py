"""Migration v6: Treat default-shaped ``tools.disabled_tools`` as the sentinel.

Older generated configs (v5) shipped the literal default list
``[click_at, click_area, long_press_at]``. The schema now uses ``None`` as the
"use framework default" sentinel — an explicit list is honored verbatim and
disables the vision auto-unmask for ``click_at`` (and raises in
screenshot-only modes when coordinate tools are listed).

Migration rules for ``tools.disabled_tools``:

* Exact match for the old default list → convert to ``None`` (implicit default).
* Superset of the old default (e.g. ``[click_at, click_area, long_press_at,
  wait]``) → strip the three legacy coordinate entries and keep the user's
  additions. If nothing remains, fall through to ``None``.
* Anything else (custom list, missing entries, ``None``) → left untouched.
"""

from typing import Any, Dict

VERSION = 6

_OLD_DEFAULT = {"click_at", "click_area", "long_press_at"}


def migrate(config: Dict[str, Any]) -> Dict[str, Any]:
    tools = config.get("tools")
    if not isinstance(tools, dict):
        return config

    disabled = tools.get("disabled_tools")
    if not isinstance(disabled, list):
        return config

    if not _OLD_DEFAULT.issubset(set(disabled)):
        # User removed entries from the old default — treat as explicit choice.
        return config

    # Drop legacy default entries while preserving user additions and order.
    extras = [t for t in disabled if t not in _OLD_DEFAULT]
    tools["disabled_tools"] = extras if extras else None

    return config
