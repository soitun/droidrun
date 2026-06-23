"""Compatibility shim for the iOS Portal driver.

The implementation lives in ``mobilerun-core-local`` so local drivers are
owned by one package. This module preserves the historical
``mobilerun.tools.driver.ios`` import path used by the framework, CLI, and docs.
"""

from mobilerun_core_local.driver.ios.http import (
    SYSTEM_APP_LABELS,
    IOSDriver,
    IOSPortalDriver,
    discover_ios_portal,
    validate_ios_portal_url,
)

__all__ = [
    "IOSDriver",
    "IOSPortalDriver",
    "SYSTEM_APP_LABELS",
    "discover_ios_portal",
    "validate_ios_portal_url",
]
