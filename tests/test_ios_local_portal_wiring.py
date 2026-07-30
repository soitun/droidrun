from __future__ import annotations

import asyncio

import httpx


def run(coro):
    return asyncio.run(coro)


def test_auth_token_defaults_to_none(monkeypatch):
    from mobilerun.config_manager.config_manager import DeviceConfig

    monkeypatch.delenv("MOBILERUN_DEVICE_TOKEN", raising=False)
    assert DeviceConfig().resolve_auth_token() is None


def test_auth_token_reads_config_value(monkeypatch):
    from mobilerun.config_manager.config_manager import DeviceConfig

    monkeypatch.delenv("MOBILERUN_DEVICE_TOKEN", raising=False)
    assert DeviceConfig(auth_token="from-config").resolve_auth_token() == "from-config"


def test_auth_token_env_overrides_config(monkeypatch):
    from mobilerun.config_manager.config_manager import DeviceConfig

    monkeypatch.setenv("MOBILERUN_DEVICE_TOKEN", "from-env")
    assert DeviceConfig(auth_token="from-config").resolve_auth_token() == "from-env"
