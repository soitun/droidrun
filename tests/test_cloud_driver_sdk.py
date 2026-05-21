import asyncio
import re
import unittest
from importlib.metadata import version
from unittest.mock import patch

from mobilerun_sdk import AsyncMobilerun

from mobilerun.tools.driver.cloud import CloudDriver


class FakeState:
    def __init__(self):
        self.time_calls = []

    async def time(self, device_id, **kwargs):
        self.time_calls.append((device_id, kwargs))
        return "2026-05-21T12:34:56Z"


class FakeDevices:
    def __init__(self):
        self.state = FakeState()


class FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.devices = FakeDevices()


def _version_tuple(package_name: str) -> tuple[int, int, int]:
    package_version = version(package_name)
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", package_version)
    if match is None:
        raise AssertionError(f"Unexpected {package_name} version: {package_version}")
    return tuple(int(part) for part in match.groups())


class CloudDriverSdkTest(unittest.TestCase):
    def test_get_date_uses_restored_devices_state_time_sdk_path(self):
        with patch("mobilerun.tools.driver.cloud.AsyncMobilerun", FakeClient):
            driver = CloudDriver(
                "device-123",
                display_id=7,
                api_key="test-api-key",
            )

        result = asyncio.run(driver.get_date())

        self.assertEqual(result, "2026-05-21T12:34:56Z")
        state = driver._client.devices.state
        self.assertEqual(
            state.time_calls,
            [("device-123", {"x_device_display_id": 7})],
        )

    def test_installed_sdk_exposes_devices_state_time(self):
        self.assertGreaterEqual(_version_tuple("mobilerun-sdk"), (3, 2, 0))

        client = AsyncMobilerun(api_key="test-api-key")
        try:
            self.assertTrue(callable(client.devices.state.time))
        finally:
            asyncio.run(client.close())
