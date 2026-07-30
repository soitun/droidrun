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


LOCAL_PORTAL_STATE = {
    "a11y_tree": {
        "boundsInScreen": {"left": 0, "top": 0, "right": 393, "bottom": 852},
        "className": "Application",
        "resourceId": "",
        "text": "",
        "contentDescription": "",
        "packageName": "com.apple.Preferences",
        "children": [
            {
                "boundsInScreen": {"left": 16, "top": 180, "right": 377, "bottom": 224},
                "className": "Button",
                "resourceId": "Privacy",
                "text": "Privacy & Security",
                "contentDescription": "",
                "packageName": "com.apple.Preferences",
                "children": [],
            }
        ],
    },
    "phone_state": {
        "currentApp": "Settings",
        "packageName": "com.apple.Preferences",
        "keyboardVisible": False,
    },
    "device_context": {
        "screen_bounds": {"width": 393, "height": 852},
        "filtering_params": {"min_element_size": 5, "overlay_offset": 0},
        "display_metrics": {"widthPixels": 393, "heightPixels": 852},
    },
}


class FakeLocalPortalDriver:
    """Duck-typed stand-in for IOSPortalHttpDriver in provider tests."""

    platform = "iOS"

    async def get_ui_tree(self):
        return LOCAL_PORTAL_STATE


def test_android_state_provider_consumes_local_portal_state():
    from mobilerun.tools.filters import DetailedFilter
    from mobilerun.tools.formatters import IndexedFormatter
    from mobilerun.tools.ui.provider import AndroidStateProvider

    provider = AndroidStateProvider(
        FakeLocalPortalDriver(),
        tree_filter=DetailedFilter(),
        tree_formatter=IndexedFormatter(),
    )
    state = run(provider.get_state())

    assert state.screen_width == 393
    assert state.screen_height == 852
    assert state.elements, "expected elements parsed from the portal a11y tree"
    assert "Privacy & Security" in state.formatted_text


def test_android_state_provider_vision_contract_uses_point_space():
    from mobilerun.tools.filters import ConciseFilter
    from mobilerun.tools.formatters import IndexedFormatter
    from mobilerun.tools.ui.provider import AndroidStateProvider

    provider = AndroidStateProvider(
        FakeLocalPortalDriver(),
        tree_filter=ConciseFilter(),
        tree_formatter=IndexedFormatter(),
        vision_enabled=True,
    )
    state = run(provider.get_state())

    # 393x852 points fit under the legacy 2048 max-side cap, so the model
    # space equals the point space and taps need no rescaling.
    assert state.coordinate_contract_active
    assert (state.model_screenshot_width, state.model_screenshot_height) == (393, 852)
    assert (state.coordinate_scale_x, state.coordinate_scale_y) == (1.0, 1.0)
