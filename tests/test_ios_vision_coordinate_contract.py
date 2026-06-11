"""Tests for the iOS a11y+vision coordinate contract.

iOS screenshots are physical pixels while taps and a11y bounds are points,
so the display space is derived from the actual screenshot bytes and
``convert_point`` maps the model's display-space output back to POINTS.
Mirrors tests/test_android_vision_coordinate_contract.py.
"""

import asyncio
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from mobilerun.agent.utils.actions import _convert_action_point
from mobilerun.tools.helpers.images import fit_dimensions_to_max_side
from mobilerun.tools.ui.ios_provider import IOSStateProvider
from mobilerun.tools.ui.provider import should_resize_model_screenshot

POINTS_W, POINTS_H = 440, 956
PIXELS_W, PIXELS_H = 1320, 2868  # 3x physical screenshot

RAW_A11Y = """Attributes: Application, pid: 123, label: 'Settings'
Element subtree:
 ->Application, pid: 123, label: 'Settings'
    Window (Main), {{0.0, 0.0}, {440.0, 956.0}}
      Cell, {{20.0, 200.0}, {400.0, 44.0}}, label: 'General'
      StaticText, {{20.0, 119.7}, {133.0, 40.7}}, label: 'Settings'
"""


def _png(width, height):
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class FakeIOSDriver:
    def __init__(self, screenshot_bytes=None, screenshot_error=None):
        self._shot = screenshot_bytes
        self._err = screenshot_error
        self.screenshot_calls = 0

    async def get_ui_tree(self):
        return {
            "phone_state": {"currentApp": "Settings", "packageName": "com.apple.Preferences"},
            "device_context": {"screen_bounds": {"width": POINTS_W, "height": POINTS_H}},
            "a11y_tree": RAW_A11Y,
        }

    async def screenshot(self):
        self.screenshot_calls += 1
        if self._err:
            raise self._err
        return self._shot


def _state(provider):
    return asyncio.run(provider.get_state())


def _general_element(state):
    return next(e for e in state.elements if e.get("text") == "General")


def test_vision_disabled_keeps_legacy_behavior():
    driver = FakeIOSDriver(screenshot_bytes=_png(8, 8))
    state = _state(IOSStateProvider(driver))

    assert state.coordinate_scale_x == 1.0
    assert state.coordinate_scale_y == 1.0
    assert "coordinate space" not in state.formatted_text
    assert "displayBounds" not in str(state.elements)
    assert state.convert_point(220, 478) == (220, 478)
    # no extra screenshot traffic in non-vision mode
    assert driver.screenshot_calls == 0


def test_vision_enabled_declares_display_space_and_scales_to_points():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)

    display_w, display_h = fit_dimensions_to_max_side(PIXELS_W, PIXELS_H)
    assert (display_w, display_h) == (943, 2048)
    assert state.coordinate_scale_x == POINTS_W / display_w
    assert state.coordinate_scale_y == POINTS_H / display_h

    # Contract declared in the display space
    assert f"{display_w}x{display_h} coordinate space" in state.formatted_text

    # Model-facing bounds in display space; tap bounds stay points
    element = _general_element(state)
    assert element["bounds"] == "20,200,420,244"
    scale_x, scale_y = state.coordinate_scale_x, state.coordinate_scale_y
    expected_display = ",".join(
        str(round(v / s))
        for v, s in zip((20, 200, 420, 244), (scale_x, scale_y, scale_x, scale_y))
    )
    assert element["displayBounds"] == expected_display
    assert expected_display in state.formatted_text
    assert "(20,200,420,244)" not in state.formatted_text

    # convert_point maps display-space output back to POINTS (hard anchors)
    assert state.convert_point(471, 1024) == (220, 478)
    # element tap path keeps points
    assert state.get_element_coords(element["index"]) == (220, 222)
    assert driver.screenshot_calls == 1


def test_normalized_mode_opts_out_of_the_contract():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, use_normalized=True, vision_enabled=True)
    state = _state(provider)

    assert provider.resize_model_screenshot is False
    assert state.coordinate_scale_x == 1.0
    assert "coordinate space" not in state.formatted_text
    assert driver.screenshot_calls == 0


def test_screenshot_failure_falls_back_to_legacy_behavior():
    driver = FakeIOSDriver(screenshot_error=RuntimeError("portal hiccup"))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)

    assert state.coordinate_scale_x == 1.0
    assert state.coordinate_scale_y == 1.0
    assert "coordinate space" not in state.formatted_text
    assert "displayBounds" not in str(state.elements)
    assert state.convert_point(220, 478) == (220, 478)


def test_resize_flag_pairs_with_agent_gating():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    assert should_resize_model_screenshot(IOSStateProvider(driver, vision_enabled=True))
    assert not should_resize_model_screenshot(IOSStateProvider(driver))
    assert not should_resize_model_screenshot(
        IOSStateProvider(driver, use_normalized=True, vision_enabled=True)
    )


def test_convert_action_point_validates_ios_model_space():
    driver = FakeIOSDriver(screenshot_bytes=_png(PIXELS_W, PIXELS_H))
    provider = IOSStateProvider(driver, vision_enabled=True)
    state = _state(provider)
    ctx = SimpleNamespace(ui=state, state_provider=provider)

    # display-space coordinates convert to points
    assert _convert_action_point(471, 1024, ctx=ctx) == (220, 478)

    # raw physical-pixel coordinates (the pre-contract failure mode) are
    # outside the 943x2048 display space and get rejected
    try:
        _convert_action_point(1200, 2500, ctx=ctx)
    except ValueError as e:
        assert "943x2048" in str(e)
    else:
        raise AssertionError("expected ValueError for image-space point")
