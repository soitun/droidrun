"""Screenshot-only state provider for visual control backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mobilerun.tools.helpers.images import image_dimensions
from mobilerun.tools.ui.provider import StateProvider
from mobilerun.tools.ui.state import UIState

if TYPE_CHECKING:
    from mobilerun.tools.driver.base import DeviceDriver


class ScreenshotOnlyStateProvider(StateProvider):
    """Build UI state from screenshots without reading an accessibility tree."""

    supported = {"convert_point", "direct_text_input"}
    requires_coordinate_tools = True

    def __init__(
        self,
        driver: "DeviceDriver",
        use_normalized: bool = True,
    ) -> None:
        super().__init__(driver)
        self.use_normalized = use_normalized

    async def get_state(self) -> UIState:
        screenshot = await self.driver.screenshot()
        screen_width, screen_height = image_dimensions(screenshot)
        coordinate_space = (
            "normalized 0-1000 coordinates"
            if self.use_normalized
            else "screenshot pixel coordinates"
        )

        return UIState(
            elements=[],
            formatted_text=(
                "Screenshot-only mode is active. There is no accessibility tree "
                "or element index list. Inspect the screenshot and use coordinate "
                f"actions in {coordinate_space}. For text entry, focus a field "
                "with a coordinate action first, then use direct text typing."
            ),
            focused_text="",
            phone_state={
                "observationMode": "screenshot_only",
                "accessibilityTree": False,
            },
            screen_width=screen_width,
            screen_height=screen_height,
            use_normalized=self.use_normalized,
        )
