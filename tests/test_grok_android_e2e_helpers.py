from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tests.e2e import test_grok_android16 as android_e2e


class _BlackScreenshotDevice:
    capabilities = {"platform": "android"}

    def __init__(self, screenshot: bytes) -> None:
        self._screenshot = screenshot
        self.actions: list[tuple[Any, ...]] = []

    def key(self, name: str) -> None:
        self.actions.append(("key", name))

    def screen_size(self) -> tuple[int, int]:
        return 1080, 2400

    def swipe(self, *args: Any, **kwargs: Any) -> None:
        self.actions.append(("swipe", *args, kwargs))

    def wait_for_idle(self, *, timeout: float) -> bool:
        self.actions.append(("wait_for_idle", timeout))
        return True

    def current_app_id(self) -> str:
        return "com.android.launcher3"

    def ui(self) -> dict[str, object]:
        return {"phone_state": {"package_name": "com.android.launcher3"}}

    def screenshot(self, *, hide_overlay: bool) -> str:
        assert hide_overlay is True
        return base64.b64encode(self._screenshot).decode("ascii")


def _black_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 4), color="black").save(output, format="PNG")
    return output.getvalue()


def _context(tmp_path: Path) -> android_e2e._LiveContext:
    emulator = android_e2e._OwnedEmulator(
        binary=tmp_path / "emulator",
        artifact_root=tmp_path,
    )
    return android_e2e._LiveContext(
        serial="emulator-5558",
        repo_root=tmp_path,
        artifact_root=tmp_path,
        configured_secrets=(),
        emulator=emulator,
    )


def test_public_core_black_png_raises_graphics_readiness_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    device = _BlackScreenshotDevice(_black_png())
    monkeypatch.setattr(android_e2e, "_connect_core_device", lambda context: device)
    monkeypatch.setattr(android_e2e.time, "sleep", lambda seconds: None)

    with pytest.raises(
        android_e2e._AllBlackCoreScreenshot,
        match="all-black screenshot",
    ):
        android_e2e._capture_core_evidence(
            _context(tmp_path),
            tmp_path,
            label="pre-task",
            ensure_home=True,
        )

    assert (tmp_path / "pre-task-core.png").read_bytes() == device._screenshot
    assert device.actions[0] == ("key", "wakeup")
    assert ("key", "home") in device.actions


def test_snapshot_black_framebuffer_uses_no_wipe_cold_boot_and_rechecks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "scenario"
    artifact_dir.mkdir()
    context = _context(tmp_path)
    setup_dirs: list[Path] = []
    capture_dirs: list[Path] = []
    cold_boot_dirs: list[Path] = []
    preflight = android_e2e._DevicePreflight(
        android_version="16",
        sdk_level="36",
        portal_version_after_setup="0.7.1",
        portal_version_after_doctor="0.7.1",
    )
    recovered = android_e2e._CoreEvidence(
        current_app_id="com.android.launcher3",
        ui_path=artifact_dir / "recovered-ui.json",
        screenshot_path=artifact_dir / "recovered.png",
    )

    monkeypatch.setattr(
        android_e2e,
        "_restore_clean_snapshot",
        lambda emulator, serial, artifacts: "snapshot",
    )

    def setup_and_check(
        unused_context: android_e2e._LiveContext,
        artifacts: Path,
    ) -> android_e2e._DevicePreflight:
        setup_dirs.append(artifacts)
        return preflight

    monkeypatch.setattr(android_e2e, "_setup_and_check_device", setup_and_check)

    def capture(
        unused_context: android_e2e._LiveContext,
        artifacts: Path,
        *,
        label: str,
        ensure_home: bool,
    ) -> android_e2e._CoreEvidence:
        assert label == "pre-task"
        assert ensure_home is True
        capture_dirs.append(artifacts)
        if len(capture_dirs) == 1:
            raise android_e2e._AllBlackCoreScreenshot("all-black public frame")
        return recovered

    monkeypatch.setattr(android_e2e, "_capture_core_evidence", capture)
    monkeypatch.setattr(
        android_e2e,
        "_cold_boot_owned_emulator",
        lambda emulator, artifacts: cold_boot_dirs.append(artifacts),
    )
    monkeypatch.setattr(android_e2e, "_adb", lambda serial, *args: "30000")
    monkeypatch.setattr(
        android_e2e,
        "_timeout_labels",
        lambda timeout_ms: ("30 seconds",),
    )

    result = android_e2e._prepare_scenario_device(
        context,
        android_e2e._API_DIRECT,
        artifact_dir,
    )

    fallback_dir = artifact_dir / "snapshot-graphics-cold-boot"
    assert setup_dirs == [artifact_dir, fallback_dir]
    assert capture_dirs == [artifact_dir, fallback_dir]
    assert cold_boot_dirs == [fallback_dir]
    assert result.pre_evidence is recovered
    fallback = json.loads(
        (artifact_dir / "snapshot-graphics-fallback.json").read_text()
    )
    assert fallback["failure"] == "all_black_public_core_framebuffer"
    assert fallback["wipe_data"] is False
    summary = json.loads((artifact_dir / "preflight.json").read_text())
    assert summary["snapshot_attempted"] is True
    assert summary["snapshot_restored"] is True
    assert summary["reset_mode"] == "snapshot_graphics_cold_boot_fallback"
    assert summary["cold_boot_without_wipe_data"] is True
