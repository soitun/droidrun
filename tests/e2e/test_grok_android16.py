"""Opt-in Grok API/OAuth E2E coverage on the dedicated Android 16 emulator.

These tests exercise the natural-language agent through public ``mobilerun
run`` and use the mobile harness's public ``mobilerun_core.Mobilerun`` surface
for harness-owned device actions and evidence. They are excluded from normal
test runs because they consume live xAI credentials and mutate the emulator's
foreground UI. To run them, set both of the following explicitly::

    MOBILERUN_RUN_ANDROID_E2E=1
    ANDROID_SERIAL=emulator-5558

The host must provide AVD ``mobilerun_agent_bench_api36`` with snapshot
``mobilerun_eval_clean_api36``. The session claims only port 5558, launches and
owns that exact emulator, and tears down only ``emulator-5558``. Every scenario
restores the snapshot, runs ``mobilerun setup``, and verifies the Portal before
starting the task. Agent auto-setup remains disabled during the task itself.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import pytest

_EXPECTED_SERIAL = "emulator-5558"
_EMULATOR_PORT = 5558
_AVD_NAME = "mobilerun_agent_bench_api36"
_SNAPSHOT_NAME = "mobilerun_eval_clean_api36"
_BOOT_TIMEOUT_SECONDS = 240.0
_RUN_LIVE_E2E = os.environ.get("MOBILERUN_RUN_ANDROID_E2E") == "1"

if _RUN_LIVE_E2E:
    try:
        from mobilerun_core import Mobilerun
    except ModuleNotFoundError as exc:
        raise pytest.UsageError(
            "The opted-in Grok Android E2E suite requires the public "
            "mobilerun-core package in this Python environment. Install "
            f"`{sys.executable} -m pip install \"mobilerun-core[local]\"` "
            "and retry."
        ) from exc
else:
    Mobilerun = None

pytestmark = pytest.mark.skipif(
    not _RUN_LIVE_E2E,
    reason="set MOBILERUN_RUN_ANDROID_E2E=1 to run live Grok Android E2E tests",
)

_TIMEOUT_TASK = (
    "Open Android Settings, find Screen timeout, report the currently visible "
    "timeout value, then return to the Home screen. Do not change any setting."
)
_ANDROID_VERSION_TASK = (
    "Open Android Settings, go to About phone, and report the visible Android "
    "version, then return to the Home screen. Do not change any setting."
)


@dataclass(frozen=True)
class _Scenario:
    scenario_id: str
    provider: str
    auth_mode: str
    task: str
    mode_flags: tuple[str, ...]
    expected_state: str


@dataclass
class _OwnedEmulator:
    binary: Path
    artifact_root: Path
    process: subprocess.Popen[str] | None = None
    log_handle: TextIO | None = None
    launch_mode: str = ""


@dataclass(frozen=True)
class _LiveContext:
    serial: str
    repo_root: Path
    artifact_root: Path
    configured_secrets: tuple[str, ...]
    emulator: _OwnedEmulator


@dataclass(frozen=True)
class _ScenarioDeviceState:
    home_package: str
    android_version: str
    timeout_labels: tuple[str, ...]
    pre_evidence: _CoreEvidence


@dataclass(frozen=True)
class _CoreEvidence:
    current_app_id: str
    ui_path: Path
    screenshot_path: Path


@dataclass(frozen=True)
class _DevicePreflight:
    android_version: str
    sdk_level: str
    portal_version_after_setup: str
    portal_version_after_doctor: str


@dataclass(frozen=True)
class _FileFingerprint:
    exists: bool
    sha256: str | None
    mtime_ns: int | None


@dataclass(frozen=True)
class _ScenarioResult:
    output: str
    artifact_dir: Path
    trajectory_dir: Path
    events: tuple[dict[str, object], ...]
    device_state: _ScenarioDeviceState
    post_evidence: _CoreEvidence


class _AllBlackCoreScreenshot(AssertionError):
    """The public Mobilerun observation returned a valid but black PNG."""


_API_DIRECT = _Scenario(
    scenario_id="01-xai-api-direct-ui-tree",
    provider="XAI",
    auth_mode="api_key",
    task=_TIMEOUT_TASK,
    mode_flags=("--no-reasoning", "--no-vision", "--no-vision-only"),
    expected_state="screen_timeout",
)
_API_REASONING_VISION = _Scenario(
    scenario_id="02-xai-api-reasoning-vision-only",
    provider="XAI",
    auth_mode="api_key",
    task=_ANDROID_VERSION_TASK,
    mode_flags=("--reasoning", "--vision", "--vision-only"),
    expected_state="android_version",
)
_OAUTH_DIRECT = _Scenario(
    scenario_id="03-grok-oauth-direct-ui-tree",
    provider="grok_oauth",
    auth_mode="oauth",
    task=_TIMEOUT_TASK,
    mode_flags=("--no-reasoning", "--no-vision", "--no-vision-only"),
    expected_state="screen_timeout",
)
_OAUTH_REASONING_VISION = _Scenario(
    scenario_id="04-grok-oauth-reasoning-vision-only",
    provider="grok_oauth",
    auth_mode="oauth",
    task=_ANDROID_VERSION_TASK,
    mode_flags=("--reasoning", "--vision", "--vision-only"),
    expected_state="android_version",
)


def _adb(serial: str, *args: str, timeout: float = 30.0) -> str:
    completed = subprocess.run(
        ["adb", "-s", serial, *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"adb {' '.join(args)} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _adb_readiness_probe(serial: str, *args: str) -> tuple[int, str]:
    """Best-effort ADB probe while the restored emulator is still stabilizing."""

    try:
        completed = subprocess.run(
            ["adb", "-s", serial, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return -1, ""
    return completed.returncode, completed.stdout.strip()


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _resolve_emulator_binary() -> Path:
    candidates: list[Path] = []
    for env_name in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        sdk_root = os.environ.get(env_name)
        if sdk_root:
            candidates.append(Path(sdk_root) / "emulator" / "emulator")
    candidates.append(
        Path.home() / "Library" / "Android" / "sdk" / "emulator" / "emulator"
    )
    discovered = shutil.which("emulator")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    pytest.fail(
        "Android emulator binary not found; set ANDROID_SDK_ROOT or ANDROID_HOME"
    )


def _emulator_command(binary: Path, *, load_snapshot: bool) -> list[str]:
    command = [
        str(binary),
        "-avd",
        _AVD_NAME,
        "-port",
        str(_EMULATOR_PORT),
    ]
    if load_snapshot:
        command.extend(("-snapshot", _SNAPSHOT_NAME))
    else:
        # The recovery path intentionally preserves AVD data.
        command.append("-no-snapshot-load")
    command.extend(("-no-snapshot-save", "-no-boot-anim", "-no-audio"))
    assert "-wipe-data" not in command
    return command


def _wait_for_boot(
    serial: str,
    artifact_dir: Path,
    *,
    process: subprocess.Popen[str] | None,
    boot_label: str,
) -> None:
    """Wait for the owned emulator and Android package-manager readiness."""

    started_at = time.monotonic()
    last_boot_completed = ""
    last_boot_animation = ""
    package_manager_ready = False
    deadline = started_at + _BOOT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"owned emulator exited during {boot_label} boot "
                f"with status {process.returncode}"
            )
        boot_returncode, last_boot_completed = _adb_readiness_probe(
            serial, "shell", "getprop", "sys.boot_completed"
        )
        _, last_boot_animation = _adb_readiness_probe(
            serial, "shell", "getprop", "init.svc.bootanim"
        )
        package_returncode, package_output = _adb_readiness_probe(
            serial,
            "shell",
            "cmd",
            "package",
            "list",
            "packages",
            "android",
        )
        package_manager_ready = (
            package_returncode == 0 and "package:android" in package_output
        )
        if (
            boot_returncode == 0
            and last_boot_completed == "1"
            # With the required ``-no-boot-anim`` launch flag, Android 16 may
            # leave this service property empty instead of reporting stopped.
            and last_boot_animation in {"", "stopped"}
            and package_manager_ready
        ):
            (artifact_dir / f"boot-readiness-{boot_label}.json").write_text(
                json.dumps(
                    {
                        "avd": _AVD_NAME,
                        "serial": serial,
                        "port": _EMULATOR_PORT,
                        "boot_label": boot_label,
                        "sys.boot_completed": last_boot_completed,
                        "init.svc.bootanim": last_boot_animation,
                        "package_manager_ready": package_manager_ready,
                        "elapsed_seconds": round(time.monotonic() - started_at, 3),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return
        time.sleep(2.0)

    raise RuntimeError(
        "the restored emulator did not finish booting "
        f"(sys.boot_completed={last_boot_completed!r}, "
        f"bootanim={last_boot_animation!r}, "
        f"package_manager_ready={package_manager_ready})"
    )


def _owned_avd_name(serial: str) -> str:
    return _adb(serial, "emu", "avd", "name").splitlines()[0].strip()


def _launch_owned_emulator(
    emulator: _OwnedEmulator,
    *,
    load_snapshot: bool,
    evidence_dir: Path,
) -> None:
    mode = "snapshot" if load_snapshot else "cold-boot"
    command = _emulator_command(emulator.binary, load_snapshot=load_snapshot)
    (evidence_dir / f"emulator-launch-{mode}.json").write_text(
        json.dumps(
            {
                "command": command,
                "avd": _AVD_NAME,
                "serial": _EXPECTED_SERIAL,
                "port": _EMULATOR_PORT,
                "snapshot": _SNAPSHOT_NAME if load_snapshot else None,
                "wipe_data": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    emulator.log_handle = (emulator.artifact_root / "emulator.log").open(
        "a", encoding="utf-8"
    )
    emulator.process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=emulator.log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    emulator.launch_mode = mode


def _stop_owned_emulator(emulator: _OwnedEmulator) -> None:
    """Stop only the process this session launched on emulator-5558."""

    process = emulator.process
    if process is not None and process.poll() is None:
        avd_probe = _adb_readiness_probe(
            _EXPECTED_SERIAL, "emu", "avd", "name"
        )[1]
        if avd_probe.splitlines()[:1] == [_AVD_NAME]:
            try:
                subprocess.run(
                    ["adb", "-s", _EXPECTED_SERIAL, "emu", "kill"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
            except subprocess.TimeoutExpired:
                pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    if emulator.log_handle is not None:
        emulator.log_handle.close()
    emulator.process = None
    emulator.log_handle = None


def _wait_for_port_release(timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _port_is_open(_EMULATOR_PORT):
            return
        time.sleep(0.5)
    raise RuntimeError(f"TCP port {_EMULATOR_PORT} did not become free")


def _cold_boot_owned_emulator(
    emulator: _OwnedEmulator, evidence_dir: Path
) -> None:
    _stop_owned_emulator(emulator)
    _wait_for_port_release()
    _launch_owned_emulator(
        emulator, load_snapshot=False, evidence_dir=evidence_dir
    )
    _wait_for_boot(
        _EXPECTED_SERIAL,
        evidence_dir,
        process=emulator.process,
        boot_label="cold-boot",
    )
    if _owned_avd_name(_EXPECTED_SERIAL) != _AVD_NAME:
        raise RuntimeError("cold-booted emulator reported the wrong AVD name")


def _start_session_emulator(artifact_root: Path) -> _OwnedEmulator:
    if _port_is_open(_EMULATOR_PORT):
        pytest.fail(
            f"TCP port {_EMULATOR_PORT} is already occupied; the E2E suite "
            "will not reuse or terminate an emulator it does not own"
        )
    binary = _resolve_emulator_binary()
    listed = subprocess.run(
        [str(binary), "-list-avds"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if listed.returncode != 0 or _AVD_NAME not in listed.stdout.splitlines():
        pytest.fail(f"required Android AVD {_AVD_NAME!r} is not installed")

    evidence_dir = artifact_root / "emulator-session"
    evidence_dir.mkdir()
    emulator = _OwnedEmulator(binary=binary, artifact_root=artifact_root)
    _launch_owned_emulator(
        emulator, load_snapshot=True, evidence_dir=evidence_dir
    )
    try:
        _wait_for_boot(
            _EXPECTED_SERIAL,
            evidence_dir,
            process=emulator.process,
            boot_label="snapshot-launch",
        )
        if _owned_avd_name(_EXPECTED_SERIAL) != _AVD_NAME:
            raise RuntimeError("snapshot-launched emulator reported the wrong AVD name")
    except RuntimeError as snapshot_error:
        (evidence_dir / "snapshot-launch-fallback.json").write_text(
            json.dumps(
                {
                    "fallback": "cold-boot",
                    "reason": str(snapshot_error),
                    "wipe_data": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            _cold_boot_owned_emulator(emulator, evidence_dir)
        except RuntimeError as cold_error:
            _stop_owned_emulator(emulator)
            pytest.fail(f"could not boot required AVD {_AVD_NAME!r}: {cold_error}")
    return emulator


def _restore_clean_snapshot(
    emulator: _OwnedEmulator, serial: str, artifact_dir: Path
) -> str:
    """Reset a scenario, cold-booting without wipe-data if snapshot load fails."""

    try:
        completed = subprocess.run(
            [
                "adb",
                "-s",
                serial,
                "emu",
                "avd",
                "snapshot",
                "load",
                _SNAPSHOT_NAME,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        completed = None

    response = (
        (completed.stdout + "\n" + completed.stderr).strip()
        if completed is not None
        else "snapshot load timed out"
    )
    (artifact_dir / "snapshot-restore.txt").write_text(
        response + "\n", encoding="utf-8"
    )
    snapshot_confirmed = (
        completed is not None
        and completed.returncode == 0
        and not re.search(r"(?im)^KO\b", response)
        and bool(re.search(r"(?im)^OK\b", response))
    )
    if snapshot_confirmed:
        try:
            _wait_for_boot(
                serial,
                artifact_dir,
                process=emulator.process,
                boot_label="snapshot-restore",
            )
            return "snapshot"
        except RuntimeError as snapshot_error:
            response = f"{response}\nreadiness failure: {snapshot_error}"

    (artifact_dir / "snapshot-load-fallback.json").write_text(
        json.dumps(
            {
                "fallback": "cold-boot",
                "reason": response,
                "wipe_data": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        _cold_boot_owned_emulator(emulator, artifact_dir)
    except RuntimeError as cold_error:
        pytest.fail(
            "snapshot load and no-wipe-data cold-boot fallback both failed: "
            f"{cold_error}"
        )
    return "cold_boot_fallback"


def _focused_package(serial: str) -> str:
    window_dump = _adb(serial, "shell", "dumpsys", "window", "windows")
    activity_dump = _adb(serial, "shell", "dumpsys", "activity", "activities")
    ui_dump = _adb(serial, "exec-out", "uiautomator", "dump", "/dev/tty")
    for output, patterns in (
        (
            window_dump,
            (
                r"mCurrentFocus=Window\{[^\n]*\s([A-Za-z0-9_.]+)/",
                r"mFocusedApp=.*\s([A-Za-z0-9_.]+)/",
            ),
        ),
        (
            activity_dump,
            (
                r"mResumedActivity:[^\n]*\s([A-Za-z0-9_.]+)/",
                r"topResumedActivity=[^\n]*\s([A-Za-z0-9_.]+)/",
            ),
        ),
        (ui_dump, (r'package="([A-Za-z0-9_.]+)"',)),
    ):
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1)
    pytest.fail("could not determine the emulator's focused Android package")


def _timeout_labels(milliseconds: str) -> tuple[str, ...]:
    try:
        value = int(milliseconds.strip())
    except ValueError:
        pytest.fail("screen_off_timeout was not an integer")

    if value <= 0 or value >= 2_000_000_000:
        return ("never",)

    seconds = value // 1000
    if seconds == 1:
        return ("1 second", "1 sec")
    if seconds < 60:
        return (f"{seconds} seconds", f"{seconds} sec")

    minutes = seconds // 60
    if seconds % 60 == 0:
        if minutes == 1:
            return ("1 minute", "1 min")
        return (f"{minutes} minutes", f"{minutes} min")
    return (f"{seconds} seconds", f"{seconds} sec")


def _credential_values(value: object, *, key: str = "") -> set[str]:
    """Extract only secret-shaped values; never include metadata in failures."""

    found: set[str] = set()
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            found.update(_credential_values(nested_value, key=str(nested_key)))
    elif isinstance(value, list):
        for nested_value in value:
            found.update(_credential_values(nested_value, key=key))
    elif isinstance(value, str) and len(value) >= 8:
        normalized_key = key.casefold().replace("_", "").replace("-", "")
        if "token" in normalized_key or normalized_key in {
            "apikey",
            "secret",
            "password",
        }:
            found.add(value)
    return found


def _load_grok_oauth_secrets() -> set[str]:
    from mobilerun.config_manager.credential_paths import GROK_OAUTH_CREDENTIAL_PATH

    credential_path = Path(GROK_OAUTH_CREDENTIAL_PATH)
    if not credential_path.is_file():
        pytest.fail(
            "Grok OAuth credentials are required; run `mobilerun configure grok` "
            "before the opted-in E2E suite"
        )
    try:
        profiles = json.loads(credential_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pytest.fail("MobileRun's OAuth credential file is missing or malformed")

    grok_profile = profiles.get("grokOauth") if isinstance(profiles, dict) else None
    if not isinstance(grok_profile, dict) or not (
        grok_profile.get("accessToken") or grok_profile.get("refreshToken")
    ):
        pytest.fail(
            "the grokOauth slot is missing; run `mobilerun configure grok` "
            "before the opted-in E2E suite"
        )
    return _credential_values(grok_profile)


def _file_fingerprint(path: Path) -> _FileFingerprint:
    """Capture only existence, SHA-256, and mtime; never parse credential data."""

    try:
        before = path.stat()
    except FileNotFoundError:
        return _FileFingerprint(exists=False, sha256=None, mtime_ns=None)
    except OSError:
        pytest.fail("could not stat the external Grok CLI credential file")

    digest = hashlib.sha256()
    try:
        with path.open("rb") as credential_file:
            for chunk in iter(lambda: credential_file.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError:
        pytest.fail("could not hash the external Grok CLI credential file")
    if before.st_mtime_ns != after.st_mtime_ns:
        pytest.fail("the external Grok CLI credential changed while being hashed")
    return _FileFingerprint(
        exists=True,
        sha256=digest.hexdigest(),
        mtime_ns=after.st_mtime_ns,
    )


def _write_e2e_config(
    path: Path, serial: str, trajectory_path: Path
) -> None:
    import yaml

    from mobilerun.config_manager import MobileConfig
    from mobilerun.config_manager.migrations import CURRENT_VERSION

    config = MobileConfig()
    config.agent.max_steps = 30
    config.agent.streaming = False
    config.agent.app_cards.enabled = False
    config.device.serial = serial
    config.device.platform = "android"
    config.device.use_tcp = True
    config.device.portal_mode = "required"
    config.device.auto_setup = False
    config.telemetry.enabled = False
    config.tracing.enabled = False
    config.logging.debug = False
    config.logging.rich_text = False
    config.logging.save_trajectory = "action"
    config.logging.trajectory_gifs = False
    config.logging.trajectory_path = str(trajectory_path)
    config.mcp.enabled = False

    payload = config.to_dict()
    payload["_version"] = CURRENT_VERSION
    path.write_text(
        yaml.safe_dump(payload, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def _portal_version(serial: str) -> str:
    output = _adb(
        serial,
        "shell",
        "content",
        "query",
        "--uri",
        "content://com.mobilerun.portal/version",
    )
    match = re.search(r"\bresult=(\{.*\})\s*$", output)
    if not match:
        pytest.fail("Portal content provider did not return version evidence")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        pytest.fail("Portal content provider returned malformed version evidence")
    if payload.get("status") != "success":
        pytest.fail("Portal content provider version query was not successful")
    version = payload.get("result") or payload.get("data")
    if not isinstance(version, str) or not version.strip():
        pytest.fail("Portal content provider returned an empty version")
    return version.strip()


def _run_public_cli(
    repo_root: Path,
    serial: str,
    *,
    args: tuple[str, ...],
    secrets: tuple[str, ...],
    output_path: Path,
    timeout: float,
) -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "ANDROID_SERIAL": serial,
            "MOBILERUN_TELEMETRY_ENABLED": "false",
            "DROIDRUN_TELEMETRY_ENABLED": "false",
        }
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "mobilerun", *args],
            cwd=repo_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"`mobilerun {args[0]}` exceeded its E2E timeout")
    output = _ANSI_ESCAPE.sub("", completed.stdout + "\n" + completed.stderr)
    _assert_credentials_redacted(output, secrets)
    output_path.write_text(output, encoding="utf-8")
    assert completed.returncode == 0, _sanitized_tail(output, secrets)
    return output


def _setup_portal(
    repo_root: Path,
    serial: str,
    *,
    secrets: tuple[str, ...],
    artifact_dir: Path,
) -> None:
    last_output = ""
    for attempt in range(1, 4):
        output_name = (
            "mobilerun-setup.txt"
            if attempt == 1
            else f"mobilerun-setup-retry-{attempt - 1}.txt"
        )
        last_output = _run_public_cli(
            repo_root,
            serial,
            args=("setup", "--device", serial),
            secrets=secrets,
            output_path=artifact_dir / output_name,
            timeout=5 * 60,
        )
        assert "setup complete!" in last_output.casefold(), (
            "`mobilerun setup` did not report successful Portal setup: "
            + _sanitized_tail(last_output, secrets)
        )
        assert "setup failed" not in last_output.casefold(), (
            "`mobilerun setup` reported failure: "
            + _sanitized_tail(last_output, secrets)
        )
        if (
            "did not become responsive" not in last_output.casefold()
            and _portal_accessibility_enabled(serial)
        ):
            return
        if attempt < 3:
            time.sleep(5.0)
    pytest.fail(
        "Portal did not become responsive with accessibility enabled after "
        "three public `mobilerun setup` attempts: "
        + _sanitized_tail(last_output, secrets)
    )


def _ping_portal(
    repo_root: Path,
    serial: str,
    *,
    use_tcp: bool,
    secrets: tuple[str, ...],
    artifact_dir: Path,
) -> None:
    mode_flag = "--tcp" if use_tcp else "--no-tcp"
    mode_name = "tcp" if use_tcp else "content"
    output = _run_public_cli(
        repo_root,
        serial,
        args=("ping", "--device", serial, mode_flag, "--no-debug"),
        secrets=secrets,
        output_path=artifact_dir / f"mobilerun-ping-{mode_name}.txt",
        timeout=60,
    )
    mode = "TCP" if use_tcp else "content-provider"
    assert "portal is installed and accessible" in output.casefold(), (
        f"{mode} Portal ping did not report success: "
        + _sanitized_tail(output, secrets)
    )


def _doctor_portal(
    repo_root: Path,
    serial: str,
    *,
    secrets: tuple[str, ...],
    artifact_dir: Path,
) -> None:
    output = _run_public_cli(
        repo_root,
        serial,
        args=("doctor", "--device", serial, "--no-debug"),
        secrets=secrets,
        output_path=artifact_dir / "mobilerun-doctor.txt",
        timeout=5 * 60,
    )
    assert "mobilerun doctor" in output.casefold(), (
        "`mobilerun doctor` did not start correctly"
    )
    assert re.search(r"(?mi)^\s*Portal Version\s{2,}.*$", output), (
        "`mobilerun doctor` did not report its Portal Version check"
    )
    required_checks = (
        "Device",
        "Portal",
        "Accessibility",
        "Content Provider",
        "State (content)",
        "Screenshot (content)",
        "TCP Mode",
        "State (tcp)",
        "Screenshot (tcp)",
    )
    # Rich wraps long doctor rows according to the terminal width (for
    # example, State (content) can continue on the next line). Bound each
    # section by the following row so a checkmark from a later row cannot make
    # an earlier failed row pass accidentally.
    row_boundaries = (*required_checks, "Keyboard")
    for index, check_name in enumerate(required_checks):
        next_name = row_boundaries[index + 1]
        row = re.search(
            rf"(?ms)^\s*{re.escape(check_name)}\s{{2,}}.*?"
            rf"(?=^\s*{re.escape(next_name)}\s{{2,}})",
            output,
        )
        assert row and "✓" in row.group(0), (
            f"`mobilerun doctor` did not pass its {check_name} check"
        )
    assert not re.search(r"(?mi)^\s*\d+ issue\(s\):", output), (
        "`mobilerun doctor` reported one or more failing checks: "
        + _sanitized_tail(output, secrets)
    )


@pytest.fixture(scope="session")
def live_context() -> _LiveContext:
    serial = os.environ.get("ANDROID_SERIAL")
    if serial != _EXPECTED_SERIAL:
        pytest.fail(
            "the opted-in Grok E2E suite requires "
            f"ANDROID_SERIAL={_EXPECTED_SERIAL} exactly"
        )
    if shutil.which("adb") is None:
        pytest.fail("adb is required for the opted-in Android E2E suite")

    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        pytest.fail("XAI_API_KEY is required for the opted-in Grok API E2E cases")
    configured_secrets = _load_grok_oauth_secrets()
    configured_secrets.add(api_key)
    secrets = tuple(configured_secrets)

    repo_root = Path(__file__).resolve().parents[2]
    configured_artifact_root = os.environ.get("MOBILERUN_GROK_E2E_ARTIFACTS")
    if configured_artifact_root:
        artifact_base = Path(configured_artifact_root).expanduser().resolve()
        artifact_base.mkdir(parents=True, exist_ok=True)
        artifact_root = artifact_base / (
            f"run-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.time_ns()}"
        )
        artifact_root.mkdir()
    else:
        artifact_root = Path(
            tempfile.mkdtemp(prefix="mobilerun-grok-android16-")
        ).resolve()

    external_grok_auth = Path.home() / ".grok" / "auth.json"
    external_grok_before = _file_fingerprint(external_grok_auth)
    emulator = _start_session_emulator(artifact_root)
    context = _LiveContext(
        serial=serial,
        repo_root=repo_root,
        artifact_root=artifact_root,
        configured_secrets=secrets,
        emulator=emulator,
    )
    try:
        yield context
    finally:
        _stop_owned_emulator(emulator)
        external_grok_after = _file_fingerprint(external_grok_auth)
        hash_unchanged = external_grok_before.sha256 == external_grok_after.sha256
        mtime_unchanged = (
            external_grok_before.mtime_ns == external_grok_after.mtime_ns
        )
        existence_unchanged = (
            external_grok_before.exists == external_grok_after.exists
        )
        (artifact_root / "external-grok-cli-auth-invariant.json").write_text(
            json.dumps(
                {
                    "path": "~/.grok/auth.json",
                    "existence_unchanged": existence_unchanged,
                    "sha256_unchanged": hash_unchanged,
                    "mtime_unchanged": mtime_unchanged,
                    "contents_parsed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not (existence_unchanged and hash_unchanged and mtime_unchanged):
            pytest.fail(
                "MobileRun's live Grok suite modified the separate Grok CLI "
                "credential at ~/.grok/auth.json"
            )


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TOKEN_ASSIGNMENT = re.compile(
    r"(?i)(?:access[_-]?token|refresh[_-]?token|authorization)\s*[:=]\s*"
    r"[\"']?(?:bearer\s+)?[A-Za-z0-9._~+/-]{12,}"
)


def _assert_credentials_redacted(output: str, secrets: tuple[str, ...]) -> None:
    for secret in secrets:
        assert secret not in output, "Mobilerun output exposed a configured secret"
    assert not _TOKEN_ASSIGNMENT.search(output), (
        "Mobilerun output exposed a token-shaped credential assignment"
    )


def _assert_artifacts_redacted(root: Path, secrets: tuple[str, ...]) -> None:
    """Scan every artifact as bytes and every UTF-8 artifact as text."""

    for artifact in sorted(path for path in root.rglob("*") if path.is_file()):
        data = artifact.read_bytes()
        relative_path = artifact.relative_to(root)
        for secret in secrets:
            assert secret.encode("utf-8") not in data, (
                f"artifact {relative_path} exposed a configured secret"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        assert not _TOKEN_ASSIGNMENT.search(text), (
            f"text artifact {relative_path} exposed a token-shaped credential"
        )


def _sanitized_tail(output: str, secrets: tuple[str, ...]) -> str:
    sanitized = output
    for secret in secrets:
        sanitized = sanitized.replace(secret, "[REDACTED]")
    sanitized = _TOKEN_ASSIGNMENT.sub("credential=[REDACTED]", sanitized)
    return sanitized[-4000:]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _connect_core_device(context: _LiveContext) -> Any:
    if Mobilerun is None:
        pytest.fail(
            "mobilerun_core.Mobilerun is unavailable in the live E2E environment"
        )
    try:
        device = Mobilerun().connect(
            context.serial,
            backend="local-android-adb",
            portal_mode="required",
        )
    except Exception as exc:
        pytest.fail(f"public mobilerun-core device connection failed: {exc}")
    assert device.capabilities.get("platform") == "android"
    return device


def _capture_core_evidence(
    context: _LiveContext,
    artifact_dir: Path,
    *,
    label: str,
    ensure_home: bool,
) -> _CoreEvidence:
    """Use only the public mobile-harness surface for control and evidence."""

    device = _connect_core_device(context)
    if ensure_home:
        # Snapshot restores may preserve an asleep display. Use only the
        # public mobile-harness key/swipe surface for the ordinary wake,
        # unlock, and Home sequence before observing the framebuffer.
        device.key("wakeup")
        time.sleep(0.5)
        width, height = device.screen_size()
        device.swipe(width // 2, height * 4 // 5, width // 2, height // 5, ms=350)
        time.sleep(0.5)
        device.key("home")
        device.wait_for_idle(timeout=5.0)
    current_app_id = device.current_app_id() or ""
    ui = device.ui()
    screenshot = base64.b64decode(device.screenshot(hide_overlay=True))
    ui_path = artifact_dir / f"{label}-core-ui.json"
    screenshot_path = artifact_dir / f"{label}-core.png"
    ui_path.write_text(
        json.dumps(_jsonable(ui), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    screenshot_path.write_bytes(screenshot)
    assert screenshot.startswith(b"\x89PNG\r\n\x1a\n"), (
        "mobilerun-core evidence screenshot was not a PNG"
    )
    from PIL import Image

    with Image.open(io.BytesIO(screenshot)) as image:
        if image.convert("RGB").getbbox() is None:
            raise _AllBlackCoreScreenshot(
                "public mobilerun-core returned an all-black screenshot after "
                "wake, unlock, and Home"
            )
    return _CoreEvidence(
        current_app_id=current_app_id,
        ui_path=ui_path,
        screenshot_path=screenshot_path,
    )


def _portal_accessibility_enabled(serial: str) -> bool:
    enabled_services = _adb(
        serial,
        "shell",
        "settings",
        "get",
        "secure",
        "enabled_accessibility_services",
    )
    return "com.mobilerun.portal" in enabled_services


def _setup_and_check_device(
    context: _LiveContext,
    artifact_dir: Path,
) -> _DevicePreflight:
    """Run public setup/ping/doctor and independent readiness assertions."""

    _setup_portal(
        context.repo_root,
        context.serial,
        secrets=context.configured_secrets,
        artifact_dir=artifact_dir,
    )

    android_version = _adb(
        context.serial, "shell", "getprop", "ro.build.version.release"
    )
    assert android_version == "16", (
        f"{context.serial} must run Android 16 during scenario preflight "
        f"(reported {android_version!r})"
    )
    sdk_level = _adb(
        context.serial, "shell", "getprop", "ro.build.version.sdk"
    )
    assert sdk_level == "36", (
        f"{context.serial} must use Android SDK 36 during scenario preflight "
        f"(reported {sdk_level!r})"
    )

    portal_package = _adb(
        context.serial, "shell", "pm", "path", "com.mobilerun.portal"
    )
    assert portal_package.startswith("package:"), (
        "`mobilerun setup` did not install Mobilerun Portal"
    )
    portal_version_after_setup = _portal_version(context.serial)
    assert re.fullmatch(
        r"v?\d+(?:\.\d+){1,3}(?:[-+][A-Za-z0-9.-]+)?",
        portal_version_after_setup,
    ), "Portal content provider returned an invalid version string"
    assert _portal_accessibility_enabled(context.serial), (
        "`mobilerun setup` did not enable Portal accessibility"
    )

    _ping_portal(
        context.repo_root,
        context.serial,
        use_tcp=False,
        secrets=context.configured_secrets,
        artifact_dir=artifact_dir,
    )
    _ping_portal(
        context.repo_root,
        context.serial,
        use_tcp=True,
        secrets=context.configured_secrets,
        artifact_dir=artifact_dir,
    )
    _doctor_portal(
        context.repo_root,
        context.serial,
        secrets=context.configured_secrets,
        artifact_dir=artifact_dir,
    )

    portal_version_after_doctor = _portal_version(context.serial)
    assert re.fullmatch(
        r"v?\d+(?:\.\d+){1,3}(?:[-+][A-Za-z0-9.-]+)?",
        portal_version_after_doctor,
    ), "Portal content provider returned an invalid post-doctor version string"
    assert _portal_accessibility_enabled(context.serial), (
        "Portal accessibility was not enabled after `mobilerun doctor`"
    )
    return _DevicePreflight(
        android_version=android_version,
        sdk_level=sdk_level,
        portal_version_after_setup=portal_version_after_setup,
        portal_version_after_doctor=portal_version_after_doctor,
    )


def _prepare_scenario_device(
    context: _LiveContext,
    scenario: _Scenario,
    artifact_dir: Path,
) -> _ScenarioDeviceState:
    """Restore, boot, set up, and fully validate the device for one scenario."""

    initial_reset_mode = _restore_clean_snapshot(
        context.emulator, context.serial, artifact_dir
    )
    final_reset_mode = initial_reset_mode
    graphics_fallback_reason: str | None = None
    preflight_artifact_dir = artifact_dir
    preflight = _setup_and_check_device(context, preflight_artifact_dir)
    try:
        pre_evidence = _capture_core_evidence(
            context,
            preflight_artifact_dir,
            label="pre-task",
            ensure_home=True,
        )
    except _AllBlackCoreScreenshot as exc:
        if initial_reset_mode != "snapshot":
            raise
        graphics_fallback_reason = str(exc)
        fallback_dir = artifact_dir / "snapshot-graphics-cold-boot"
        fallback_dir.mkdir()
        (artifact_dir / "snapshot-graphics-fallback.json").write_text(
            json.dumps(
                {
                    "snapshot_attempted": True,
                    "snapshot_restored": True,
                    "failure": "all_black_public_core_framebuffer",
                    "reason": graphics_fallback_reason,
                    "fallback": "cold-boot",
                    "wipe_data": False,
                    "rerun_after_fallback": [
                        "mobilerun setup",
                        "mobilerun ping --no-tcp",
                        "mobilerun ping --tcp",
                        "mobilerun doctor",
                        "public mobilerun-core evidence",
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            _cold_boot_owned_emulator(context.emulator, fallback_dir)
        except RuntimeError as cold_error:
            pytest.fail(
                "snapshot framebuffer was all-black and the no-wipe-data "
                f"cold-boot fallback failed: {cold_error}"
            )
        preflight_artifact_dir = fallback_dir
        preflight = _setup_and_check_device(context, preflight_artifact_dir)
        try:
            pre_evidence = _capture_core_evidence(
                context,
                preflight_artifact_dir,
                label="pre-task",
                ensure_home=True,
            )
        except _AllBlackCoreScreenshot:
            pytest.fail(
                "public mobilerun-core framebuffer remained all-black after "
                "the no-wipe-data cold-boot fallback"
            )
        final_reset_mode = "snapshot_graphics_cold_boot_fallback"

    home_package = pre_evidence.current_app_id
    assert home_package, "mobilerun-core could not identify the Home package"
    timeout_labels = _timeout_labels(
        _adb(
            context.serial,
            "shell",
            "settings",
            "get",
            "system",
            "screen_off_timeout",
        )
    )
    (artifact_dir / "preflight.json").write_text(
        json.dumps(
            {
                "scenario_id": scenario.scenario_id,
                "serial": context.serial,
                "avd": _AVD_NAME,
                "emulator_port": _EMULATOR_PORT,
                "snapshot": _SNAPSHOT_NAME,
                "snapshot_attempted": True,
                "snapshot_restored": initial_reset_mode == "snapshot",
                "initial_reset_mode": initial_reset_mode,
                "reset_mode": final_reset_mode,
                "snapshot_graphics_fallback_reason": graphics_fallback_reason,
                "cold_boot_without_wipe_data": final_reset_mode
                in {
                    "cold_boot_fallback",
                    "snapshot_graphics_cold_boot_fallback",
                },
                "boot_completed": True,
                "mobilerun_setup": "passed",
                "android_version": preflight.android_version,
                "sdk_level": preflight.sdk_level,
                "portal_version_after_setup": preflight.portal_version_after_setup,
                "portal_version_after_doctor": preflight.portal_version_after_doctor,
                "portal_accessibility": "enabled",
                "content_provider_ping": "passed",
                "tcp_ping": "passed",
                "mobilerun_doctor": "passed",
                "portal_mode": "required",
                "auto_setup_during_task": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _assert_artifacts_redacted(artifact_dir, context.configured_secrets)
    return _ScenarioDeviceState(
        home_package=home_package,
        android_version=preflight.android_version,
        timeout_labels=timeout_labels,
        pre_evidence=pre_evidence,
    )


def _load_and_assert_trajectory(trajectory_root: Path) -> tuple[Path, tuple[dict[str, object], ...]]:
    trajectory_files = sorted(trajectory_root.glob("*/trajectory.json"))
    assert len(trajectory_files) == 1, (
        "each Grok scenario must produce exactly one trajectory.json artifact"
    )
    trajectory_file = trajectory_files[0]
    try:
        raw_events = json.loads(trajectory_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        pytest.fail("trajectory.json was not valid JSON")
    assert isinstance(raw_events, list) and raw_events, (
        "trajectory.json must contain recorded MobileRun events"
    )
    events = tuple(event for event in raw_events if isinstance(event, dict))
    assert len(events) == len(raw_events), "trajectory events must be JSON objects"

    response_events = [
        event
        for event in events
        if str(event.get("type", "")).endswith("ResponseEvent")
    ]
    assert any(
        isinstance(event.get("usage"), dict)
        and int(event["usage"].get("requests", 0) or 0) > 0
        and int(event["usage"].get("total_tokens", 0) or 0) > 0
        for event in response_events
    ), "trajectory must include positive request and token usage on a ResponseEvent"

    assert any(
        event.get("type") == "ToolExecutionEvent" and event.get("success") is True
        for event in events
    ), "trajectory must include a successful ToolExecutionEvent"
    direct_completion = any(
        event.get("type") == "ToolExecutionEvent"
        and event.get("tool_name") == "complete"
        and event.get("success") is True
        and isinstance(event.get("tool_args"), dict)
        and event["tool_args"].get("success") is True
        for event in events
    )
    reasoning_completion = any(
        event.get("type") == "ManagerResponseEvent"
        and isinstance(event.get("response"), str)
        and re.search(
            r"<request_accomplished\b[^>]*\bsuccess=[\"']true[\"']",
            event["response"],
            flags=re.IGNORECASE,
        )
        for event in events
    )
    assert direct_completion or reasoning_completion, (
        "trajectory must include a successful MobileRun completion result"
    )

    trajectory_dir = trajectory_file.parent
    ui_state_files = sorted((trajectory_dir / "ui_states").glob("*.json"))
    screenshot_files = sorted((trajectory_dir / "screenshots").glob("*.png"))
    assert ui_state_files, "trajectory must include recorded UI-state artifacts"
    assert screenshot_files, "trajectory must include screenshot artifacts"
    for ui_state_file in ui_state_files:
        try:
            ui_state = json.loads(ui_state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pytest.fail("a recorded UI-state artifact was not valid JSON")
        assert isinstance(ui_state, list), "recorded UI state must be a JSON list"
    for screenshot_file in screenshot_files:
        screenshot = screenshot_file.read_bytes()
        assert screenshot.startswith(b"\x89PNG\r\n\x1a\n"), (
            "recorded screenshot must be a non-empty PNG artifact"
        )
    return trajectory_dir, events


def _recorded_screen_timeout_label(trajectory_dir: Path) -> str:
    """Read the visible Settings summary from independently recorded Portal UI."""

    for ui_state_file in sorted((trajectory_dir / "ui_states").glob("*.json")):
        raw_nodes = json.loads(ui_state_file.read_text(encoding="utf-8"))
        if not isinstance(raw_nodes, list):
            continue
        for index, node in enumerate(raw_nodes):
            if not isinstance(node, dict) or str(node.get("text", "")).casefold() != "screen timeout":
                continue
            for summary in raw_nodes[index + 1 : index + 6]:
                if not isinstance(summary, dict):
                    continue
                resource_id = str(summary.get("resourceId", ""))
                label = str(summary.get("text", "")).strip()
                if resource_id.endswith("id/summary") and label:
                    return label
    pytest.fail("recorded Portal UI did not contain the visible Screen timeout summary")


def _run_scenario(
    context: _LiveContext, scenario: _Scenario
) -> _ScenarioResult:
    artifact_dir = context.artifact_root / scenario.scenario_id
    artifact_dir.mkdir(parents=False, exist_ok=False)
    trajectory_root = artifact_dir / "trajectories"
    config_path = artifact_dir / "config.yaml"
    (artifact_dir / "scenario.json").write_text(
        json.dumps(
            {
                "scenario_id": scenario.scenario_id,
                "snapshot": _SNAPSHOT_NAME,
                "provider": scenario.provider,
                "auth_mode": scenario.auth_mode,
                "model": "grok-4.5",
                "mode_flags": list(scenario.mode_flags),
                "portal_mode": "required",
                "auto_setup": False,
                "tcp": True,
                "tracing": False,
                "telemetry": False,
                "trajectory": "action",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    device_state = _prepare_scenario_device(context, scenario, artifact_dir)
    _write_e2e_config(config_path, context.serial, trajectory_root)
    command = [
        sys.executable,
        "-m",
        "mobilerun",
        "run",
        scenario.task,
        "--config",
        str(config_path),
        "--device",
        context.serial,
        "--tcp",
        "--provider",
        scenario.provider,
        "--model",
        "grok-4.5",
        "--steps",
        "30",
        "--no-stream",
        "--no-tracing",
        "--no-debug",
        "--save-trajectory",
        "action",
        *scenario.mode_flags,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "ANDROID_SERIAL": context.serial,
            "MOBILERUN_CONFIG": str(config_path),
            "MOBILERUN_TELEMETRY_ENABLED": "false",
            "DROIDRUN_TELEMETRY_ENABLED": "false",
        }
    )
    # The natural-language agent CLI is the Grok integration under test. All
    # harness-owned device actions and evidence use mobilerun_core.Mobilerun.
    completed = subprocess.run(
        command,
        cwd=context.repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20 * 60,
    )
    output = _ANSI_ESCAPE.sub("", completed.stdout + "\n" + completed.stderr)
    # OAuth may rotate tokens during inference. Scan for both the credentials
    # that existed before the suite and the newly persisted credential values.
    runtime_secrets = set(context.configured_secrets)
    runtime_secrets.update(_load_grok_oauth_secrets())
    runtime_secrets.add(os.environ["XAI_API_KEY"])
    secrets = tuple(runtime_secrets)
    _assert_credentials_redacted(output, secrets)
    (artifact_dir / "mobilerun-run.txt").write_text(output, encoding="utf-8")
    post_evidence = _capture_core_evidence(
        context,
        artifact_dir,
        label="post-task",
        ensure_home=False,
    )
    _assert_artifacts_redacted(artifact_dir, secrets)
    assert completed.returncode == 0, _sanitized_tail(
        output, secrets
    )
    trajectory_dir, events = _load_and_assert_trajectory(trajectory_root)
    return _ScenarioResult(
        output=output,
        artifact_dir=artifact_dir,
        trajectory_dir=trajectory_dir,
        events=events,
        device_state=device_state,
        post_evidence=post_evidence,
    )


def _assert_scenario_state(
    context: _LiveContext,
    scenario: _Scenario,
    result: _ScenarioResult,
) -> None:
    normalized_output = " ".join(result.output.casefold().split())
    assert result.post_evidence.current_app_id == result.device_state.home_package, (
        "Grok did not finish the task on the Home screen"
    )
    assert _focused_package(context.serial) == result.device_state.home_package, (
        "independent foreground diagnostics did not confirm the Home screen"
    )
    if scenario.expected_state == "screen_timeout":
        visible_label = _recorded_screen_timeout_label(result.trajectory_dir)
        assert " ".join(visible_label.casefold().split()) in normalized_output, (
            "Grok did not report the emulator's visible Screen timeout value"
        )
        return

    assert result.device_state.android_version.casefold() in normalized_output, (
        "Grok did not report the emulator's Android version"
    )
    assert (
        _adb(context.serial, "shell", "getprop", "ro.build.version.release")
        == result.device_state.android_version
    )


def test_xai_api_direct_ui_tree_screen_timeout(live_context: _LiveContext) -> None:
    result = _run_scenario(live_context, _API_DIRECT)
    _assert_scenario_state(live_context, _API_DIRECT, result)


def test_xai_api_reasoning_vision_only_android_version(
    live_context: _LiveContext,
) -> None:
    result = _run_scenario(live_context, _API_REASONING_VISION)
    _assert_scenario_state(live_context, _API_REASONING_VISION, result)


def test_grok_oauth_direct_ui_tree_screen_timeout(
    live_context: _LiveContext,
) -> None:
    result = _run_scenario(live_context, _OAUTH_DIRECT)
    _assert_scenario_state(live_context, _OAUTH_DIRECT, result)


def test_grok_oauth_reasoning_vision_only_android_version(
    live_context: _LiveContext,
) -> None:
    result = _run_scenario(live_context, _OAUTH_REASONING_VISION)
    _assert_scenario_state(live_context, _OAUTH_REASONING_VISION, result)
