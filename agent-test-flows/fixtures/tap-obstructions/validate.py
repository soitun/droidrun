"""Capture an indexed click and its independently observed device outcome."""

import argparse
import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

from mobilerun_core import Mobilerun

from mobilerun.agent.utils.actions import click
from mobilerun.tools.filters import ConciseFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.provider import AndroidStateProvider


async def validate(args):
    device = await asyncio.to_thread(
        Mobilerun().connect, args.serial, backend="local-android-adb"
    )

    class Driver:
        def __init__(self):
            self.raw = None
            self.taps = []

        async def get_ui_tree(self):
            self.raw = await asyncio.to_thread(device.ui)
            return self.raw

        async def tap(self, x, y):
            self.taps.append([x, y])
            await asyncio.to_thread(device.tap, x, y, stealth=False)

    driver = Driver()
    provider = AndroidStateProvider(driver, ConciseFilter(), IndexedFormatter())
    args.output.mkdir(parents=True, exist_ok=True)

    async def snapshot(name):
        state = await provider.get_state()
        (args.output / f"{name}.json").write_text(json.dumps(driver.raw, indent=2))
        screenshot = await asyncio.to_thread(device.screenshot)
        (args.output / f"{name}.png").write_bytes(base64.b64decode(screenshot))
        return state

    if args.launch:
        await asyncio.to_thread(device.start_app, args.launch)
        await asyncio.sleep(1)
    before = await snapshot("before")
    if args.target is None and args.index is None:
        print(before.formatted_text)
        return
    matches = [
        e
        for e in before.elements
        if (
            e["text"] == args.target
            if args.target is not None
            else e["index"] == args.index
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one target, found {len(matches)}")
    target = matches[0]
    result = await click(
        target["index"],
        ctx=SimpleNamespace(ui=before, driver=driver, macro_recorder=None),
    )
    await asyncio.sleep(0.5)
    after = await snapshot("after")
    texts = [e["text"] for e in after.elements]
    evidence = {
        "target": target,
        "taps": driver.taps,
        "action_success": result.success,
        "texts": texts,
        "expected": args.expect,
        "verified": any(args.expect in text for text in texts),
    }
    (args.output / "result.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
    if not result.success or not evidence["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch", metavar="PACKAGE")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target", help="Exact text from the formatted state")
    target.add_argument("--index", type=int, help="Index from a fresh state snapshot")
    parser.add_argument("--expect", help="Text required on the resulting screen")
    args = parser.parse_args()
    if (args.target is not None or args.index is not None) and not args.expect:
        parser.error("--target/--index requires --expect")
    asyncio.run(validate(args))
