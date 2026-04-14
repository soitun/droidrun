"""Mobilerun Terminal User Interface."""

from mobilerun.cli.tui.app import DroidTUI


def run_tui():
    """Run the Mobilerun TUI application."""
    app = DroidTUI()
    app.run()


__all__ = ["DroidTUI", "run_tui"]
