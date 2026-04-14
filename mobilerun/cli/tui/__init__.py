"""Mobilerun Terminal User Interface."""

from mobilerun.cli.tui.app import MobileTUI


def run_tui():
    """Run the Mobilerun TUI application."""
    app = MobileTUI()
    app.run()


__all__ = ["MobileTUI", "run_tui"]
