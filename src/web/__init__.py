"""The browser console for Spot: one page for the operator — video, missions, Stop and the log.

``python start.py`` builds one `SpotAgent` and hands it to `SpotWebConsole.serve()`.
"""
from .console import MENU, SpotWebConsole

__all__ = ["MENU", "SpotWebConsole"]
