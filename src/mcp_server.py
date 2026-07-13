"""
Spot Robot MCP server.

Exposed via start.py. Functions can also be imported directly.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(Path(__file__).parent.parent / ".env")

# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("Spot Robot Controller")


@mcp.tool()
def navigate_to_waypoint(waypoint_name: str) -> str:
    """Navigate the Spot robot to a named waypoint, e.g. 'waypoint_59'."""
    from .spot_agent import SpotAgent
    with SpotAgent(stream=False, navigation=True) as agent:
        wp_id = agent.get_waypoint_id_by_name(agent.graph, waypoint_name)
        agent.navigate_to_waypoint(wp_id)
    return f"Successfully navigated to {waypoint_name}"


def run_mcp(transport: str = "sse") -> None:
    mcp.run(transport=transport)
