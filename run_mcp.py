"""
Standalone entry point to run the Spot MCP Server.

Usage:
    python run_mcp.py
"""
import os

from src.mcp_server import run_mcp

if __name__ == "__main__":
    # You can change transport to "stdio" if connecting to Claude Desktop directly
    # "sse" is better for web-based tools and remote access.
    transport = os.getenv("MCP_TRANSPORT", "sse")
    
    print(f"🚀 Starting Spot MCP Server using {transport} transport...")
    run_mcp(transport=transport)
