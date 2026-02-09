#!/usr/bin/env python
"""
MCP Server Runner for Ultra Doc-Intelligence

Entry point for running the MCP server via stdio.
"""

import asyncio
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Load environment variables
from dotenv import load_dotenv
import os
env_path = backend_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Import and run server
from mcp_server import main

if __name__ == "__main__":
    asyncio.run(main())
