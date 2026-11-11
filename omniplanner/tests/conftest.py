"""Pytest configuration for omniplanner tests."""

import sys
from pathlib import Path

# Add directories to path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
omniplanner_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(omniplanner_src))
sys.path.insert(0, str(repo_root))
