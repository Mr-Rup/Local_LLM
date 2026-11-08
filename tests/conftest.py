"""Pytest configuration and shared fixtures for LocalAI test suite."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path for test discovery
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
