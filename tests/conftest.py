"""
tests/conftest.py

Shared pytest configuration and fixtures.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on sys.path so imports work from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
