"""Pytest configuration for world-model-adversarial tests."""

import sys
from pathlib import Path

# Ensure the submission root is on sys.path so `from src import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))
