# conftest.py — ensures pytest can import from src/
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
