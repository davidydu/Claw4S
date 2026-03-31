# conftest.py -- ensures pytest can import from src/
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
