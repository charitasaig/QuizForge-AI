"""
conftest.py — pytest configuration for QuizForge AI tests.
Adds the project root to sys.path so backend imports work
without needing to install the package.
"""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
