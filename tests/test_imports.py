"""
Smoke tests to ensure core modules import and basic classes are available.
These tests avoid network calls and validate packaging/import paths.
"""

import importlib.util


def test_can_import_data_validator():
    from src.data.data_validator import DataValidator  # noqa: F401


def test_can_import_data_manager():
    openbb_available = importlib.util.find_spec("openbb") is not None
    if not openbb_available:
        return  # Skip when openbb isn't installed
    from src.data.data_manager import DataManager  # noqa: F401

def test_can_import_data_loader():
    openbb_available = importlib.util.find_spec("openbb") is not None
    if not openbb_available:
        return  # Skip when openbb isn't installed
    from src.data.data_loader import DataLoader  # noqa: F401
