# Placeholder test to avoid heavy model imports
import pytest

@pytest.mark.skip(reason="Skipping heavy model loading in CI environment")
def test_placeholder():
    assert True