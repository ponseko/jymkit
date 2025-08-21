"""Shared pytest configuration and fixtures for improved tests."""

import jax
import pytest

# Set up JAX for testing
jax.config.update("jax_enable_x64", False)  # Use float32 for faster tests
jax.config.update("jax_platform_name", "cpu")  # Force CPU for consistent testing


@pytest.fixture(scope="session", autouse=True)
def setup_jax():
    """Set up JAX configuration for all tests."""
    # Disable JIT compilation warnings during tests
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="jax")


@pytest.fixture
def rng_key():
    """Provide a consistent PRNG key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def rng_keys():
    """Provide multiple PRNG keys for tests that need them."""
    base_key = jax.random.PRNGKey(42)
    return jax.random.split(base_key, 10)


@pytest.fixture
def minimal_env():
    """Provide a minimal test environment."""
    from .helpers import MockEnvironment
    return MockEnvironment()


# Pytest collection hooks for better test organization
def pytest_collection_modifyitems(config, items):
    """Add marks to tests based on their location."""
    for item in items:
        # Add marks based on test file path
        if "unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.fast)
        elif "integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.fast)
        elif "performance/" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Add external mark for tests that use external libraries
        if any(keyword in str(item.fspath) for keyword in ["gymnax", "brax", "jumanji", "external"]):
            item.add_marker(pytest.mark.external)