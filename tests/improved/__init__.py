"""Improved test suite for jymkit with better organization and performance."""

# This module contains reorganized tests with better separation of concerns:
# - unit/: Fast, isolated tests for individual components
# - integration/: Fast tests for component interactions  
# - performance/: Slower tests that verify actual learning performance
# - helpers/: Shared utilities to reduce code duplication