"""Search space definition — re-exported from opt_tool for backward compatibility."""

# These classes have moved to opt_tool.space.
# This module re-exports them so that existing imports from bo.space continue to work.
from opt_tool.space import HyperParameter, SearchSpace

__all__ = ["HyperParameter", "SearchSpace"]
