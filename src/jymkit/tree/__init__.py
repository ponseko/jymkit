try:
    from ._tree import (
        tree_get_first as tree_get_first,
        tree_map_one_level as tree_map_one_level,
        tree_mean as tree_mean,
        tree_stack as tree_stack,
        tree_unstack as tree_unstack,
    )

    get_first = tree_get_first
    map_one_level = tree_map_one_level
    mean = tree_mean
    stack = tree_stack
    unstack = tree_unstack

    __all__ = ["get_first", "map_one_level", "mean", "stack", "unstack"]
except ImportError:
    print(
        "JymKit.tree module requires `optax` to be installed. Please install via `pip install optax`."
    )
