try:
    from ._tree import (
        batch_sum as batch_sum,
        concatenate as concatenate,
        gather_actions as gather_actions,
        get_first as get_first,
        map_distribution as map_distribution,
        map_one_level as map_one_level,
        mean as mean,
        split_key_like_structure as split_key_like_structure,
        stack as stack,
        unstack as unstack,
    )

    __all__ = [
        "get_first",
        "map_one_level",
        "mean",
        "stack",
        "unstack",
        "concatenate",
        "map_distribution",
        "batch_sum",
        "gather_actions",
        "split_key_like_structure",
    ]
except ImportError:
    print(
        "Jaxnasium.tree module requires `optax` to be installed. Please install via `pip install optax`."
    )
