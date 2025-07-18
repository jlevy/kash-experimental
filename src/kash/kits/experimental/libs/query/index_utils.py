from collections.abc import Callable, Sequence
from functools import partial
from types import NoneType
from typing import Any


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, str]:
    """
    Flatten a dict with nested structure to a single level.
    """
    items: list[tuple[str, str]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                items.append((f"{new_key}{sep}{i}", item))
        else:
            items.append((new_key, v))
    return dict(items)


def drop_non_atomic(d: dict[str, Any]) -> dict[str, Any]:
    """
    Drop values that are not atomic (str, int, float, None).
    """
    allowed_types = (str, int, float, NoneType)
    return {k: v for k, v in d.items() if isinstance(v, allowed_types)}


def tiktoken_tokenizer(model: str = "gpt-4o") -> Callable[[str], Sequence[int]]:
    import tiktoken

    enc = tiktoken.encoding_for_model(model)
    tokenizer = partial(enc.encode, allowed_special="all")
    return tokenizer
