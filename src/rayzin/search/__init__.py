from typing import Any

from rayzin.search.backends import SearchBackend, make_search_backend
from rayzin.search.block_searcher import BlockSearcher

__all__ = [
    "SearchBackend",
    "make_search_backend",
    "NumpySearchBackend",
    "FaissSearchBackend",
    "BlockSearcher",
]


def __getattr__(name: str) -> Any:
    if name in {"NumpySearchBackend", "FaissSearchBackend"}:
        from rayzin.search.backends import __getattr__ as backends_getattr

        return backends_getattr(name)
    raise AttributeError(name)
