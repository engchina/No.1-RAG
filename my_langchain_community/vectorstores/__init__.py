import importlib
from typing import TYPE_CHECKING, Any

from my_langchain_community.vectorstores.myoraclevs import MyOracleVS

if TYPE_CHECKING:
    from my_langchain_community.vectorstores.oraclevs import (
        MyOracleVS,
    )

__all__ = [
    "MyOracleVS",
]

_module_lookup = {
    "MyOracleVS": "my_langchain_community.vectorstores.myoraclevs",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
