import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from my_langchain_community.chat_models.oci_generative_ai import (
        ChatOCIGenAI,  # noqa: F401
    )

__all__ = [
    "ChatOCIGenAI",
]

_module_lookup = {
    "ChatOCIGenAI": "my_langchain_community.chat_models.oci_generative_ai",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
