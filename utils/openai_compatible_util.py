"""
OpenAI-Compatible設定を扱うユーティリティ。
"""

import os

from dotenv import find_dotenv, load_dotenv


OPENAI_COMPATIBLE_MODEL_KEY = "openai_compatible"
DEFAULT_OPENAI_COMPATIBLE_MODEL = "gpt-4o"


def get_openai_compatible_model(load_env=True):
    if load_env:
        load_dotenv(find_dotenv())
    model = (os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_COMPATIBLE_MODEL).strip()
    return model or DEFAULT_OPENAI_COMPATIBLE_MODEL


def get_openai_compatible_llm_name():
    return f"{OPENAI_COMPATIBLE_MODEL_KEY}/{get_openai_compatible_model()}"


def get_openai_compatible_label():
    return f"OpenAI-Compatible/{get_openai_compatible_model()}"


def get_openai_compatible_message_label():
    return f"{get_openai_compatible_label()} メッセージ"


def get_openai_compatible_client_kwargs():
    load_dotenv(find_dotenv())
    project = (os.environ.get("OPENAI_PROJECT") or "").strip()
    client_kwargs = {
        "model": get_openai_compatible_model(load_env=False),
        "temperature": 0,
        "top_p": 0.75,
        "seed": 42,
        "max_tokens": None,
        "timeout": None,
        "max_retries": 2,
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": os.environ["OPENAI_BASE_URL"],
    }
    if project:
        client_kwargs["default_headers"] = {"OpenAI-Project": project}
    return client_kwargs
