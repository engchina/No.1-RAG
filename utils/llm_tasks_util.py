"""
LLMタスク処理ユーティリティモジュール

このモジュールは、各種LLMモデルのタスク処理を行うための関数を提供します。
OCI GenAI、OpenAI、Azure OpenAIなどの様々なLLMプロバイダーに対応しています。
"""

import logging
import os
import time

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI

# ロガーの設定
logger = logging.getLogger(__name__)

# my_langchain_community.chat_models から ChatOCIGenAI をインポート
from my_langchain_community.chat_models import ChatOCIGenAI
from utils.common_util import get_region
from utils.langfuse_util import get_safe_stream_config
from utils.image_util import encode_image


async def xai_grok_4_task(system_text, query_text, xai_grok_4_checkbox):
    """XAI Grok-4モデルでのタスク処理"""
    region = get_region()
    if xai_grok_4_checkbox:
        xai_grok_4 = ChatOCIGenAI(
            model_id="xai.grok-4",
            provider="xai",
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
        )

        if system_text:
            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=query_text),
            ]
        else:
            messages = [
                HumanMessage(content=query_text),
            ]
        start_time = time.time()

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("xai.grok-4")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in xai_grok_4.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                yield content
        except Exception as e:
            logger.error(f"XAI grok-4 ストリーム処理中にエラーが発生しました: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def command_a_task(system_text, query_text, command_a_checkbox):
    """Command-Aモデルでのタスク処理"""
    region = get_region()
    if command_a_checkbox:
        command_a = ChatOCIGenAI(
            model_id="cohere.command-a-03-2025",
            provider="cohere",
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
        )

        if system_text:
            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=query_text),
            ]
        else:
            messages = [
                HumanMessage(content=query_text),
            ]
        start_time = time.time()

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("cohere.command-a-03-2025")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in command_a.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                yield content
        except Exception as e:
            logger.error(f"Command-A stream処理中にエラーが発生しました: {e}")
            print(f"ERROR: Command-A streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもstream処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def llama_4_scout_task(system_text, query_image, query_text, llama_4_scout_checkbox):
    """Llama-4-Scoutモデルでのタスク処理"""
    region = get_region()
    if llama_4_scout_checkbox:
        llama_4_scout = ChatOCIGenAI(
            model_id="meta.llama-4-scout-17b-16e-instruct",
            provider="meta",
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
        )

        # 画像がある場合とない場合でメッセージを分ける
        if query_image is not None:
            # 画像がある場合
            base64_image = encode_image(query_image)
            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": query_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ])
            ]
        else:
            # 画像がない場合
            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=query_text)
            ]

        stream_config = get_safe_stream_config()
        start_time = time.time()

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in llama_4_scout.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                yield content
        except Exception as e:
            logger.error(f"Llama-4-Scout ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: Llama-4-Scout streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"
        

async def openai_gpt4o_task(system_text, query_text, openai_gpt4o_checkbox):
    """OpenAI GPT-4oモデルでのタスク処理"""
    if openai_gpt4o_checkbox:
        load_dotenv(find_dotenv())
        openai_gpt4o = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            top_p=0.75,
            seed=42,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        )

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=query_text)
        ]

        stream_config = get_safe_stream_config()
        start_time = time.time()

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in openai_gpt4o.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                yield content
        except Exception as e:
            logger.error(f"OpenAI GPT-4o ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: OpenAI GPT-4o streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def azure_openai_gpt4o_task(system_text, query_text, azure_openai_gpt4o_checkbox):
    """Azure OpenAI GPT-4oモデルでのタスク処理"""
    if azure_openai_gpt4o_checkbox:
        load_dotenv(find_dotenv())
        azure_openai_gpt4o = AzureChatOpenAI(
            deployment_name="gpt-4o",
            temperature=0,
            top_p=0.75,
            seed=42,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_GPT_4O"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_GPT_4O"],
        )

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=query_text)
        ]

        stream_config = get_safe_stream_config()
        start_time = time.time()

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in azure_openai_gpt4o.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                yield content
        except Exception as e:
            logger.error(f"Azure OpenAI GPT-4o ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: Azure OpenAI GPT-4o streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"
