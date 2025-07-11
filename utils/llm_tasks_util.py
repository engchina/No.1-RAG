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


async def xai_grok_3_task(system_text, query_text, xai_grok_3_checkbox):
    """XAI Grok-3モデルでのタスク処理"""
    region = get_region()
    if xai_grok_3_checkbox:
        print(f"DEBUG: Starting XAI Grok-3 task with query: {query_text[:100]}...")
        xai_grok_3 = ChatOCIGenAI(
            model_id="xai.grok-3",
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
        print(f"DEBUG: XAI Grok-3 start_time={start_time}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("xai.grok-3")
        print(f"DEBUG: XAI Grok-3 stream_config={stream_config}")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in xai_grok_3.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                print(f"DEBUG: XAI Grok-3 chunk {chunk_count}: '{content}' (length: {len(content)})")
                yield content
        except Exception as e:
            logger.error(f"XAI Grok-3 ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: XAI Grok-3 streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        print(f"DEBUG: XAI Grok-3 end_time={end_time}")
        inference_time = end_time - start_time
        print(f"DEBUG: XAI Grok-3 total chunks: {chunk_count}, total content length: {len(total_content)}")
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        print("DEBUG: XAI Grok-3 task skipped (checkbox not selected)")
        yield "TASK_DONE"


async def command_a_task(system_text, query_text, command_a_checkbox):
    """Command-Aモデルでのタスク処理"""
    region = get_region()
    if command_a_checkbox:
        print(f"DEBUG: Starting Command-A task with query: {query_text[:100]}...")
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
        print(f"DEBUG: Command-A start_time={start_time}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("cohere.command-a-03-2025")
        print(f"DEBUG: Command-A stream_config={stream_config}")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in command_a.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                print(f"DEBUG: Command-A chunk {chunk_count}: '{content}' (length: {len(content)})")
                yield content
        except Exception as e:
            logger.error(f"Command-A stream処理中にエラーが発生しました: {e}")
            print(f"ERROR: Command-A streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもstream処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        print(f"DEBUG: Command-A end_time={end_time}")
        inference_time = end_time - start_time
        print(f"DEBUG: Command-A total chunks: {chunk_count}, total content length: {len(total_content)}")
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        print("DEBUG: Command-A task skipped (checkbox not selected)")
        yield "TASK_DONE"


async def llama_3_3_70b_task(system_text, query_text, llama_3_3_70b_checkbox):
    """Llama-3.3-70Bモデルでのタスク処理"""
    region = get_region()
    if llama_3_3_70b_checkbox:
        print(f"DEBUG: Starting Llama-3.3-70B task with query: {query_text[:100]}...")
        llama_3_3_70b = ChatOCIGenAI(
            model_id="meta.llama-3.3-70b-instruct",
            provider="meta",
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
        print(f"DEBUG: Llama-3.3-70B start_time={start_time}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("meta.llama-3.3-70b-instruct")
        print(f"DEBUG: Llama-3.3-70B stream_config={stream_config}")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in llama_3_3_70b.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                print(f"DEBUG: Llama-3.3-70B chunk {chunk_count}: '{content}' (length: {len(content)})")
                yield content
        except Exception as e:
            logger.error(f"Llama-3.3-70B ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: Llama-3.3-70B streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        print(f"DEBUG: Llama-3.3-70B end_time={end_time}")
        inference_time = end_time - start_time
        print(f"DEBUG: Llama-3.3-70B total chunks: {chunk_count}, total content length: {len(total_content)}")
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        print("DEBUG: Llama-3.3-70B task skipped (checkbox not selected)")
        yield "TASK_DONE"


async def llama_3_2_90b_vision_task(system_text, query_image, query_text, llama_3_2_90b_vision_checkbox):
    """Llama-3.2-90B-Visionモデルでのタスク処理"""
    region = get_region()
    if llama_3_2_90b_vision_checkbox:
        print(f"DEBUG: Starting Llama-3.2-90B-Vision task with query: {query_text[:100]}...")
        print(f"DEBUG: Llama-3.2-90B-Vision has image: {query_image is not None}")
        llama_3_2_90b_vision = ChatOCIGenAI(
            model_id="meta.llama-3.2-90b-vision-instruct",
            provider="meta",
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600, "presence_penalty": 2,
                          "frequency_penalty": 2},
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
        print(f"DEBUG: Llama-3.2-90B-Vision start_time={start_time}")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in llama_3_2_90b_vision.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                print(f"DEBUG: Llama-3.2-90B-Vision chunk {chunk_count}: '{content}' (length: {len(content)})")
                yield content
        except Exception as e:
            logger.error(f"Llama-3.2-90B-Vision ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: Llama-3.2-90B-Vision streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        print(f"DEBUG: Llama-3.2-90B-Vision end_time={end_time}")
        inference_time = end_time - start_time
        print(f"DEBUG: Llama-3.2-90B-Vision total chunks: {chunk_count}, total content length: {len(total_content)}")
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        print("DEBUG: Llama-3.2-90B-Vision task skipped (checkbox not selected)")
        yield "TASK_DONE"


async def llama_4_maverick_task(system_text, query_image, query_text, llama_4_maverick_checkbox):
    """Llama-4-Maverickモデルでのタスク処理"""
    region = get_region()
    if llama_4_maverick_checkbox:
        print(f"DEBUG: Starting Llama-4-Maverick task with query: {query_text[:100]}...")
        print(f"DEBUG: Llama-4-Maverick has image: {query_image is not None}")
        llama_4_maverick = ChatOCIGenAI(
            model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
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
        print(f"DEBUG: Llama-4-Maverick start_time={start_time}")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in llama_4_maverick.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                print(f"DEBUG: Llama-4-Maverick chunk {chunk_count}: '{content}' (length: {len(content)})")
                yield content
        except Exception as e:
            logger.error(f"Llama-4-Maverick ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: Llama-4-Maverick streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        print(f"DEBUG: Llama-4-Maverick end_time={end_time}")
        inference_time = end_time - start_time
        print(f"DEBUG: Llama-4-Maverick total chunks: {chunk_count}, total content length: {len(total_content)}")
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        print("DEBUG: Llama-4-Maverick task skipped (checkbox not selected)")
        yield "TASK_DONE"


async def llama_4_scout_task(system_text, query_image, query_text, llama_4_scout_checkbox):
    """Llama-4-Scoutモデルでのタスク処理"""
    region = get_region()
    if llama_4_scout_checkbox:
        print(f"DEBUG: Starting Llama-4-Scout task with query: {query_text[:100]}...")
        print(f"DEBUG: Llama-4-Scout has image: {query_image is not None}")
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
        print(f"DEBUG: Llama-4-Scout start_time={start_time}")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in llama_4_scout.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                print(f"DEBUG: Llama-4-Scout chunk {chunk_count}: '{content}' (length: {len(content)})")
                yield content
        except Exception as e:
            logger.error(f"Llama-4-Scout ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: Llama-4-Scout streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        print(f"DEBUG: Llama-4-Scout end_time={end_time}")
        inference_time = end_time - start_time
        print(f"DEBUG: Llama-4-Scout total chunks: {chunk_count}, total content length: {len(total_content)}")
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        print("DEBUG: Llama-4-Scout task skipped (checkbox not selected)")
        yield "TASK_DONE"


async def openai_gpt4o_task(system_text, query_text, openai_gpt4o_checkbox):
    """OpenAI GPT-4oモデルでのタスク処理"""
    if openai_gpt4o_checkbox:
        print(f"DEBUG: Starting OpenAI GPT-4o task with query: {query_text[:100]}...")
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
        print(f"DEBUG: OpenAI GPT-4o start_time={start_time}")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in openai_gpt4o.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                print(f"DEBUG: OpenAI GPT-4o chunk {chunk_count}: '{content}' (length: {len(content)})")
                yield content
        except Exception as e:
            logger.error(f"OpenAI GPT-4o ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: OpenAI GPT-4o streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        print(f"DEBUG: OpenAI GPT-4o end_time={end_time}")
        inference_time = end_time - start_time
        print(f"DEBUG: OpenAI GPT-4o total chunks: {chunk_count}, total content length: {len(total_content)}")
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        print("DEBUG: OpenAI GPT-4o task skipped (checkbox not selected)")
        yield "TASK_DONE"


async def openai_gpt4_task(system_text, query_text, openai_gpt4_checkbox):
    """OpenAI GPT-4モデルでのタスク処理"""
    if openai_gpt4_checkbox:
        print(f"DEBUG: Starting OpenAI GPT-4 task with query: {query_text[:100]}...")
        load_dotenv(find_dotenv())
        openai_gpt4 = ChatOpenAI(
            model="gpt-4",
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
        print(f"DEBUG: OpenAI GPT-4 start_time={start_time}")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in openai_gpt4.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                print(f"DEBUG: OpenAI GPT-4 chunk {chunk_count}: '{content}' (length: {len(content)})")
                yield content
        except Exception as e:
            logger.error(f"OpenAI GPT-4 ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: OpenAI GPT-4 streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        print(f"DEBUG: OpenAI GPT-4 end_time={end_time}")
        inference_time = end_time - start_time
        print(f"DEBUG: OpenAI GPT-4 total chunks: {chunk_count}, total content length: {len(total_content)}")
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        print("DEBUG: OpenAI GPT-4 task skipped (checkbox not selected)")
        yield "TASK_DONE"


async def azure_openai_gpt4o_task(system_text, query_text, azure_openai_gpt4o_checkbox):
    """Azure OpenAI GPT-4oモデルでのタスク処理"""
    if azure_openai_gpt4o_checkbox:
        print(f"DEBUG: Starting Azure OpenAI GPT-4o task with query: {query_text[:100]}...")
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
        print(f"DEBUG: Azure OpenAI GPT-4o start_time={start_time}")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in azure_openai_gpt4o.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                print(f"DEBUG: Azure OpenAI GPT-4o chunk {chunk_count}: '{content}' (length: {len(content)})")
                yield content
        except Exception as e:
            logger.error(f"Azure OpenAI GPT-4o ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: Azure OpenAI GPT-4o streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        print(f"DEBUG: Azure OpenAI GPT-4o end_time={end_time}")
        inference_time = end_time - start_time
        print(f"DEBUG: Azure OpenAI GPT-4o total chunks: {chunk_count}, total content length: {len(total_content)}")
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        print("DEBUG: Azure OpenAI GPT-4o task skipped (checkbox not selected)")
        yield "TASK_DONE"


async def azure_openai_gpt4_task(system_text, query_text, azure_openai_gpt4_checkbox):
    """Azure OpenAI GPT-4モデルでのタスク処理"""
    if azure_openai_gpt4_checkbox:
        print(f"DEBUG: Starting Azure OpenAI GPT-4 task with query: {query_text[:100]}...")
        load_dotenv(find_dotenv())
        azure_openai_gpt4 = AzureChatOpenAI(
            deployment_name="gpt-4",
            temperature=0,
            top_p=0.75,
            seed=42,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_GPT_4"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_GPT_4"],
        )

        messages = [
            SystemMessage(content=system_text),
            HumanMessage(content=query_text)
        ]

        stream_config = get_safe_stream_config()
        start_time = time.time()
        print(f"DEBUG: Azure OpenAI GPT-4 start_time={start_time}")

        chunk_count = 0
        total_content = ""
        try:
            async for chunk in azure_openai_gpt4.astream(messages, config=stream_config):
                chunk_count += 1
                content = chunk.content if chunk.content else ""
                total_content += content
                print(f"DEBUG: Azure OpenAI GPT-4 chunk {chunk_count}: '{content}' (length: {len(content)})")
                yield content
        except Exception as e:
            logger.error(f"Azure OpenAI GPT-4 ストリーム処理中にエラーが発生しました: {e}")
            print(f"ERROR: Azure OpenAI GPT-4 streaming failed after {chunk_count} chunks: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"

        end_time = time.time()
        print(f"DEBUG: Azure OpenAI GPT-4 end_time={end_time}")
        inference_time = end_time - start_time
        print(f"DEBUG: Azure OpenAI GPT-4 total chunks: {chunk_count}, total content length: {len(total_content)}")
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        print("DEBUG: Azure OpenAI GPT-4 task skipped (checkbox not selected)")
        yield "TASK_DONE"
