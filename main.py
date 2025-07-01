import argparse
import base64
import json
import logging
import os
import platform
import re
import shutil
import time
from io import BytesIO
from itertools import combinations
from typing import List, Tuple

import cohere
import gradio as gr
import oci
import oracledb
import pandas as pd
import requests
from PIL import Image
from dotenv import load_dotenv, find_dotenv, set_key, get_key
from gradio.themes import GoogleFont
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langfuse.callback import CallbackHandler
from oracledb import DatabaseError
from unstructured.partition.auto import partition

from markitdown import MarkItDown
# from langchain_community.chat_models import ChatOCIGenAI
from my_langchain_community.chat_models import ChatOCIGenAI
from utils.chunk_util import RecursiveCharacterTextSplitter
from utils.common_util import get_dict_value
from utils.css_gradio import custom_css
from utils.generator_util import generate_unique_id
from utils.prompts import (
    get_sub_query_prompt, get_rag_fusion_prompt, get_hyde_prompt, get_step_back_prompt,
    get_langgpt_rag_prompt, get_llm_evaluation_system_message, get_chat_system_message,
    get_markitdown_llm_prompt, get_query_generation_prompt, update_langgpt_rag_prompt,
    get_image_qa_prompt, update_image_qa_prompt
)

# read local .env file
load_dotenv(find_dotenv())

DEFAULT_COLLECTION_NAME = os.environ["DEFAULT_COLLECTION_NAME"]

# ログ設定
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def compress_image_for_display(image_url, quality=85, max_width=800, max_height=1200):
    """
    画像URLを圧縮して表示用の新しいURLを生成する

    Args:
        image_url: 元の画像URL（data:image/...;base64,... 形式）
        quality: JPEG圧縮品質 (1-100)
        max_width: 最大幅
        max_height: 最大高さ

    Returns:
        str: 圧縮された画像のdata URL
    """
    output_buffer = None
    try:
        # data URLからbase64データを抽出
        if not image_url.startswith('data:image/'):
            return image_url

        # base64データ部分を取得
        header, base64_data = image_url.split(',', 1)

        # base64をデコードして画像データを取得
        image_data = base64.b64decode(base64_data)

        # PILで画像を開く
        with Image.open(BytesIO(image_data)) as img:
            # RGBモードに変換（必要に応じて）
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # 画像サイズを調整（アスペクト比を保持）
            original_width, original_height = img.size

            # 縮小が必要かチェック
            if original_width > max_width or original_height > max_height:
                # アスペクト比を保持して縮小
                ratio = min(max_width / original_width, max_height / original_height)
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"画像サイズを圧縮: {original_width}x{original_height} -> {new_width}x{new_height}")

            # 圧縮された画像をバイトストリームに保存
            output_buffer = BytesIO()
            img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
            compressed_data = output_buffer.getvalue()

            # base64エンコード
            compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')

            # 新しいdata URLを生成
            compressed_url = f"data:image/jpeg;base64,{compressed_base64}"

            # 圧縮率を計算してログ出力
            original_size = len(base64_data)
            compressed_size = len(compressed_base64)
            compression_ratio = (1 - compressed_size / original_size) * 100
            print(f"画像圧縮完了: {original_size} -> {compressed_size} bytes ({compression_ratio:.1f}% 削減)")

            return compressed_url

    except Exception as e:
        print(f"画像圧縮中にエラーが発生しました: {e}")
        return image_url  # エラー時は元の画像URLを返す
    finally:
        # BytesIOバッファを明示的に閉じる
        if output_buffer is not None:
            try:
                output_buffer.close()
            except Exception:
                pass  # クリーンアップエラーは無視


def cleanup_llm_client(llm_client):
    """
    LLMクライアントのリソースを安全にクリーンアップする（同期版）

    Args:
        llm_client: クリーンアップするLLMクライアント
    """
    if llm_client is None:
        return

    try:
        # OpenAI系クライアントの場合
        if hasattr(llm_client, 'client'):
            if hasattr(llm_client.client, 'close'):
                # 同期的にクローズ
                if hasattr(llm_client.client.close, '__call__'):
                    try:
                        llm_client.client.close()
                    except Exception as e:
                        print(f"OpenAI クライアントの同期クローズ中にエラー: {e}")

        # _clientアトリビュートを持つ場合
        elif hasattr(llm_client, '_client'):
            if hasattr(llm_client._client, 'close'):
                try:
                    llm_client._client.close()
                except Exception as e:
                    print(f"_client クローズ中にエラー: {e}")

            # OCI GenAI系クライアントのセッション処理
            if hasattr(llm_client._client, '_session'):
                if hasattr(llm_client._client._session, 'close'):
                    try:
                        llm_client._client._session.close()
                    except Exception as e:
                        print(f"OCI セッションクローズ中にエラー: {e}")

        print(f"LLMクライアント {type(llm_client).__name__} のリソースクリーンアップが完了しました")

    except Exception as cleanup_error:
        print(f"LLMクライアントのクリーンアップ中に予期しないエラーが発生しました: {cleanup_error}")


async def cleanup_llm_client_async(llm_client):
    """
    LLMクライアントのリソースを安全にクリーンアップする（非同期版）

    Args:
        llm_client: クリーンアップするLLMクライアント
    """
    if llm_client is None:
        return

    try:
        # OpenAI系クライアントの場合（非同期クローズ）
        if hasattr(llm_client, 'client'):
            # HTTPXクライアントの場合
            if hasattr(llm_client.client, 'aclose'):
                try:
                    await llm_client.client.aclose()
                    print(f"OpenAI HTTPXクライアントの非同期クローズが完了しました")
                except Exception as e:
                    # OpenAI API type エラーは無視
                    if "Ambiguous use of module client" not in str(e):
                        print(f"OpenAI クライアントの非同期クローズ中にエラー: {e}")
            elif hasattr(llm_client.client, 'close'):
                try:
                    llm_client.client.close()
                    print(f"OpenAI クライアントの同期クローズが完了しました")
                except Exception as e:
                    # OpenAI API type エラーは無視
                    if "Ambiguous use of module client" not in str(e):
                        print(f"OpenAI クライアントの同期クローズ中にエラー: {e}")

            # 追加のHTTP接続プール清理
            if hasattr(llm_client.client, '_client'):
                if hasattr(llm_client.client._client, 'aclose'):
                    try:
                        await llm_client.client._client.aclose()
                        print(f"OpenAI 内部HTTPクライアントの非同期クローズが完了しました")
                    except Exception as e:
                        if "Ambiguous use of module client" not in str(e):
                            print(f"OpenAI 内部HTTPクライアントクローズ中にエラー: {e}")

        # _clientアトリビュートを持つ場合（OCI GenAI等）
        elif hasattr(llm_client, '_client'):
            if hasattr(llm_client._client, 'aclose'):
                try:
                    await llm_client._client.aclose()
                    print(f"OCI _client の非同期クローズが完了しました")
                except Exception as e:
                    print(f"_client 非同期クローズ中にエラー: {e}")
            elif hasattr(llm_client._client, 'close'):
                try:
                    llm_client._client.close()
                    print(f"OCI _client の同期クローズが完了しました")
                except Exception as e:
                    print(f"_client クローズ中にエラー: {e}")

            # OCI GenAI系クライアントのセッション処理
            if hasattr(llm_client._client, '_session'):
                if hasattr(llm_client._client._session, 'aclose'):
                    try:
                        await llm_client._client._session.aclose()
                        print(f"OCI セッションの非同期クローズが完了しました")
                    except Exception as e:
                        print(f"OCI セッション非同期クローズ中にエラー: {e}")
                elif hasattr(llm_client._client._session, 'close'):
                    try:
                        llm_client._client._session.close()
                        print(f"OCI セッションの同期クローズが完了しました")
                    except Exception as e:
                        print(f"OCI セッションクローズ中にエラー: {e}")

        # 追加のリソース清理：HTTPコネクションプールの強制クリーンアップ（軽量版）
        try:
            await force_cleanup_http_connections(llm_client)
        except Exception as force_cleanup_error:
            # 強制クリーンアップのエラーは詳細を表示しない
            if "Ambiguous use of module client" not in str(force_cleanup_error):
                print(f"HTTP接続プール強制クリーンアップ中にエラー: {force_cleanup_error}")

        print(f"LLMクライアント {type(llm_client).__name__} の非同期リソースクリーンアップが完了しました")

    except Exception as cleanup_error:
        # OpenAI API type エラーは無視
        if "Ambiguous use of module client" not in str(cleanup_error):
            print(f"LLMクライアントの非同期クリーンアップ中に予期しないエラーが発生しました: {cleanup_error}")


async def force_cleanup_http_connections(llm_client):
    """
    HTTPコネクションプールを強制的にクリーンアップする

    Args:
        llm_client: LLMクライアント
    """
    try:
        # httpxライブラリのコネクションプールを探してクリーンアップ
        import httpx
        import gc

        # ガベージコレクションを実行してオブジェクトを収集
        gc.collect()

        # httpxクライアントを探してクリーンアップ（最大3つまで）
        cleaned_count = 0
        for obj in gc.get_objects():
            if cleaned_count >= 3:  # 処理数を制限
                break

            if isinstance(obj, httpx.AsyncClient):
                try:
                    if not obj.is_closed:
                        await obj.aclose()
                        cleaned_count += 1
                        print(f"未閉鎖のHTTPXクライアント #{cleaned_count} をクリーンアップしました")
                except Exception as e:
                    # OpenAI API type エラーは無視
                    if "Ambiguous use of module client" not in str(e):
                        print(f"HTTPXクライアントクリーンアップ中にエラー: {e}")
            elif isinstance(obj, httpx.Client):
                try:
                    if not obj.is_closed:
                        obj.close()
                        cleaned_count += 1
                        print(f"未閉鎖の同期HTTPXクライアント #{cleaned_count} をクリーンアップしました")
                except Exception as e:
                    # OpenAI API type エラーは無視
                    if "Ambiguous use of module client" not in str(e):
                        print(f"同期HTTPXクライアントクリーンアップ中にエラー: {e}")

    except ImportError:
        # httpxがインストールされていない場合は無視
        pass
    except Exception as e:
        # OpenAI API type エラーは無視
        if "Ambiguous use of module client" not in str(e):
            print(f"HTTP接続プール強制クリーンアップ中にエラー: {e}")


async def cleanup_all_http_connections():
    """
    システム全体のHTTP接続プールをクリーンアップする
    """
    try:
        import httpx
        import aiohttp
        import gc

        print("システム全体のHTTP接続プールクリーンアップを開始...")

        # ガベージコレクションを実行
        gc.collect()

        cleaned_count = 0

        # すべてのオブジェクトをスキャンしてHTTPクライアントを探す
        for obj in gc.get_objects():
            try:
                # httpxクライアント
                if isinstance(obj, httpx.AsyncClient):
                    if not obj.is_closed:
                        await obj.aclose()
                        cleaned_count += 1
                        print(f"HTTPXクライアント #{cleaned_count} をクリーンアップしました")
                elif isinstance(obj, httpx.Client):
                    if not obj.is_closed:
                        obj.close()
                        cleaned_count += 1
                        print(f"同期HTTPXクライアント #{cleaned_count} をクリーンアップしました")

                # aiohttpクライアント
                elif hasattr(obj, '__class__') and 'aiohttp' in str(obj.__class__):
                    if hasattr(obj, 'close') and not getattr(obj, 'closed', True):
                        try:
                            await obj.close()
                            cleaned_count += 1
                            print(f"aiohttpクライアント #{cleaned_count} をクリーンアップしました")
                        except Exception:
                            pass

            except Exception as e:
                # 個別のオブジェクトクリーンアップエラーは無視
                pass

        # 最終ガベージコレクション
        gc.collect()

        print(f"HTTP接続プールクリーンアップ完了: {cleaned_count}個のクライアントを処理しました")

    except ImportError:
        print("HTTP接続プールクリーンアップ: 必要なライブラリがインストールされていません")
    except Exception as e:
        print(f"システム全体のHTTP接続プールクリーンアップ中にエラー: {e}")


async def lightweight_cleanup():
    """
    軽量なリソースクリーンアップ（画像処理後に使用）
    """
    try:
        import gc

        # 軽量なガベージコレクション
        collected = gc.collect()
        if collected > 0:
            print(f"軽量クリーンアップ: {collected} オブジェクトを回収しました")

        # 基本的なHTTP接続チェック（重い処理は避ける）
        try:
            import httpx
            # 明らかに閉じられていないクライアントのみチェック
            cleaned_count = 0
            for obj in gc.get_objects():
                if isinstance(obj, httpx.AsyncClient) and hasattr(obj, 'is_closed'):
                    if not obj.is_closed:
                        try:
                            await obj.aclose()
                            cleaned_count += 1
                            print(f"未閉鎖のHTTPXクライアント #{cleaned_count} をクリーンアップしました")
                        except Exception as close_error:
                            # OpenAI API type エラーなどは無視
                            if "Ambiguous use of module client" not in str(close_error):
                                print(f"HTTPXクライアントクローズ中にエラー: {close_error}")
                        if cleaned_count >= 3:  # 最大3つまで処理
                            break
        except ImportError:
            pass
        except Exception as http_error:
            # HTTP関連のエラーは詳細を表示しない
            print(f"HTTP接続チェック中にエラーが発生しました（無視されます）")

    except Exception as e:
        print(f"軽量クリーンアップ中にエラー: {e}")


def enable_resource_warnings():
    """
    リソース警告を有効にして詳細な情報を取得する
    """
    import warnings
    import tracemalloc

    # ResourceWarningを有効にする
    warnings.filterwarnings("always", category=ResourceWarning)

    # tracemallocを有効にしてメモリ追跡を開始
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        print("リソース追跡が有効になりました")

    # カスタムwarning処理を設定
    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        if category == ResourceWarning:
            print(f"🚨 リソース警告: {message}")
            print(f"   ファイル: {filename}:{lineno}")
            if tracemalloc.is_tracing():
                # メモリ使用量の統計を表示
                current, peak = tracemalloc.get_traced_memory()
                print(f"   メモリ使用量: 現在={current / 1024 / 1024:.1f}MB, ピーク={peak / 1024 / 1024:.1f}MB")

    warnings.showwarning = custom_warning_handler


async def final_resource_cleanup():
    """
    プログラム終了時の最終リソースクリーンアップ
    """
    print("\n=== 最終リソースクリーンアップを開始 ===")

    try:
        # HTTP接続プールの全体クリーンアップ
        await cleanup_all_http_connections()

        # 追加のリソースクリーンアップ
        import gc
        import asyncio

        # 現在のタスクを取得（自分自身は除外）
        current_task = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks()
                 if not task.done() and task != current_task]

        if tasks:
            print(f"実行中のタスク {len(tasks)} 個をキャンセルします...")
            # タスクを安全にキャンセル
            for task in tasks:
                try:
                    if not task.done() and not task.cancelled():
                        task.cancel()
                except Exception as cancel_error:
                    print(f"タスクキャンセル中にエラー: {cancel_error}")

            # キャンセルされたタスクの完了を待つ（タイムアウト付き）
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=5.0
                )
                print("すべてのタスクがキャンセルされました")
            except asyncio.TimeoutError:
                print("タスクキャンセルがタイムアウトしました（一部のタスクが残っている可能性があります）")
            except Exception as gather_error:
                print(f"タスク待機中にエラー: {gather_error}")

        # 強制ガベージコレクション
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                print(f"ガベージコレクション {i + 1}: {collected} オブジェクトを回収しました")

        # メモリ統計を表示
        import tracemalloc
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            print(f"最終メモリ使用量: 現在={current / 1024 / 1024:.1f}MB, ピーク={peak / 1024 / 1024:.1f}MB")
            tracemalloc.stop()

        print("=== 最終リソースクリーンアップ完了 ===\n")

    except Exception as e:
        print(f"最終リソースクリーンアップ中にエラー: {e}")
        import traceback
        traceback.print_exc()


def check_langfuse_availability():
    """
    Langfuse サービスの可用性を事前に確認する

    Returns:
        bool: Langfuse サービスが利用可能な場合は True、そうでなければ False
    """
    try:
        # 環境変数の存在確認
        required_env_vars = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"]
        for var in required_env_vars:
            if not os.environ.get(var):
                logger.warning(f"Langfuse 環境変数 {var} が設定されていません")
                return False

        # Langfuse クライアントの初期化テスト
        import requests
        from urllib.parse import urljoin

        host = os.environ["LANGFUSE_HOST"].rstrip('/')

        # ヘルスチェックエンドポイントを試行
        health_url = urljoin(host, "/api/public/health")

        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info("Langfuse サービスが利用可能です")
                return True
            else:
                logger.warning(f"Langfuse ヘルスチェック失敗: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Langfuse サービスへの接続に失敗しました: {e}")
            return False

    except Exception as e:
        logger.warning(f"Langfuse 可用性チェック中にエラーが発生しました: {e}")
        return False


def create_safe_langfuse_handler():
    """
    安全なlangfuse handlerを作成する
    エラーが発生してもstream処理を中断しないようにする

    Returns:
        CallbackHandler or None: 正常に作成できた場合はCallbackHandler、エラーの場合はNone
    """
    try:
        # 事前にLangfuseサービスの可用性をチェック
        if not check_langfuse_availability():
            logger.warning("Langfuse サービスが利用できないため、callback を無効にします")
            return None

        return CallbackHandler(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )
    except Exception as e:
        logger.warning(f"Langfuse handlerの作成に失敗しました: {e}")
        return None


def get_safe_stream_config(model_name=None):
    """
    安全なstream設定を取得する
    langfuse handlerが利用できない場合は空の設定を返す

    Args:
        model_name: モデル名（メタデータ用）

    Returns:
        dict: stream設定
    """
    try:
        langfuse_handler = create_safe_langfuse_handler()
        if langfuse_handler is None:
            logger.info(f"Langfuse が利用できないため、{model_name} のストリーミングは callback なしで実行されます")
            return {}

        config = {"callbacks": [langfuse_handler]}
        if model_name:
            config["metadata"] = {"ls_model_name": model_name}
        logger.info(f"Langfuse callback が有効になりました: {model_name}")
        return config
    except Exception as e:
        logger.warning(f"Stream設定の作成に失敗しました: {e}")
        return {}


if platform.system() == 'Linux':
    oracledb.init_oracle_client(lib_dir=os.environ["ORACLE_CLIENT_LIB_DIR"])

# データベース接続プールを初期化（ブロッキングを避けるため接続数を増加）
pool = oracledb.create_pool(
    dsn=os.environ["ORACLE_23AI_CONNECTION_STRING"],
    min=5,
    max=20,
    increment=2,
    timeout=30,  # 接続タイムアウト30秒
    getmode=oracledb.POOL_GETMODE_WAIT  # 利用可能な接続を待機
)


def do_auth(username, password):
    dsn = os.environ["ORACLE_23AI_CONNECTION_STRING"]
    pattern = r"^([^/]+)/([^@]+)@"
    match = re.match(pattern, dsn)

    if match:
        if username.lower() == match.group(1).lower() and password == match.group(2):
            return True
    return False


def get_region():
    oci_config_path = find_dotenv("/root/.oci/config")
    region = get_key(oci_config_path, "region")
    return region


def check_database_pool_health():
    """
    データベース接続プールの健康状態をチェックする

    Returns:
        bool: プールが正常な場合True、問題がある場合False
    """
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM DUAL")
                result = cursor.fetchone()
                return result is not None
    except Exception as e:
        logger.error(f"データベース接続プールの健康チェックに失敗しました: {e}")
        return False


def generate_embedding_response(inputs: List[str]):
    config = oci.config.from_file('/root/.oci/config', "DEFAULT")
    region = get_region()
    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240))
    batch_size = 96
    all_embeddings = []

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]

        embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
        embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=os.environ["OCI_COHERE_EMBED_MODEL"]
        )
        embed_text_detail.inputs = batch
        embed_text_detail.truncate = "NONE"
        embed_text_detail.compartment_id = os.environ["OCI_COMPARTMENT_OCID"]

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)
                print(f"Processed batch {i // batch_size + 1} of {(len(inputs) - 1) // batch_size + 1}")
                all_embeddings.extend(embed_text_response.data.embeddings)
                break
            except Exception as e:
                print(f"Exception: {e}")
                retry_count += 1
                print(f"Error embedding text: {e}. Retrying ({retry_count}/{max_retries})...")
                time.sleep(10 * retry_count)
                if retry_count == max_retries:
                    gr.Warning("保存中にエラーが発生しました。しばらくしてから再度お試しください。")
                    all_embeddings = []
                    return all_embeddings

        time.sleep(1)

    return all_embeddings


def rerank_documents_response(input_text, inputs: List[str], rerank_model):
    all_document_ranks = []
    batch_size = 200

    if rerank_model in ["cohere/rerank-multilingual-v3.1", "cohere/rerank-english-v3.1"]:
        rerank_model = rerank_model.replace("/", ".")
        config = oci.config.from_file('/root/.oci/config', "DEFAULT")
        region = get_region()
        rerank_generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240))

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]

            rerank_text_detail = oci.generative_ai_inference.models.RerankTextDetails()
            rerank_text_detail.input = input_text
            rerank_text_detail.documents = batch
            rerank_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                serving_type="ON_DEMAND",
                model_id=rerank_model
            )
            rerank_text_detail.compartment_id = os.environ["OCI_COMPARTMENT_OCID"]
            rerank_response = rerank_generative_ai_inference_client.rerank_text(rerank_text_detail)
            print(f"Processed batch {i // batch_size + 1} of {(len(inputs) - 1) // batch_size + 1}")
            adjusted_results = []
            for rank in rerank_response.data.document_ranks:
                adjusted_result = {
                    "document": rank.document,
                    "index": i + rank.index,
                    "relevance_score": rank.relevance_score
                }
                adjusted_results.append(adjusted_result)
            all_document_ranks.extend(adjusted_results)
    else:
        print(f"{os.environ['COHERE_API_KEY']=}")
        cohere_reranker = rerank_model.split('/')[1]
        cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]

            rerank_response = cohere_client.rerank(
                query=input_text,
                documents=batch,
                top_n=len(batch),
                model=cohere_reranker
            )
            print(f"{rerank_response=}")
            adjusted_results = []
            for rank in rerank_response.results:
                adjusted_result = {
                    "document": rank.document,
                    "index": i + rank.index,
                    "relevance_score": rank.relevance_score
                }
                adjusted_results.append(adjusted_result)
            all_document_ranks.extend(adjusted_results)

    return all_document_ranks


def get_doc_list() -> List[Tuple[str, str]]:
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(f"""
SELECT
    json_value(cmetadata, '$.file_name') name,
    id
FROM
    {DEFAULT_COLLECTION_NAME}_collection
ORDER BY name """)
                return [(f"{row[0]}", row[1]) for row in cursor.fetchall()]
            except DatabaseError as de:
                return []


def refresh_doc_list():
    doc_list = get_doc_list()
    return (
        gr.Radio(choices=doc_list, value=None),
        gr.CheckboxGroup(choices=doc_list, value=[]),
        gr.CheckboxGroup(choices=doc_list, value=[])
    )


def get_server_path(doc_id: str) -> str:
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"""
SELECT json_value(cmetadata, '$.server_path') AS server_path
FROM {DEFAULT_COLLECTION_NAME}_collection
WHERE id = :doc_id """, doc_id=doc_id)
            return cursor.fetchone()[0]


def process_text_chunks(unstructured_chunks):
    chunks = []
    chunk_id = 1
    start_offset = 1

    for chunk in unstructured_chunks:
        # chunk_length = len(chunk.text)
        chunk_length = len(chunk)
        if chunk_length == 0:
            continue
        chunks.append({
            'CHUNK_ID': chunk_id,
            'CHUNK_OFFSET': start_offset,
            'CHUNK_LENGTH': chunk_length,
            # 'CHUNK_DATA': chunk.text
            'CHUNK_DATA': chunk
        })

        # IDとオフセットを更新
        chunk_id += 1
        start_offset += chunk_length

    return chunks


async def command_a_task(system_text, query_text, command_a_checkbox):
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
        print(f"{start_time=}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("cohere.command-a-03-2025")

        try:
            async for chunk in command_a.astream(messages, config=stream_config):
                yield chunk.content
        except Exception as e:
            logger.error(f"Command-A stream処理中にエラーが発生しました: {e}")
            # エラーが発生してもstream処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def xai_grok_3_task(system_text, query_text, xai_grok_3_checkbox):
    region = get_region()
    if xai_grok_3_checkbox:
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
        print(f"{start_time=}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("xai.grok-3")

        try:
            async for chunk in xai_grok_3.astream(messages, config=stream_config):
                yield chunk.content
        except Exception as e:
            logger.error(f"XAI Grok-3 ストリーム処理中にエラーが発生しました: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def llama_3_3_70b_task(system_text, query_text, llama_3_3_70b_checkbox):
    region = get_region()
    if llama_3_3_70b_checkbox:
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
        print(f"{start_time=}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("meta.llama-3.3-70b-instruct")

        try:
            async for chunk in llama_3_3_70b.astream(messages, config=stream_config):
                yield chunk.content
        except Exception as e:
            logger.error(f"Llama-3.3-70B ストリーム処理中にエラーが発生しました: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def llama_3_2_90b_vision_task(system_text, query_image, query_text, llama_3_2_90b_vision_checkbox):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    region = get_region()
    if llama_3_2_90b_vision_checkbox:
        llama_3_2_90b_vision = ChatOCIGenAI(
            model_id="meta.llama-3.2-90b-vision-instruct",
            provider="meta",
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600, "presence_penalty": 2,
                          "frequency_penalty": 2},
        )
        if query_image:
            base64_image = encode_image(query_image)
            human_message = HumanMessage(content=[
                {"type": "text", "text": query_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ], )
        else:
            human_message = HumanMessage(content=query_text)

        if system_text:
            messages = [
                SystemMessage(content=system_text),
                human_message,
            ]
        else:
            messages = [
                human_message,
            ]
        start_time = time.time()
        print(f"{start_time=}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("meta.llama-3.2-90b-vision-instruct")

        try:
            async for chunk in llama_3_2_90b_vision.astream(messages, config=stream_config):
                yield chunk.content
        except Exception as e:
            logger.error(f"Llama-3.2-90B-Vision ストリーム処理中にエラーが発生しました: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def llama_4_maverick_task(system_text, query_image, query_text, llama_4_maverick_checkbox):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    region = get_region()
    if llama_4_maverick_checkbox:
        llama_4_maverick = ChatOCIGenAI(
            model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
            provider="meta",
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
        )
        if query_image:
            base64_image = encode_image(query_image)
            human_message = HumanMessage(content=[
                {"type": "text", "text": query_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ], )
        else:
            human_message = HumanMessage(content=query_text)

        if system_text:
            messages = [
                SystemMessage(content=system_text),
                human_message,
            ]
        else:
            messages = [
                human_message,
            ]
        start_time = time.time()
        print(f"{start_time=}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("meta.llama-4-maverick-17b-128e-instruct-fp8")

        try:
            async for chunk in llama_4_maverick.astream(messages, config=stream_config):
                yield chunk.content
        except Exception as e:
            logger.error(f"Llama-4-Maverick ストリーム処理中にエラーが発生しました: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def llama_4_scout_task(system_text, query_image, query_text, llama_4_scout_checkbox):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    region = get_region()
    if llama_4_scout_checkbox:
        llama_4_scout = ChatOCIGenAI(
            model_id="meta.llama-4-scout-17b-16e-instruct",
            provider="meta",
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
        )
        if query_image:
            base64_image = encode_image(query_image)
            human_message = HumanMessage(content=[
                {"type": "text", "text": query_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ], )
        else:
            human_message = HumanMessage(content=query_text)

        if system_text:
            messages = [
                SystemMessage(content=system_text),
                human_message,
            ]
        else:
            messages = [
                human_message,
            ]
        start_time = time.time()
        print(f"{start_time=}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("meta.llama-4-scout-17b-16e-instruct")
        print(f"{stream_config=}")

        try:
            async for chunk in llama_4_scout.astream(messages, config=stream_config):
                yield chunk.content
        except Exception as e:
            logger.error(f"Llama-4-Scout ストリーム処理中にエラーが発生しました: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def openai_gpt4o_task(system_text, query_text, openai_gpt4o_checkbox):
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
        print(f"{start_time=}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("gpt-4o")

        try:
            async for chunk in openai_gpt4o.astream(messages, config=stream_config):
                yield chunk.content
        except Exception as e:
            logger.error(f"OpenAI GPT-4o ストリーム処理中にエラーが発生しました: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def openai_gpt4_task(system_text, query_text, openai_gpt4_checkbox):
    if openai_gpt4_checkbox:
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
        print(f"{start_time=}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("gpt-4")

        try:
            async for chunk in openai_gpt4.astream(messages, config=stream_config):
                yield chunk.content
        except Exception as e:
            logger.error(f"OpenAI GPT-4 ストリーム処理中にエラーが発生しました: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def azure_openai_gpt4o_task(system_text, query_text, azure_openai_gpt4o_checkbox):
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
        print(f"{start_time=}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("azure-gpt-4o")

        try:
            async for chunk in azure_openai_gpt4o.astream(messages, config=stream_config):
                yield chunk.content
        except Exception as e:
            logger.error(f"Azure OpenAI GPT-4o ストリーム処理中にエラーが発生しました: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def azure_openai_gpt4_task(system_text, query_text, azure_openai_gpt4_checkbox):
    if azure_openai_gpt4_checkbox:
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
        print(f"{start_time=}")

        # 安全なlangfuse設定を取得
        stream_config = get_safe_stream_config("azure-gpt-4")

        try:
            async for chunk in azure_openai_gpt4.astream(messages, config=stream_config):
                yield chunk.content
        except Exception as e:
            logger.error(f"Azure OpenAI GPT-4 ストリーム処理中にエラーが発生しました: {e}")
            # エラーが発生してもストリーム処理を継続するため、エラーメッセージをyield
            yield f"\n\nエラーが発生しました: {e}\n\n"
        end_time = time.time()
        print(f"{end_time=}")
        inference_time = end_time - start_time
        print(f"\n\n推論時間: {inference_time:.2f}秒")
        yield f"\n\n推論時間: {inference_time:.2f}秒"
        yield "TASK_DONE"
    else:
        yield "TASK_DONE"


async def chat(
        system_text,
        xai_grok_3_user_text,
        command_a_user_text,
        llama_4_maverick_user_image,
        llama_4_maverick_user_text,
        llama_4_scout_user_image,
        llama_4_scout_user_text,
        llama_3_3_70b_user_text,
        llama_3_2_90b_vision_user_image,
        llama_3_2_90b_vision_user_text,
        openai_gpt4o_user_text,
        openai_gpt4_user_text,
        azure_openai_gpt4o_user_text,
        azure_openai_gpt4_user_text,
        xai_grok_3_checkbox,
        command_a_checkbox,
        llama_4_maverick_checkbox,
        llama_4_scout_checkbox,
        llama_3_3_70b_checkbox,
        llama_3_2_90b_vision_checkbox,
        openai_gpt4o_gen_checkbox,
        openai_gpt4_gen_checkbox,
        azure_openai_gpt4o_gen_checkbox,
        azure_openai_gpt4_gen_checkbox
):
    xai_grok_3_gen = xai_grok_3_task(system_text, xai_grok_3_user_text, xai_grok_3_checkbox)
    command_a_gen = command_a_task(system_text, command_a_user_text, command_a_checkbox)
    llama_4_maverick_gen = llama_4_maverick_task(system_text, llama_4_maverick_user_image,
                                                 llama_4_maverick_user_text, llama_4_maverick_checkbox)
    llama_4_scout_gen = llama_4_scout_task(system_text, llama_4_scout_user_image,
                                           llama_4_scout_user_text, llama_4_scout_checkbox)
    llama_3_3_70b_gen = llama_3_3_70b_task(system_text, llama_3_3_70b_user_text, llama_3_3_70b_checkbox)
    llama_3_2_90b_vision_gen = llama_3_2_90b_vision_task(system_text, llama_3_2_90b_vision_user_image,
                                                         llama_3_2_90b_vision_user_text,
                                                         llama_3_2_90b_vision_checkbox)
    openai_gpt4o_gen = openai_gpt4o_task(system_text, openai_gpt4o_user_text, openai_gpt4o_gen_checkbox)
    openai_gpt4_gen = openai_gpt4_task(system_text, openai_gpt4_user_text, openai_gpt4_gen_checkbox)
    azure_openai_gpt4o_gen = azure_openai_gpt4o_task(system_text, azure_openai_gpt4o_user_text,
                                                     azure_openai_gpt4o_gen_checkbox)
    azure_openai_gpt4_gen = azure_openai_gpt4_task(system_text, azure_openai_gpt4_user_text,
                                                   azure_openai_gpt4_gen_checkbox)

    responses_status = ["", "", "", "", "", "", "", "", "", ""]
    while True:
        responses = ["", "", "", "", "", "", "", "", "", ""]
        generators = [xai_grok_3_gen, command_a_gen,
                      llama_4_maverick_gen, llama_4_scout_gen,
                      llama_3_3_70b_gen, llama_3_2_90b_vision_gen,
                      openai_gpt4o_gen, openai_gpt4_gen,
                      azure_openai_gpt4o_gen, azure_openai_gpt4_gen]

        for i, gen in enumerate(generators):
            try:
                response = await anext(gen)
                if response:
                    if response == "TASK_DONE":
                        responses_status[i] = response
                    else:
                        responses[i] = response
            except StopAsyncIteration:
                pass

        yield tuple(responses)

        if all(response_status == "TASK_DONE" for response_status in responses_status):
            print("All tasks completed with DONE")
            break


def set_chat_llm_answer(llm_answer_checkbox):
    xai_grok_3_answer_visible = False
    command_a_answer_visible = False
    llama_4_maverick_answer_visible = False
    llama_4_scout_answer_visible = False
    llama_3_3_70b_answer_visible = False
    llama_3_2_90b_vision_answer_visible = False
    openai_gpt4o_answer_visible = False
    openai_gpt4_answer_visible = False
    azure_openai_gpt4o_answer_visible = False
    azure_openai_gpt4_answer_visible = False
    if "xai/grok-3" in llm_answer_checkbox:
        xai_grok_3_answer_visible = True
    if "cohere/command-a" in llm_answer_checkbox:
        command_a_answer_visible = True
    if "meta/llama-4-maverick-17b-128e-instruct-fp8" in llm_answer_checkbox:
        llama_4_maverick_answer_visible = True
    if "meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox:
        llama_4_scout_answer_visible = True
    if "meta/llama-3-3-70b" in llm_answer_checkbox:
        llama_3_3_70b_answer_visible = True
    if "meta/llama-3-2-90b-vision" in llm_answer_checkbox:
        llama_3_2_90b_vision_answer_visible = True
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt4o_answer_visible = True
    if "openai/gpt-4" in llm_answer_checkbox:
        openai_gpt4_answer_visible = True
    if "azure_openai/gpt-4o" in llm_answer_checkbox:
        azure_openai_gpt4o_answer_visible = True
    if "azure_openai/gpt-4" in llm_answer_checkbox:
        azure_openai_gpt4_answer_visible = True
    return (
        gr.Accordion(visible=xai_grok_3_answer_visible),
        gr.Accordion(visible=command_a_answer_visible),
        gr.Accordion(visible=llama_4_maverick_answer_visible),
        gr.Accordion(visible=llama_4_scout_answer_visible),
        gr.Accordion(visible=llama_3_3_70b_answer_visible),
        gr.Accordion(visible=llama_3_2_90b_vision_answer_visible),
        gr.Accordion(visible=openai_gpt4o_answer_visible),
        gr.Accordion(visible=openai_gpt4_answer_visible),
        gr.Accordion(visible=azure_openai_gpt4o_answer_visible),
        gr.Accordion(visible=azure_openai_gpt4_answer_visible)
    )


def set_chat_llm_evaluation(llm_evaluation_checkbox):
    xai_grok_3_evaluation_visible = False
    command_a_evaluation_visible = False
    llama_4_maverick_evaluation_visible = False
    llama_4_scout_evaluation_visible = False
    llama_3_3_70b_evaluation_visible = False
    llama_3_2_90b_vision_evaluation_visible = False
    openai_gpt4o_evaluation_visible = False
    openai_gpt4_evaluation_visible = False
    azure_openai_gpt4o_evaluation_visible = False
    azure_openai_gpt4_evaluation_visible = False
    if llm_evaluation_checkbox:
        xai_grok_3_evaluation_visible = True
        command_a_evaluation_visible = True
        llama_4_maverick_evaluation_visible = True
        llama_4_scout_evaluation_visible = True
        llama_3_3_70b_evaluation_visible = True
        llama_3_2_90b_vision_evaluation_visible = True
        openai_gpt4o_evaluation_visible = True
        openai_gpt4_evaluation_visible = True
        azure_openai_gpt4o_evaluation_visible = True
        azure_openai_gpt4_evaluation_visible = True
    return (
        gr.Accordion(visible=xai_grok_3_evaluation_visible),
        gr.Accordion(visible=command_a_evaluation_visible),
        gr.Accordion(visible=llama_4_maverick_evaluation_visible),
        gr.Accordion(visible=llama_4_scout_evaluation_visible),
        gr.Accordion(visible=llama_3_3_70b_evaluation_visible),
        gr.Accordion(visible=llama_3_2_90b_vision_evaluation_visible),
        gr.Accordion(visible=openai_gpt4o_evaluation_visible),
        gr.Accordion(visible=openai_gpt4_evaluation_visible),
        gr.Accordion(visible=azure_openai_gpt4o_evaluation_visible),
        gr.Accordion(visible=azure_openai_gpt4_evaluation_visible)
    )


def set_image_answer_visibility(llm_answer_checkbox, use_image):
    """
    画像回答の可視性を制御する関数
    選択されたLLMモデルと「画像を使って回答」の状態に基づいて、
    対象のモデルの画像回答Accordionの可視性を決定する
    """
    llama_4_maverick_image_visible = False
    llama_4_scout_image_visible = False
    llama_3_2_90b_vision_image_visible = False
    openai_gpt4o_image_visible = False
    azure_openai_gpt4o_image_visible = False

    # 画像を使って回答がオンで、かつ対応するモデルが選択されている場合のみ表示
    if use_image:
        if "meta/llama-4-maverick-17b-128e-instruct-fp8" in llm_answer_checkbox:
            llama_4_maverick_image_visible = True
        if "meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox:
            llama_4_scout_image_visible = True
        if "meta/llama-3-2-90b-vision" in llm_answer_checkbox:
            llama_3_2_90b_vision_image_visible = True
        if "openai/gpt-4o" in llm_answer_checkbox:
            openai_gpt4o_image_visible = True
        if "azure_openai/gpt-4o" in llm_answer_checkbox:
            azure_openai_gpt4o_image_visible = True

    return (
        gr.Accordion(visible=llama_4_maverick_image_visible),
        gr.Accordion(visible=llama_4_scout_image_visible),
        gr.Accordion(visible=llama_3_2_90b_vision_image_visible),
        gr.Accordion(visible=openai_gpt4o_image_visible),
        gr.Accordion(visible=azure_openai_gpt4o_image_visible)
    )


async def chat_stream(system_text, query_image, query_text, llm_answer_checkbox):
    has_error = False
    if not llm_answer_checkbox or len(llm_answer_checkbox) == 0:
        has_error = True
        gr.Warning("LLM モデルを選択してください")
    if not query_text:
        has_error = True
        gr.Warning("ユーザー・メッセージを入力してください")

    if has_error:
        yield (
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            ""
        )
        return
    xai_grok_3_user_text = query_text
    command_a_user_text = query_text
    llama_4_maverick_user_image = query_image
    llama_4_maverick_user_text = query_text
    llama_4_scout_user_image = query_image
    llama_4_scout_user_text = query_text
    llama_3_3_70b_user_text = query_text
    llama_3_2_90b_vision_user_image = query_image
    llama_3_2_90b_vision_user_text = query_text
    openai_gpt4o_user_text = query_text
    openai_gpt4_user_text = query_text
    azure_openai_gpt4o_user_text = query_text
    azure_openai_gpt4_user_text = query_text
    xai_grok_3_checkbox = False
    command_a_checkbox = False
    llama_4_maverick_checkbox = False
    llama_4_scout_checkbox = False
    llama_3_3_70b_checkbox = False
    llama_3_2_90b_vision_checkbox = False
    openai_gpt4o_checkbox = False
    openai_gpt4_checkbox = False
    azure_openai_gpt4o_checkbox = False
    azure_openai_gpt4_checkbox = False
    if "xai/grok-3" in llm_answer_checkbox:
        xai_grok_3_checkbox = True
    if "cohere/command-a" in llm_answer_checkbox:
        command_a_checkbox = True
    if "meta/llama-4-maverick-17b-128e-instruct-fp8" in llm_answer_checkbox:
        llama_4_maverick_checkbox = True
    if "meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox:
        llama_4_scout_checkbox = True
    if "meta/llama-3-3-70b" in llm_answer_checkbox:
        llama_3_3_70b_checkbox = True
    if "meta/llama-3-2-90b-vision" in llm_answer_checkbox:
        llama_3_2_90b_vision_checkbox = True
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt4o_checkbox = True
    if "openai/gpt-4" in llm_answer_checkbox:
        openai_gpt4_checkbox = True
    if "azure_openai/gpt-4o" in llm_answer_checkbox:
        azure_openai_gpt4o_checkbox = True
    if "azure_openai/gpt-4" in llm_answer_checkbox:
        azure_openai_gpt4_checkbox = True
    # ChatOCIGenAI
    xai_grok_3_response = ""
    command_a_response = ""
    llama_4_maverick_response = ""
    llama_4_scout_response = ""
    llama_3_3_70b_response = ""
    llama_3_2_90b_vision_response = ""
    openai_gpt4o_response = ""
    openai_gpt4_response = ""
    azure_openai_gpt4o_response = ""
    azure_openai_gpt4_response = ""
    async for xai_grok_3, command_a, llama_4_maverick, llama_4_scout, llama_3_3_70b, llama_3_2_90b_vision, gpt4o, gpt4, azure_gpt4o, azure_gpt4 in chat(
            system_text,
            xai_grok_3_user_text,
            command_a_user_text,
            llama_4_maverick_user_image,
            llama_4_maverick_user_text,
            llama_4_scout_user_image,
            llama_4_scout_user_text,
            llama_3_3_70b_user_text,
            llama_3_2_90b_vision_user_image,
            llama_3_2_90b_vision_user_text,
            openai_gpt4o_user_text,
            openai_gpt4_user_text,
            azure_openai_gpt4o_user_text,
            azure_openai_gpt4_user_text,
            xai_grok_3_checkbox,
            command_a_checkbox,
            llama_4_maverick_checkbox,
            llama_4_scout_checkbox,
            llama_3_3_70b_checkbox,
            llama_3_2_90b_vision_checkbox,
            openai_gpt4o_checkbox,
            openai_gpt4_checkbox,
            azure_openai_gpt4o_checkbox,
            azure_openai_gpt4_checkbox
    ):
        xai_grok_3_response += xai_grok_3
        command_a_response += command_a
        llama_4_maverick_response += llama_4_maverick
        llama_4_scout_response += llama_4_scout
        llama_3_3_70b_response += llama_3_3_70b
        llama_3_2_90b_vision_response += llama_3_2_90b_vision
        openai_gpt4o_response += gpt4o
        openai_gpt4_response += gpt4
        azure_openai_gpt4o_response += azure_gpt4o
        azure_openai_gpt4_response += azure_gpt4
        yield (
            gr.Markdown(value=xai_grok_3_response),
            gr.Markdown(value=command_a_response),
            gr.Markdown(value=llama_4_maverick_response),
            gr.Markdown(value=llama_4_scout_response),
            gr.Markdown(value=llama_3_3_70b_response),
            gr.Markdown(value=llama_3_2_90b_vision_response),
            gr.Markdown(value=openai_gpt4o_response),
            gr.Markdown(value=openai_gpt4_response),
            gr.Markdown(value=azure_openai_gpt4o_response),
            gr.Markdown(value=azure_openai_gpt4_response)
        )


def reset_all_llm_messages():
    """
    すべてのLLMメッセージをリセットする
    """
    return (
        gr.Markdown(value=""),  # tab_chat_document_xai_grok_3_answer_text
        gr.Markdown(value=""),  # tab_chat_document_command_a_answer_text
        gr.Markdown(value=""),  # tab_chat_document_llama_4_maverick_answer_text
        gr.Markdown(value=""),  # tab_chat_document_llama_4_scout_answer_text
        gr.Markdown(value=""),  # tab_chat_document_llama_3_3_70b_answer_text
        gr.Markdown(value=""),  # tab_chat_document_llama_3_2_90b_vision_answer_text
        gr.Markdown(value=""),  # tab_chat_document_openai_gpt4o_answer_text
        gr.Markdown(value=""),  # tab_chat_document_openai_gpt4_answer_text
        gr.Markdown(value=""),  # tab_chat_document_azure_openai_gpt4o_answer_text
        gr.Markdown(value=""),  # tab_chat_document_azure_openai_gpt4_answer_text
    )


def reset_image_answers():
    """
    画像回答をリセットする
    """
    return (
        gr.Markdown(value=""),  # tab_chat_document_llama_4_maverick_image_answer_text
        gr.Markdown(value=""),  # tab_chat_document_llama_4_scout_image_answer_text
        gr.Markdown(value=""),  # tab_chat_document_llama_3_2_90b_vision_image_answer_text
        gr.Markdown(value=""),  # tab_chat_document_openai_gpt4o_image_answer_text
        gr.Markdown(value=""),  # tab_chat_document_azure_openai_gpt4o_image_answer_text
    )


def reset_llm_evaluations():
    """
    LLM評価をリセットする
    """
    return (
        gr.Markdown(value=""),  # tab_chat_document_xai_grok_3_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_command_a_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_llama_4_maverick_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_llama_4_scout_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_llama_3_3_70b_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_llama_3_2_90b_vision_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_openai_gpt4o_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_openai_gpt4_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_azure_openai_gpt4o_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_azure_openai_gpt4_evaluation_text
    )


def reset_eval_by_human_result():
    return (
        gr.Radio(value="good"),
        gr.Textbox(value=""),
        gr.Radio(value="good"),
        gr.Textbox(value=""),
        gr.Radio(value="good"),
        gr.Textbox(value=""),
        gr.Radio(value="good"),
        gr.Textbox(value=""),
        gr.Radio(value="good"),
        gr.Textbox(value=""),
        gr.Radio(value="good"),
        gr.Textbox(value=""),
        gr.Radio(value="good"),
        gr.Textbox(value=""),
        gr.Radio(value="good"),
        gr.Textbox(value=""),
        gr.Radio(value="good"),
        gr.Textbox(value=""),
        gr.Radio(value="good"),
        gr.Textbox(value=""),
    )


def eval_by_human(
        query_id,
        llm_name,
        human_evaluation_result,
        user_comment,
):
    print("eval_by_human() start...")
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            update_sql = """
                         UPDATE RAG_QA_FEEDBACK
                         SET human_evaluation_result = :1,
                                user_comment = :2
                         WHERE query_id = :3 AND llm_name = :4 \
                         """
            cursor.execute(
                update_sql,
                [
                    human_evaluation_result,
                    user_comment,
                    query_id,
                    llm_name
                ]
            )

            conn.commit()

    return (
        gr.Radio(),
        gr.Textbox(value=user_comment)
    )


def create_oci_cred(user_ocid, tenancy_ocid, fingerprint, private_key_file, region):
    def process_private_key(private_key_file_path):
        with open(private_key_file_path, 'r') as file:
            lines = file.readlines()

        processed_key = ''.join(line.strip() for line in lines if not line.startswith('--'))
        return processed_key

    has_error = False
    if not user_ocid:
        has_error = True
        gr.Warning("User OCIDを入力してください")
    if not tenancy_ocid:
        has_error = True
        gr.Warning("Tenancy OCIDを入力してください")
    if not fingerprint:
        has_error = True
        gr.Warning("Fingerprintを入力してください")
    if not private_key_file:
        has_error = True
        gr.Warning("Private Keyを入力してください")
    if not region:
        has_error = True
        gr.Warning("Regionを選択してください")

    if has_error:
        return gr.Accordion(), gr.Textbox()

    user_ocid = user_ocid.strip()
    tenancy_ocid = tenancy_ocid.strip()
    fingerprint = fingerprint.strip()
    region = region.strip()

    # set up OCI config
    if not os.path.exists("/root/.oci"):
        os.makedirs("/root/.oci")
    if not os.path.exists("/root/.oci/config"):
        shutil.copy("./.oci/config", "/root/.oci/config")
    oci_config_path = find_dotenv("/root/.oci/config")
    key_file_path = '/root/.oci/oci_api_key.pem'
    set_key(oci_config_path, "user", user_ocid, quote_mode="never")
    set_key(oci_config_path, "tenancy", tenancy_ocid, quote_mode="never")
    set_key(oci_config_path, "region", region, quote_mode="never")
    set_key(oci_config_path, "fingerprint", fingerprint, quote_mode="never")
    set_key(oci_config_path, "key_file", key_file_path, quote_mode="never")
    shutil.copy(private_key_file.name, key_file_path)
    load_dotenv(oci_config_path)

    # set up OCI Credential on database
    private_key = process_private_key(private_key_file.name)

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            try:
                # Define the PL/SQL statement
                append_acl_sql = """
BEGIN
  DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
    host => '*',
    ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                       principal_name => 'admin',
                       principal_type => xs_acl.ptype_db));
END;
                """

                # Execute the PL/SQL statement
                cursor.execute(append_acl_sql)
            except DatabaseError as de:
                print(f"DatabaseError={de}")

            try:
                drop_oci_cred_sql = "BEGIN dbms_vector.drop_credential('OCI_CRED'); END;"
                cursor.execute(drop_oci_cred_sql)
            except DatabaseError as de:
                print(f"DatabaseError={de}")

            oci_cred = {
                'user_ocid': user_ocid,
                'tenancy_ocid': tenancy_ocid,
                'compartment_ocid': os.environ["OCI_COMPARTMENT_OCID"],
                'private_key': private_key.strip(),
                'fingerprint': fingerprint
            }

            create_oci_cred_sql = """
BEGIN
   dbms_vector.create_credential(
       credential_name => 'OCI_CRED',
       params => json(:json_params)
   );
END; """

            cursor.execute(create_oci_cred_sql, json_params=json.dumps(oci_cred))
            conn.commit()

    create_oci_cred_sql = f"""
-- Append Host ACE
BEGIN
  DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
    host => '*',
    ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                       principal_name => 'admin',
                       principal_type => xs_acl.ptype_db));
END;

-- Drop Existing OCI Credential
BEGIN dbms_vector.drop_credential('OCI_CRED'); END;

-- Create New OCI Credential
BEGIN
    dbms_vector.create_credential(
        credential_name => 'OCI_CRED',
        params => json('{json.dumps(oci_cred)}')
    );
END;
"""
    gr.Info("OCI API Keyの設定が完了しました")
    return gr.Accordion(), gr.Textbox(value=create_oci_cred_sql.strip())


def create_cohere_cred(cohere_cred_api_key):
    has_error = False
    if not cohere_cred_api_key:
        has_error = True
        gr.Warning("Cohere API Keyを入力してください")
    if has_error:
        return gr.Textbox()
    cohere_cred_api_key = cohere_cred_api_key.strip()
    env_path = find_dotenv()
    os.environ["COHERE_API_KEY"] = cohere_cred_api_key
    set_key(env_path, "COHERE_API_KEY", cohere_cred_api_key, quote_mode="never")
    load_dotenv(env_path)
    gr.Info("Cohere API Keyの設定が完了しました")
    return gr.Textbox(value=cohere_cred_api_key)


def create_openai_cred(openai_cred_base_url, openai_cred_api_key):
    has_error = False
    if not openai_cred_base_url:
        has_error = True
        gr.Warning("OpenAI Base URLを入力してください")
    if not openai_cred_api_key:
        has_error = True
        gr.Warning("OpenAI API Keyを入力してください")
    if has_error:
        return gr.Textbox(), gr.Textbox()

    openai_cred_base_url = openai_cred_base_url.strip()
    openai_cred_api_key = openai_cred_api_key.strip()
    env_path = find_dotenv()
    os.environ["OPENAI_BASE_URL"] = openai_cred_base_url
    os.environ["OPENAI_API_KEY"] = openai_cred_api_key
    set_key(env_path, "OPENAI_BASE_URL", openai_cred_base_url, quote_mode="never")
    set_key(env_path, "OPENAI_API_KEY", openai_cred_api_key, quote_mode="never")
    load_dotenv(find_dotenv())
    gr.Info("OpenAI API Keyの設定が完了しました")
    return gr.Textbox(value=openai_cred_base_url), gr.Textbox(value=openai_cred_api_key)


def create_azure_openai_cred(
        azure_openai_cred_api_key,
        azure_openai_cred_endpoint_gpt_4o,
        azure_openai_cred_endpoint_gpt_4,
):
    has_error = False
    if not azure_openai_cred_api_key:
        has_error = True
        gr.Warning("Azure OpenAI API Keyを入力してください")
    if not azure_openai_cred_endpoint_gpt_4o:
        has_error = True
        gr.Warning("Azure OpenAI GPT-4O Endpointを入力してください")
    if 'api-version=' not in azure_openai_cred_endpoint_gpt_4o:
        has_error = True
        gr.Warning("Azure OpenAI GPT-4O Endpointにはapi-versionを入力してください")
    if azure_openai_cred_endpoint_gpt_4 and 'api-version=' not in azure_openai_cred_endpoint_gpt_4:
        has_error = True
        gr.Warning("Azure OpenAI GPT-4 Endpointにはapi-versionを入力してください")
    if has_error:
        return gr.Textbox(), gr.Textbox(), gr.Textbox()

    azure_openai_cred_api_key = azure_openai_cred_api_key.strip()
    azure_openai_cred_endpoint_gpt_4o = azure_openai_cred_endpoint_gpt_4o.strip() if azure_openai_cred_endpoint_gpt_4o else ""
    azure_openai_cred_api_version_gpt_4o = re.search(r"api-version=([^&]+)", azure_openai_cred_endpoint_gpt_4o).group(
        1).strip()
    # azure_openai_cred_api_version_gpt_4o = azure_openai_cred_api_version_gpt_4o.strip() if azure_openai_cred_api_version_gpt_4o else ""
    azure_openai_cred_endpoint_gpt_4 = azure_openai_cred_endpoint_gpt_4.strip() if azure_openai_cred_endpoint_gpt_4 else ""
    azure_openai_cred_api_version_gpt_4 = ""
    if azure_openai_cred_endpoint_gpt_4:
        azure_openai_cred_api_version_gpt_4 = re.search(r"api-version=([^&]+)", azure_openai_cred_endpoint_gpt_4).group(
            1).strip()

    env_path = find_dotenv()

    os.environ["AZURE_OPENAI_API_KEY"] = azure_openai_cred_api_key
    os.environ["AZURE_OPENAI_ENDPOINT_GPT_4O"] = azure_openai_cred_endpoint_gpt_4o
    os.environ["AZURE_OPENAI_API_VERSION_GPT_4O"] = azure_openai_cred_api_version_gpt_4o
    os.environ["AZURE_OPENAI_ENDPOINT_GPT_4"] = azure_openai_cred_endpoint_gpt_4
    os.environ["AZURE_OPENAI_API_VERSION_GPT_4"] = azure_openai_cred_api_version_gpt_4

    set_key(env_path, "AZURE_OPENAI_API_KEY", azure_openai_cred_api_key, quote_mode="never")
    set_key(env_path, "AZURE_OPENAI_ENDPOINT_GPT_4O", azure_openai_cred_endpoint_gpt_4o, quote_mode="never")
    set_key(env_path, "AZURE_OPENAI_API_VERSION_GPT_4O", azure_openai_cred_api_version_gpt_4o, quote_mode="never")
    set_key(env_path, "AZURE_OPENAI_ENDPOINT_GPT_4", azure_openai_cred_endpoint_gpt_4, quote_mode="never")
    set_key(env_path, "AZURE_OPENAI_API_VERSION_GPT_4", azure_openai_cred_api_version_gpt_4, quote_mode="never")

    load_dotenv(find_dotenv())

    gr.Info("Azure OpenAI API Keyの設定が完了しました")
    return gr.Textbox(value=azure_openai_cred_api_key), \
        gr.Textbox(value=azure_openai_cred_endpoint_gpt_4o), \
        gr.Textbox(value=azure_openai_cred_endpoint_gpt_4)


def create_langfuse_cred(langfuse_cred_secret_key, langfuse_cred_public_key, langfuse_cred_host):
    has_error = False
    if not langfuse_cred_secret_key:
        has_error = True
        gr.Warning("Langfuse Secret Keyを入力してください")
    if not langfuse_cred_public_key:
        has_error = True
        gr.Warning("Langfuse Public Keyを入力してください")
    if not langfuse_cred_host:
        has_error = True
        gr.Warning("Langfuse Hostを入力してください")
    if has_error:
        return gr.Textbox(), gr.Textbox(), gr.Textbox()

    langfuse_cred_secret_key = langfuse_cred_secret_key.strip()
    langfuse_cred_public_key = langfuse_cred_public_key.strip()
    langfuse_cred_host = langfuse_cred_host.strip()
    env_path = find_dotenv()
    os.environ["LANGFUSE_SECRET_KEY"] = langfuse_cred_secret_key
    os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_cred_public_key
    os.environ["LANGFUSE_HOST"] = langfuse_cred_host
    set_key(env_path, "LANGFUSE_SECRET_KEY", langfuse_cred_secret_key, quote_mode="never")
    set_key(env_path, "LANGFUSE_PUBLIC_KEY", langfuse_cred_public_key, quote_mode="never")
    set_key(env_path, "LANGFUSE_HOST", langfuse_cred_host, quote_mode="never")
    load_dotenv(env_path)
    gr.Info("LangFuseの設定が完了しました")
    return (
        gr.Textbox(value=langfuse_cred_secret_key),
        gr.Textbox(value=langfuse_cred_public_key),
        gr.Textbox(value=langfuse_cred_host)
    )


def create_table():
    # Drop the preference if it exists
    check_preference_sql = """
                           SELECT PRE_NAME
                           FROM CTX_PREFERENCES
                           WHERE PRE_NAME = 'WORLD_LEXER'
                             AND PRE_OWNER = USER \
                           """

    drop_preference_plsql = """
                            -- Drop Preference
                            BEGIN
  CTX_DDL.DROP_PREFERENCE
                            ('world_lexer');
                            END; \
                            """

    create_preference_plsql = """
                              -- Create Preference
                              BEGIN
  CTX_DDL.CREATE_PREFERENCE
                              ('world_lexer','WORLD_LEXER');
                              END; \
                              """

    # Drop the index if it exists
    check_index_sql = f"""
SELECT INDEX_NAME FROM USER_INDEXES WHERE INDEX_NAME = '{DEFAULT_COLLECTION_NAME.upper()}_EMBED_DATA_IDX'
"""

    check_image_index_sql = f"""
SELECT INDEX_NAME FROM USER_INDEXES WHERE INDEX_NAME = '{DEFAULT_COLLECTION_NAME.upper()}_IMAGE_EMBED_DATA_IDX'
"""

    drop_index_sql = f"""
-- Drop Index
DROP INDEX {DEFAULT_COLLECTION_NAME.upper()}_EMBED_DATA_IDX
"""

    drop_image_index_sql = f"""
-- Drop Image Index
DROP INDEX {DEFAULT_COLLECTION_NAME.upper()}_IMAGE_EMBED_DATA_IDX
"""

    create_index_sql = f"""
-- Create Index
-- CREATE INDEX {DEFAULT_COLLECTION_NAME}_embed_data_idx ON {DEFAULT_COLLECTION_NAME}_embedding(embed_data) INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('LEXER world_lexer sync (on commit)')
CREATE INDEX {DEFAULT_COLLECTION_NAME}_embed_data_idx ON {DEFAULT_COLLECTION_NAME}_embedding(embed_data) INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('LEXER world_lexer sync (every "freq=minutely; interval=1")')
"""

    create_image_index_sql = f"""
-- Create Image Index
CREATE INDEX {DEFAULT_COLLECTION_NAME}_image_embed_data_idx ON {DEFAULT_COLLECTION_NAME}_image_embedding(embed_data) INDEXTYPE IS CTXSYS.CONTEXT PARAMETERS ('LEXER world_lexer sync (every "freq=minutely; interval=1")')
"""

    output_sql_text = f"""
-- Create Collection Table
CREATE TABLE IF NOT EXISTS {DEFAULT_COLLECTION_NAME}_collection (
    id VARCHAR2(200),
    data BLOB,
    cmetadata CLOB
);
"""

    # Get embedding function and dimension first
    region = get_region()
    embed = OCIGenAIEmbeddings(
        model_id=os.environ["OCI_COHERE_EMBED_MODEL"],
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
        compartment_id=os.environ["OCI_COMPARTMENT_OCID"]
    )

    # Get embedding dimension by creating a test embedding
    test_embedding = embed.embed_query("test")
    embedding_dim = len(test_embedding)

    output_sql_text += f"""
-- Create Embedding Table
CREATE TABLE IF NOT EXISTS {DEFAULT_COLLECTION_NAME}_embedding (
    doc_id VARCHAR2(200),
    embed_id NUMBER,
    embed_data VARCHAR2(4000),
    embed_vector VECTOR({embedding_dim}, FLOAT32),
    cmetadata CLOB
);
"""

    output_sql_text += f"""
-- Create Image Table
CREATE TABLE IF NOT EXISTS {DEFAULT_COLLECTION_NAME}_image (
    doc_id VARCHAR2(200),
    img_id NUMBER,
    text_data CLOB,
    vlm_data CLOB,
    base64_data CLOB
);
"""

    output_sql_text += f"""
-- Create Image Embedding Table
CREATE TABLE IF NOT EXISTS {DEFAULT_COLLECTION_NAME}_image_embedding (
    doc_id VARCHAR2(200),
    embed_id NUMBER,
    embed_data VARCHAR2(4000),
    embed_vector VECTOR({embedding_dim}, FLOAT32),
    cmetadata CLOB,
    img_id NUMBER
);
"""

    drop_rag_qa_result_sql = """DROP TABLE IF EXISTS RAG_QA_RESULT"""

    create_rag_qa_result_sql = """CREATE TABLE IF NOT EXISTS RAG_QA_RESULT
    (
        id
        NUMBER
        GENERATED
        ALWAYS AS
        IDENTITY
        PRIMARY
        KEY,
        query_id
        VARCHAR2
                                  (
        100
                                  ),
        query VARCHAR2
                                  (
                                      4000
                                  ),
        standard_answer VARCHAR2
                                  (
                                      30000
                                  ),
        sql CLOB,
        created_date TIMESTAMP DEFAULT TO_TIMESTAMP
                                  (
                                      TO_CHAR
                                  (
                                      SYSTIMESTAMP,
                                      'YYYY-MM-DD HH24:MI:SS'
                                  ), 'YYYY-MM-DD HH24:MI:SS')
        )"""

    drop_rag_qa_feedback_sql = """DROP TABLE IF EXISTS RAG_QA_FEEDBACK"""

    create_rag_qa_feedback_sql = """CREATE TABLE IF NOT EXISTS RAG_QA_FEEDBACK
    (
        id
        NUMBER
        GENERATED
        ALWAYS AS
        IDENTITY
        PRIMARY
        KEY,
        query_id
        VARCHAR2
                                    (
        100
                                    ),
        llm_name VARCHAR2
                                    (
                                        100
                                    ),
        llm_answer CLOB,
        ragas_evaluation_result CLOB,
        human_evaluation_result VARCHAR2
                                    (
                                        20
                                    ),
        user_comment VARCHAR2
                                    (
                                        30000
                                    ),
        created_date TIMESTAMP DEFAULT TO_TIMESTAMP
                                    (
                                        TO_CHAR
                                    (
                                        SYSTIMESTAMP,
                                        'YYYY-MM-DD HH24:MI:SS'
                                    ), 'YYYY-MM-DD HH24:MI:SS')
        )"""

    output_sql_text += "\n" + create_preference_plsql.strip() + "\n"
    output_sql_text += "\n" + drop_rag_qa_result_sql.strip() + ";"
    output_sql_text += "\n" + drop_rag_qa_feedback_sql.strip() + ";"
    output_sql_text += f"\n-- Drop Indexes\nDROP INDEX IF EXISTS {DEFAULT_COLLECTION_NAME}_embed_data_idx;"
    output_sql_text += f"\nDROP INDEX IF EXISTS {DEFAULT_COLLECTION_NAME}_image_embed_data_idx;"
    output_sql_text += "\n" + create_index_sql.strip() + ";"
    output_sql_text += "\n" + create_image_index_sql.strip() + ";"
    output_sql_text += "\n" + create_rag_qa_result_sql.strip() + ";"
    output_sql_text += "\n" + create_rag_qa_feedback_sql.strip() + ";"

    # Drop and Create table SQLs for image and image_embedding tables
    drop_image_table_sql = f"DROP TABLE {DEFAULT_COLLECTION_NAME}_image PURGE"
    drop_image_embedding_table_sql = f"DROP TABLE {DEFAULT_COLLECTION_NAME}_image_embedding PURGE"

    create_image_table_sql = f"""CREATE TABLE {DEFAULT_COLLECTION_NAME}_image (
    doc_id VARCHAR2(200),
    img_id NUMBER,
    text_data CLOB,
    vlm_data CLOB,
    base64_data CLOB
)"""

    create_image_embedding_table_sql = f"""CREATE TABLE {DEFAULT_COLLECTION_NAME}_image_embedding (
    doc_id VARCHAR2(200),
    embed_id NUMBER,
    embed_data VARCHAR2(4000),
    embed_vector VECTOR({embedding_dim}, FLOAT32),
    cmetadata CLOB,
    img_id NUMBER
)"""

    # Add drop and create statements to output SQL text
    output_sql_text += f"\n-- Drop Image Tables\n{drop_image_table_sql};"
    output_sql_text += f"\n{drop_image_embedding_table_sql};"
    output_sql_text += f"\n-- Create Image Tables\n{create_image_table_sql};"
    output_sql_text += f"\n{create_image_embedding_table_sql};"

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.execute(check_preference_sql)
            if cursor.fetchone():
                cursor.execute(drop_preference_plsql)
            else:
                print("Preference 'WORLD_LEXER' does not exist.")
            cursor.execute(create_preference_plsql)

            # Drop and recreate image tables with existence check
            # Check and drop image table
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {DEFAULT_COLLECTION_NAME}_image")
                # Table exists, drop it
                cursor.execute(drop_image_table_sql)
                print(f"テーブル {DEFAULT_COLLECTION_NAME}_image を削除しました")
            except DatabaseError as e:
                if e.args[0].code == 942:  # Table or view does not exist
                    print(f"テーブル {DEFAULT_COLLECTION_NAME}_image は存在しません")
                else:
                    print(f"テーブル {DEFAULT_COLLECTION_NAME}_image の削除エラー: {e}")

            # Check and drop image_embedding table
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {DEFAULT_COLLECTION_NAME}_image_embedding")
                # Table exists, drop it
                cursor.execute(drop_image_embedding_table_sql)
                print(f"テーブル {DEFAULT_COLLECTION_NAME}_image_embedding を削除しました")
            except DatabaseError as e:
                if e.args[0].code == 942:  # Table or view does not exist
                    print(f"テーブル {DEFAULT_COLLECTION_NAME}_image_embedding は存在しません")
                else:
                    print(f"テーブル {DEFAULT_COLLECTION_NAME}_image_embedding の削除エラー: {e}")

            # Create image table
            try:
                cursor.execute(create_image_table_sql)
                print(f"テーブル {DEFAULT_COLLECTION_NAME}_image を作成しました")
            except DatabaseError as e:
                print(f"テーブル {DEFAULT_COLLECTION_NAME}_image の作成エラー: {e}")

            # Create image_embedding table
            try:
                cursor.execute(create_image_embedding_table_sql)
                print(f"テーブル {DEFAULT_COLLECTION_NAME}_image_embedding を作成しました")
            except DatabaseError as e:
                print(f"テーブル {DEFAULT_COLLECTION_NAME}_image_embedding の作成エラー: {e}")

            # Drop and create indexes with existence check
            try:
                cursor.execute(
                    f"SELECT COUNT(*) FROM USER_INDEXES WHERE INDEX_NAME = '{DEFAULT_COLLECTION_NAME.upper()}_EMBED_DATA_IDX'")
                if cursor.fetchone()[0] > 0:
                    cursor.execute(f"DROP INDEX {DEFAULT_COLLECTION_NAME}_embed_data_idx")
                    print(f"インデックス {DEFAULT_COLLECTION_NAME}_embed_data_idx を削除しました")
            except DatabaseError as e:
                print(f"インデックス {DEFAULT_COLLECTION_NAME}_embed_data_idx の削除エラー: {e}")

            try:
                cursor.execute(
                    f"SELECT COUNT(*) FROM USER_INDEXES WHERE INDEX_NAME = '{DEFAULT_COLLECTION_NAME.upper()}_IMAGE_EMBED_DATA_IDX'")
                if cursor.fetchone()[0] > 0:
                    cursor.execute(f"DROP INDEX {DEFAULT_COLLECTION_NAME}_image_embed_data_idx")
                    print(f"インデックス {DEFAULT_COLLECTION_NAME}_image_embed_data_idx を削除しました")
            except DatabaseError as e:
                print(f"インデックス {DEFAULT_COLLECTION_NAME}_image_embed_data_idx の削除エラー: {e}")

            # Create indexes
            try:
                cursor.execute(create_index_sql)
                print(f"インデックス {DEFAULT_COLLECTION_NAME}_embed_data_idx を作成しました")
            except DatabaseError as e:
                print(f"インデックス {DEFAULT_COLLECTION_NAME}_embed_data_idx の作成エラー: {e}")

            try:
                cursor.execute(create_image_index_sql)
                print(f"インデックス {DEFAULT_COLLECTION_NAME}_image_embed_data_idx を作成しました")
            except DatabaseError as e:
                print(f"インデックス {DEFAULT_COLLECTION_NAME}_image_embed_data_idx の作成エラー: {e}")

            try:
                cursor.execute(drop_rag_qa_result_sql)
            except DatabaseError as e:
                print(f"{e}")

            try:
                cursor.execute(create_rag_qa_result_sql)
            except DatabaseError as e:
                print(f"{e}")

            try:
                cursor.execute(drop_rag_qa_feedback_sql)
            except DatabaseError as e:
                print(f"{e}")

            try:
                cursor.execute(create_rag_qa_feedback_sql)
            except DatabaseError as e:
                print(f"{e}")

            conn.commit()

    gr.Info("テーブルの作成が完了しました")
    return gr.Accordion(), gr.Textbox(value=output_sql_text.strip())


def test_oci_cred(test_query_text):
    test_query_vector = ""
    region = get_region()
    embed_genai_params = {
        "provider": "ocigenai",
        "credential_name": "OCI_CRED",
        "url": f"https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText",
        "model": "cohere.embed-multilingual-v3.0"
    }

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            plsql = """
DECLARE
    l_embed_genai_params CLOB := :embed_genai_params;
    l_result SYS_REFCURSOR;
BEGIN
    OPEN l_result FOR
        SELECT et.*
        FROM dbms_vector_chain.utl_to_embeddings(:text_to_embed, JSON(l_embed_genai_params)) et;
    :result := l_result;
END; """

            result_cursor = cursor.var(oracledb.CURSOR)

            cursor.execute(plsql,
                           embed_genai_params=json.dumps(embed_genai_params),
                           text_to_embed=test_query_text,
                           result=result_cursor)

            # Fetch the results from the ref cursor
            with result_cursor.getvalue() as ref_cursor:
                result_rows = ref_cursor.fetchall()
                for row in result_rows:
                    if isinstance(row, tuple):
                        if isinstance(row[0], oracledb.LOB):
                            lob_content = row[0].read()
                            lob_json = json.loads(lob_content)
                            test_query_vector += str(lob_json["embed_vector"]) + "\n"

    return gr.Textbox(value=test_query_vector)


def convert_excel_to_text_document(file_path):
    has_error = False
    if not file_path:
        has_error = True
        gr.Warning("ファイルを選択してください")
    if has_error:
        return gr.File(value=None)

    output_file_path = file_path.name + '.txt'
    df = pd.read_excel(file_path.name)
    data = df.to_dict('records')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for row in data:
            # Process each row to convert Timestamps to strings
            processed_row = {}
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    processed_row[key] = str(value)
                else:
                    processed_row[key] = value
            json_line = json.dumps(processed_row, ensure_ascii=False)
            f.write(json_line + ' <FIXED_DELIMITER>\n')
    return (
        gr.File(),
        gr.File(value=output_file_path)
    )


def convert_to_markdown_document(file_path, use_llm, llm_prompt):
    has_error = False
    if not file_path:
        has_error = True
        gr.Warning("ファイルを選択してください")
    if has_error:
        return gr.File(value=None)

    output_file_path = file_path.name + '.md'
    md = MarkItDown()

    file_extension = os.path.splitext(file_path.name)[-1].lower()
    if file_extension in ['.jpg', '.jpeg', '.png', '.ppt', '.pptx'] and use_llm:
        region = get_region()
        model_id = "meta.llama-3.2-90b-vision-instruct"
        if region == "us-chicago-1":
            model_id = "meta.llama-4-scout-17b-16e-instruct"
        client = ChatOCIGenAI(
            model_id="meta.llama-3.2-90b-vision-instruct",
            provider="meta",
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
        )
        md = MarkItDown(llm_client=client, llm_model="meta.llama-3.2-90b-vision-instruct")
        result = md.convert(
            file_path.name,
            llm_prompt=llm_prompt,
        )
        print(f"{result.text_content=}")
    else:
        result = md.convert(
            file_path.name,
        )
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(result.text_content)
    return (
        gr.File(),
        gr.File(value=output_file_path)
    )


def load_document(file_path, server_directory, document_metadata):
    print("in load_document() start...")
    has_error = False
    if not file_path:
        has_error = True
        gr.Warning("ファイルを選択してください")
    if document_metadata:
        document_metadata = document_metadata.strip()
        if "=" not in document_metadata or '"' in document_metadata or "'" in document_metadata or '\\' in document_metadata:
            has_error = True
            gr.Warning("メタデータの形式が正しくありません。key1=value1,key2=value2,... の形式で入力してください。")
        else:
            metadatas = document_metadata.split(",")
            for metadata in metadatas:
                if "=" not in metadata:
                    has_error = True
                    gr.Warning(
                        "メタデータの形式が正しくありません。key1=value1,key2=value2,... の形式で入力してください。")
                    break
    if has_error:
        return gr.Textbox(value=""), gr.Textbox(value=""), gr.Textbox(value=""), gr.Textbox(value="")

    if not os.path.exists(server_directory):
        os.makedirs(server_directory)
    doc_id = generate_unique_id("doc_")
    file_name = os.path.basename(file_path.name)
    file_extension = os.path.splitext(file_name)
    if isinstance(file_extension, tuple):
        file_extension = file_extension[1]
    server_path = os.path.join(server_directory, f"{doc_id}_{file_name}")
    shutil.copy(file_path.name, server_path)

    collection_cmeta = {}
    if file_extension == ".md":
        loader = TextLoader(server_path)
        documents = loader.load()
        original_contents = "".join(doc.page_content for doc in documents)
        pages_count = len(documents)
    else:
        # https://docs.unstructured.io/open-source/core-functionality/overview
        pages_count = 1
        elements = partition(filename=server_path, strategy='fast',
                             languages=["jpn", "eng", "chi_sim"],
                             extract_image_block_types=["Table"],
                             extract_image_block_to_payload=False,
                             skip_infer_table_types=["pdf", "jpg", "png", "heic", "doc", "docx"])
        for el in elements:
            print(f"{el=}")
        original_contents = " \n".join(el.text.replace('\x0b', '\n') for el in elements)
        print(f"{original_contents=}")

    collection_cmeta['file_name'] = file_name
    collection_cmeta['source'] = server_path
    collection_cmeta['server_path'] = server_path
    if document_metadata:
        metadatas = document_metadata.split(",")
        for metadata in metadatas:
            key, value = metadata.split("=")
            collection_cmeta[key] = value

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.setinputsizes(**{"data": oracledb.BLOB})
            load_document_sql = f"""
 -- (Only for Reference) Insert to table {DEFAULT_COLLECTION_NAME}_collection
 INSERT INTO {DEFAULT_COLLECTION_NAME}_collection(id, data, cmetadata)
 VALUES (:id, to_blob(:data), :cmetadata) """
            output_sql_text = load_document_sql.replace(":id", "'" + str(doc_id) + "'")
            output_sql_text = output_sql_text.replace(":data", "'...'")
            output_sql_text = output_sql_text.replace(":cmetadata", "'" + json.dumps(collection_cmeta) + "'") + ";"
            cursor.execute(load_document_sql, {
                'id': doc_id,
                'data': original_contents,
                'cmetadata': json.dumps(collection_cmeta)
            })
            conn.commit()

    return (
        gr.Textbox(value=output_sql_text.strip()),
        gr.Textbox(value=doc_id),
        gr.Textbox(value=str(pages_count)),
        gr.Textbox(value=original_contents)
    )


# def split_document_by_oracle(doc_id, chunks_by, chunks_max_size,
#                              chunks_overlap_size,
#                              chunks_split_by, chunks_split_by_custom,
#                              chunks_language, chunks_normalize,
#                              chunks_normalize_options):
#     # print(f"{chunks_normalize_options=}")
#     with pool.acquire() as conn:
#         with conn.cursor() as cursor:
#             parameter_str = '{"by": "' + chunks_by + '","max": "' + str(
#                 chunks_max_size) + '", "overlap": "' + str(
#                 int(int(
#                     chunks_max_size) * chunks_overlap_size / 100)) + '", "split": "' + chunks_split_by + '", '
#             if chunks_split_by == "CUSTOM":
#                 parameter_str += '"custom_list": [' + chunks_split_by_custom + '], '
#             parameter_str += '"language": "' + chunks_language + '", '
#             if chunks_normalize == "NONE" or chunks_normalize == "ALL":
#                 parameter_str += '"normalize": "' + chunks_normalize + '"}'
#             else:
#                 parameter_str += '"normalize": "' + chunks_normalize + '", "norm_options": [' + ", ".join(
#                     ['"' + s + '"' for s in chunks_normalize_options]) + ']}'
#
#             # print(f"{parameter_str}")
#             select_chunks_sql = f"""
# -- Select chunks
# SELECT
#     json_value(ct.column_value, '$.chunk_id')     AS chunk_id,
#     json_value(ct.column_value, '$.chunk_offset') AS chunk_offset,
#     json_value(ct.column_value, '$.chunk_length') AS chunk_length,
#     json_value(ct.column_value, '$.chunk_data')   AS chunk_data
# FROM
#     {DEFAULT_COLLECTION_NAME}_collection dt,
#     dbms_vector_chain.utl_to_chunks(
#         dbms_vector_chain.utl_to_text(dt.data),
#         JSON(
#             :parameter_str
#         )
#     ) ct
#     CROSS JOIN
#         JSON_TABLE ( ct.column_value, '$[*]'
#             COLUMNS (
#                 column_value VARCHAR2 ( 4000 ) PATH '$'
#             )
#         )
# WHERE
#     dt.id = :doc_id """
#             # cursor.setinputsizes(oracledb.DB_TYPE_VECTOR)
#             utl_to_chunks_sql_output = "\n" + select_chunks_sql.replace(':parameter_str',
#                                                                         "'" + parameter_str + "'").replace(
#                 ':doc_id', "'" + str(doc_id) + "'") + ";"
#             # print(f"{utl_to_chunks_sql_output=}")
#             cursor.execute(select_chunks_sql,
#                            [parameter_str, doc_id])
#             chunks = []
#             for row in cursor:
#                 chunks.append({'CHUNK_ID': row[0],
#                                'CHUNK_OFFSET': row[1], 'CHUNK_LENGTH': row[2], 'CHUNK_DATA': row[3]})
#
#             conn.commit()
#
#     chunks_dataframe = pd.DataFrame(chunks)
#     return (
#         gr.Textbox(utl_to_chunks_sql_output.strip()),
#         gr.Textbox(value=str(len(chunks_dataframe))),
#         gr.Dataframe(value=chunks_dataframe)
#     )


def reset_document_chunks_result_dataframe():
    return (
        gr.Dataframe(value=None, row_count=(1, "fixed"))
    )


def process_image_blocks(doc_id, doc_data, chunk_size=None, chunk_overlap=None):
    """
    画像ブロックを処理して{DEFAULT_COLLECTION_NAME}_imageテーブルに保存し、
    text_splitterを使用してデータを分割し{DEFAULT_COLLECTION_NAME}_image_embeddingテーブルに保存する

    Args:
        doc_id: ドキュメントID
        doc_data: ドキュメントデータ
        chunk_size: チャンクサイズ
        chunk_overlap: チャンクオーバーラップ
    """
    # 画像ブロックのパターンを検索
    image_blocks = re.findall(r'<!-- image_begin -->(.*?)<!-- image_end -->', doc_data, re.DOTALL)

    if not image_blocks:
        return

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            # 既存の画像データを削除
            delete_image_sql = f"""
DELETE FROM {DEFAULT_COLLECTION_NAME}_image WHERE doc_id = :doc_id
"""
            cursor.execute(delete_image_sql, [doc_id])

            # 既存の画像embedding データを削除
            delete_image_embedding_sql = f"""
DELETE FROM {DEFAULT_COLLECTION_NAME}_image_embedding WHERE doc_id = :doc_id
"""
            cursor.execute(delete_image_embedding_sql, [doc_id])

            # 画像データを挿入するSQL
            insert_image_sql = f"""
INSERT INTO {DEFAULT_COLLECTION_NAME}_image (
    doc_id,
    img_id,
    text_data,
    vlm_data,
    base64_data
) VALUES (:doc_id, :img_id, :text_data, :vlm_data, :base64_data)
"""

            img_id = 1
            for image_block in image_blocks:
                # OCRコンテンツを抽出
                text_data_match = re.search(r'<!-- image_ocr_content_begin -->(.*?)<!-- image_ocr_content_end -->',
                                            image_block, re.DOTALL)
                text_data = text_data_match.group(1).strip() if text_data_match else ""

                # VLM説明を抽出
                vlm_data_match = re.search(
                    r'<!-- image_vlm_description_begin -->(.*?)<!-- image_vlm_description_end -->', image_block,
                    re.DOTALL)
                vlm_data = vlm_data_match.group(1).strip() if vlm_data_match else ""

                # Base64データを抽出
                base64_data_match = re.search(r'<!-- image_base64_begin -->(.*?)<!-- image_base64_end -->', image_block,
                                              re.DOTALL)
                base64_data = base64_data_match.group(1).strip() if base64_data_match else ""

                # base64_dataから前後のHTMLコメント記号を除去
                if base64_data:
                    # 前後の <!-- と --> を除去
                    base64_data = re.sub(r'^<!--\s*', '', base64_data)
                    base64_data = re.sub(r'\s*-->$', '', base64_data)
                    base64_data = base64_data.strip()

                # データベースに挿入
                cursor.execute(insert_image_sql, {
                    'doc_id': doc_id,
                    'img_id': img_id,
                    'text_data': text_data,
                    'vlm_data': vlm_data,
                    'base64_data': base64_data
                })

                print(f"画像データを保存しました: doc_id={doc_id}, img_id={img_id}")
                img_id += 1

            conn.commit()
            print(f"合計 {len(image_blocks)} 個の画像ブロックを処理しました")

            # 画像データのsplit処理を実行
            _process_image_data_splitting(doc_id, cursor, chunk_size, chunk_overlap)
            conn.commit()


def _process_image_data_splitting(doc_id, cursor, chunk_size=None, chunk_overlap=None):
    """
    {DEFAULT_COLLECTION_NAME}_imageテーブルのtext_dataとvlm_dataを個別にtext_splitterで分割し、
    {DEFAULT_COLLECTION_NAME}_image_embeddingテーブルに保存する

    Args:
        doc_id: ドキュメントID
        cursor: データベースカーソル
        chunk_size: チャンクサイズ
        chunk_overlap: チャンクオーバーラップ
    """
    # 画像データを取得
    select_image_sql = f"""
SELECT img_id, text_data, vlm_data FROM {DEFAULT_COLLECTION_NAME}_image
WHERE doc_id = :doc_id ORDER BY img_id
"""
    cursor.execute(select_image_sql, [doc_id])
    image_records = cursor.fetchall()

    if not image_records:
        print(f"画像データが見つかりません: doc_id={doc_id}")
        return

    # text_splitterを初期化
    if chunk_size is not None and chunk_overlap is not None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        # デフォルト設定を使用
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,  # デフォルトのchunk_size
            chunk_overlap=25  # デフォルトのoverlap（約10%）
        )

    # 画像embedding データを挿入するSQL
    insert_embedding_sql = f"""
INSERT INTO {DEFAULT_COLLECTION_NAME}_image_embedding (
    doc_id,
    embed_id,
    embed_data,
    embed_vector,
    cmetadata,
    img_id
) VALUES (:doc_id, :embed_id, :embed_data, :embed_vector, :cmetadata, :img_id)
"""

    # text_dataとvlm_dataで独立したembed_idを使用
    text_embed_id = 1
    vlm_embed_id = 1
    total_chunks = 0

    for img_id, text_data, vlm_data in image_records:
        # LOBオブジェクトを文字列に変換
        text_data_str = text_data.read() if text_data else ""
        vlm_data_str = vlm_data.read() if vlm_data else ""

        # text_dataを個別に処理
        if text_data_str and text_data_str.strip():
            text_chunks = text_splitter.split_text(text_data_str)

            if text_chunks:
                # text_dataのchunkに対してembeddingを生成して保存
                text_chunk_texts = [chunk for chunk in text_chunks if chunk.strip()]
                if text_chunk_texts:
                    text_embed_vectors = generate_embedding_response(text_chunk_texts)

                    for i, (chunk_text, embed_vector) in enumerate(zip(text_chunk_texts, text_embed_vectors)):
                        # text_data用のメタデータを作成
                        cmetadata = json.dumps({
                            "img_id": img_id,
                            "chunk_index": i,
                            "source": "image_processing",
                            "data_type": "text_data",
                            "original_img_id": img_id
                        }, ensure_ascii=False)

                        cursor.setinputsizes(embed_vector=oracledb.DB_TYPE_VECTOR)
                        cursor.execute(insert_embedding_sql, {
                            'doc_id': doc_id,
                            'embed_id': text_embed_id,
                            'embed_data': chunk_text,
                            'embed_vector': embed_vector,
                            'cmetadata': cmetadata,
                            'img_id': img_id
                        })

                        print(
                            f"画像text_data embeddingデータを保存しました: doc_id={doc_id}, img_id={img_id}, embed_id={text_embed_id}")
                        text_embed_id += 1
                        total_chunks += 1

        # vlm_dataを個別に処理
        if vlm_data_str and vlm_data_str.strip():
            vlm_chunks = text_splitter.split_text(vlm_data_str)

            if vlm_chunks:
                # vlm_dataのchunkに対してembeddingを生成して保存
                vlm_chunk_texts = [chunk for chunk in vlm_chunks if chunk.strip()]
                if vlm_chunk_texts:
                    vlm_embed_vectors = generate_embedding_response(vlm_chunk_texts)

                    for i, (chunk_text, embed_vector) in enumerate(zip(vlm_chunk_texts, vlm_embed_vectors)):
                        # vlm_data用のメタデータを作成
                        cmetadata = json.dumps({
                            "img_id": img_id,
                            "chunk_index": i,
                            "source": "image_processing",
                            "data_type": "vlm_data",
                            "original_img_id": img_id
                        }, ensure_ascii=False)

                        cursor.setinputsizes(embed_vector=oracledb.DB_TYPE_VECTOR)
                        cursor.execute(insert_embedding_sql, {
                            'doc_id': doc_id,
                            'embed_id': vlm_embed_id,
                            'embed_data': chunk_text,
                            'embed_vector': embed_vector,
                            'cmetadata': cmetadata,
                            'img_id': img_id
                        })

                        print(
                            f"画像vlm_data embeddingデータを保存しました: doc_id={doc_id}, img_id={img_id}, embed_id={vlm_embed_id}")
                        vlm_embed_id += 1
                        total_chunks += 1

        # 両方のデータが空の場合の警告
        if (not text_data_str or not text_data_str.strip()) and (not vlm_data_str or not vlm_data_str.strip()):
            print(f"画像 {img_id} にtext_dataとvlm_dataの両方が空です")

    print(f"画像データの分割処理が完了しました: 合計 {total_chunks} 個のchunkを生成")


def reset_document_chunks_result_detail():
    return (
        gr.Textbox(value=""),
        gr.Textbox(value="")
    )


def split_document_by_unstructured(doc_id, chunks_by, chunks_max_size,
                                   chunks_overlap_size,
                                   chunks_split_by, chunks_split_by_custom,
                                   chunks_language, chunks_normalize,
                                   chunks_normalize_options):
    has_error = False
    if not doc_id:
        has_error = True
        gr.Warning("ドキュメントを選択してください")
    if has_error:
        return (
            gr.Textbox(value=""),
            gr.Textbox(value=""),
            gr.Dataframe(value=None, row_count=(1, "fixed"))
        )
    # print(f"{chunks_normalize_options=}")
    output_sql = ""
    server_path = get_server_path(doc_id)

    chunks_overlap = int(float(chunks_max_size) * (float(chunks_overlap_size) / 100))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunks_max_size - chunks_overlap,
        chunk_overlap=chunks_overlap
    )

    # Check if it's a .md file and get content from database instead of reading the file
    doc_data = ""
    if server_path.lower().endswith('.md'):
        loader = TextLoader(server_path)
        documents = loader.load()
        doc_data = "\n".join(doc.page_content for doc in documents)

        # Check if doc_data contains image blocks and extract OCR content if needed
        if re.search(r'<!-- image_begin -->.*?<!-- image_end -->', doc_data, re.DOTALL):
            # Extract all OCR content blocks
            text_contexts = re.findall(r'<!-- image_ocr_content_begin -->(.*?)<!-- image_ocr_content_end -->', doc_data,
                                       re.DOTALL)
            doc_data = "\n".join(ocr.strip() for ocr in text_contexts if ocr.strip())
    else:
        # If we can't get data from database, fall back to reading the file
        elements = partition(filename=server_path, strategy='fast',
                             languages=["jpn", "eng", "chi_sim"],
                             extract_image_block_types=["Table"],
                             extract_image_block_to_payload=False,
                             skip_infer_table_types=["pdf", "jpg", "png", "heic", "doc", "docx"])
        # Use claude to get table data
        # page_table_documents = parser_pdf(server_path)
        page_table_documents = {}
        # elements = partition(filename=server_path,
        #                      strategy='hi_res',
        #                      languages=["jpn", "eng", "chi_sim"]
        #                      )
        # for issue: https://github.com/Unstructured-IO/unstructured/issues/3396
        elements = partition(filename=server_path, strategy='fast',
                             languages=["jpn", "eng", "chi_sim"],
                             extract_image_block_types=["Table"],
                             extract_image_block_to_payload=False,
                             # skip_infer_table_types=["pdf", "ppt", "pptx", "doc", "docx", "xls", "xlsx"])
                             skip_infer_table_types=["pdf", "jpg", "png", "heic", "doc", "docx"])
        prev_page_number = 0
        table_idx = 1
        for el in elements:
            # print(f"{el.category=}")
            # print(f"{el.text=}")
            # print(f"{el.metadata.page_number=}")
            page_number = el.metadata.page_number
            if prev_page_number != page_number:
                prev_page_number = page_number
                table_idx = 1
            # print(f"{type(el.category)=}")
            if el.category == "Table":
                table_id = f"page_{page_number}_table_{table_idx}"
                print(f"{table_id=}")
                print(f"{page_table_documents=}")
                if page_table_documents:
                    page_table_document = get_dict_value(page_table_documents, table_id)
                    print(f"{page_table_document=}")
                    if page_table_document:
                        page_content = get_dict_value(page_table_document, "page_content")
                        table_to_markdown = get_dict_value(page_table_document, "table_to_markdown")
                        if page_content and table_to_markdown:
                            print(f"Before: {el.text=}")
                            el.text = page_content + "\n" + table_to_markdown + "\n"
                            print(f"After: {el.text=}")
                table_idx += 1
            for element in elements:
                # Convert element.text to string to avoid TypeError with LOB objects
                element.text = str(element.text).replace('\x0b', '\n')
            doc_data = " \n".join([str(element.text) for element in elements])

    unstructured_chunks = text_splitter.split_text(doc_data)

    chunks = process_text_chunks(unstructured_chunks)
    chunks_dataframe = pd.DataFrame(chunks)

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            delete_sql = f"""
-- Delete chunks
DELETE FROM {DEFAULT_COLLECTION_NAME}_embedding WHERE doc_id = :doc_id """
            cursor.execute(delete_sql,
                           [doc_id])
            output_sql += delete_sql.replace(':doc_id', "'" + str(doc_id) + "'").lstrip() + ";"

            save_chunks_sql = f"""
-- (Only for Reference) Insert chunks
INSERT INTO {DEFAULT_COLLECTION_NAME}_embedding (
doc_id,
embed_id,
embed_data
)
VALUES (:doc_id, :embed_id, :embed_data) """
            output_sql += "\n" + save_chunks_sql.replace(':doc_id', "'" + str(doc_id) + "'"
                                                         ).replace(':embed_id', "'...'"
                                                                   ).replace(':embed_data', "'...'"
                                                                             ) + ";"
            print(f"{output_sql=}")
            # バッチ挿入用のデータを準備
            data_to_insert = [(doc_id, chunk['CHUNK_ID'], chunk['CHUNK_DATA']) for chunk in chunks]

            # バッチ挿入を実行
            cursor.executemany(save_chunks_sql, data_to_insert)
            conn.commit()

    return (
        gr.Textbox(value=output_sql),
        gr.Textbox(value=str(len(chunks_dataframe))),
        gr.Dataframe(value=chunks_dataframe, row_count=(len(chunks_dataframe), "fixed"))
    )


def on_select_split_document_chunks_result(evt: gr.SelectData, df: pd.DataFrame):
    print("on_select_split_document_chunks_result() start...")
    selected_index = evt.index[0]  # 選択された行のインデックスを取得
    selected_row = df.iloc[selected_index]  # 選択された行のすべてのデータを取得
    return selected_row['CHUNK_ID'], \
        selected_row['CHUNK_DATA']


def update_document_chunks_result_detail(doc_id, df: pd.DataFrame, chunk_id, chunk_data):
    print("in update_document_chunks_result_detail() start...")
    has_error = False
    if not doc_id:
        has_error = True
        gr.Warning("ドキュメントを選択してください")
    if not chunk_data or chunk_data.strip() == "":
        has_error = True
        gr.Warning("CHUNK_DATAを入力してください")
    if has_error:
        return (
            gr.Dataframe(),
            gr.Textbox(),
            gr.Textbox(),
        )

    chunk_data = chunk_data.strip()

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            update_sql = f"""
UPDATE {DEFAULT_COLLECTION_NAME}_embedding
SET embed_data = :embed_data, embed_vector = :embed_vector
WHERE doc_id = :doc_id and embed_id = :embed_id
"""
            embed_vector = generate_embedding_response([chunk_data])[0]
            cursor.setinputsizes(embed_vector=oracledb.DB_TYPE_VECTOR)
            cursor.execute(update_sql,
                           {'doc_id': doc_id, 'embed_id': chunk_id, 'embed_data': chunk_data,
                            'embed_vector': embed_vector})
            conn.commit()

    print(f"{chunk_id=}")
    print(f"{chunk_data=}")

    updated_df = df.copy()
    mask = updated_df['CHUNK_ID'] == int(chunk_id)
    updated_df.loc[mask, 'CHUNK_DATA'] = chunk_data
    updated_df.loc[mask, 'CHUNK_LENGTH'] = len(chunk_data)
    # print(f"{mask.sum()} rows updated")
    print(f"{updated_df=}")

    return (
        gr.Dataframe(value=updated_df),
        gr.Textbox(),
        gr.Textbox(value=chunk_data),
    )


def embed_save_document_by_unstructured(doc_id, chunks_by, chunks_max_size,
                                        chunks_overlap_size,
                                        chunks_split_by, chunks_split_by_custom,
                                        chunks_language, chunks_normalize,
                                        chunks_normalize_options):
    has_error = False
    if not doc_id:
        has_error = True
        gr.Warning("ドキュメントを選択してください")
    if has_error:
        return (
            gr.Textbox(value=""),
            gr.Textbox(value=""),
            gr.Dataframe(value=None, row_count=(1, "fixed"))
        )

    # 画像ブロックを処理して{DEFAULT_COLLECTION_NAME}_imageテーブルに保存
    server_path = get_server_path(doc_id)
    chunks_overlap = int(float(chunks_max_size) * (float(chunks_overlap_size) / 100))

    # Check if it's a .md file and get content to process image blocks
    if server_path.lower().endswith('.md'):
        loader = TextLoader(server_path)
        documents = loader.load()
        doc_data = "\n".join(doc.page_content for doc in documents)

        # Check if doc_data contains image blocks and process them
        if re.search(r'<!-- image_begin -->.*?<!-- image_end -->', doc_data, re.DOTALL):
            process_image_blocks(doc_id, doc_data,
                                 chunk_size=chunks_max_size - chunks_overlap,
                                 chunk_overlap=chunks_overlap)

    output_sql = ""
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            select_sql = f"""
SELECT doc_id, embed_id, embed_data FROM {DEFAULT_COLLECTION_NAME}_embedding  WHERE doc_id = :doc_id
"""
            cursor.execute(select_sql, doc_id=doc_id)
            records = cursor.fetchall()
            embed_datas = [record[2] for record in records]
            embed_vectors = generate_embedding_response(embed_datas)
            update_sql = f"""
UPDATE {DEFAULT_COLLECTION_NAME}_embedding
SET embed_vector = :embed_vector
WHERE doc_id = :doc_id and embed_id = :embed_id
"""

            # update_sql = f"""
            # -- Update chunks
            # UPDATE {DEFAULT_COLLECTION_NAME}_embedding
            # SET embed_vector = dbms_vector.utl_to_embedding(embed_data, json('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}}'))
            # WHERE doc_id = :doc_id """
            # cursor.execute(update_sql,
            #                [doc_id])
            cursor.setinputsizes(embed_vector=oracledb.DB_TYPE_VECTOR)
            output_sql += update_sql.replace(':doc_id', "'" + str(doc_id) + "'"
                                             ).replace(':embed_id', "'" + str('...') + "'"
                                                       ).replace(':embed_vector', "'" + str('...') + "'").strip() + ";"
            print(f"{output_sql=}")
            cursor.executemany(update_sql,
                               [{'doc_id': record[0], 'embed_id': record[1], 'embed_vector': embed_vector}
                                for record, embed_vector in zip(records, embed_vectors)])
            conn.commit()
    return (
        gr.Textbox(output_sql),
        gr.Textbox(),
        gr.Dataframe()
    )


def generate_query(query_text, generate_query_radio):
    has_error = False
    if not query_text:
        has_error = True
        gr.Warning("クエリを入力してください")
    if has_error:
        return gr.Textbox(value=""), gr.Textbox(value=""), gr.Textbox(value="")

    generate_query1 = ""
    generate_query2 = ""
    generate_query3 = ""

    if generate_query_radio == "None":
        return gr.Textbox(value=generate_query1), gr.Textbox(value=generate_query2), gr.Textbox(value=generate_query3)

    region = get_region()
    if region == "us-chicago-1":
        chat_llm = ChatOCIGenAI(
            model_id="xai.grok-3",
            provider="xai",
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 600},
        )
    else:
        chat_llm = ChatOCIGenAI(
            model_id="cohere.command-a-03-2025",
            provider="cohere",
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 600},
        )

    # RAG-Fusion
    if generate_query_radio == "Sub-Query":
        # v1
        # sub_query_prompt = ChatPromptTemplate.from_messages([
        #     ("system",
        #      """
        #      You are an advanced assistant that specializes in breaking down complex, multifaceted input queries into more manageable sub-queries.
        #      This approach allows for each aspect of the query to be explored in depth, facilitating a comprehensive and nuanced response.
        #      Your task is to dissect the given query into its constituent elements and generate targeted sub-queries that can each be researched or answered individually,
        #      ensuring that the final response holistically addresses all components of the original query.
        #      """),
        #     ("user",
        #      "Decompose the following query into targeted sub-queries that can be individually explored: {original_query} \n OUTPUT (2 queries): )")
        # ])
        # v2
        sub_query_prompt = ChatPromptTemplate.from_messages([
            ("system", get_sub_query_prompt()),
            ("user", get_query_generation_prompt("Sub-Query", "{original_query}"))
        ])

        generate_sub_queries_chain = (
                sub_query_prompt | chat_llm | StrOutputParser() | (lambda x: x.split("\n"))
        )
        sub_queries = generate_sub_queries_chain.invoke({"original_query": query_text})
        print(f"{sub_queries=}")

        if isinstance(sub_queries, list):
            generate_query1 = re.sub(r'^1\. ', '', sub_queries[0])
            generate_query2 = re.sub(r'^2\. ', '', sub_queries[1])
            generate_query3 = re.sub(r'^3\. ', '', sub_queries[2])
    elif generate_query_radio == "RAG-Fusion":
        # v1
        # rag_fusion_prompt = ChatPromptTemplate.from_messages([
        #     ("system",
        #      "You are a helpful assistant that generates multiple similary search queries based on a single input query."),
        #     ("user", "Generate multiple search queries related to: {original_query} \n OUTPUT (2 queries):")
        # ])
        # v2
        rag_fusion_prompt = ChatPromptTemplate.from_messages([
            ("system", get_rag_fusion_prompt()),
            ("user", get_query_generation_prompt("RAG-Fusion", "{original_query}"))
        ])

        generate_rag_fusion_queries_chain = (
                rag_fusion_prompt | chat_llm | StrOutputParser() | (lambda x: x.split("\n"))
        )
        rag_fusion_queries = generate_rag_fusion_queries_chain.invoke({"original_query": query_text})
        print(f"{rag_fusion_queries=}")

        if isinstance(rag_fusion_queries, list):
            generate_query1 = re.sub(r'^1\. ', '', rag_fusion_queries[0])
            generate_query2 = re.sub(r'^2\. ', '', rag_fusion_queries[1])
            generate_query3 = re.sub(r'^3\. ', '', rag_fusion_queries[2])
    elif generate_query_radio == "HyDE":
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", get_hyde_prompt()),
            ("user", get_query_generation_prompt("HyDE", "{original_query}"))
        ])

        generate_hyde_answers_chain = (
                hyde_prompt | chat_llm | StrOutputParser() | (lambda x: x.split("\n"))
        )
        hyde_answers = generate_hyde_answers_chain.invoke({"original_query": query_text})
        print(f"{hyde_answers=}")

        if isinstance(hyde_answers, list):
            generate_query1 = re.sub(r'^1\. ', '', hyde_answers[0])
            generate_query2 = re.sub(r'^2\. ', '', hyde_answers[1])
            generate_query3 = re.sub(r'^3\. ', '', hyde_answers[2])
    elif generate_query_radio == "Step-Back-Prompting":
        step_back_prompt = ChatPromptTemplate.from_messages([
            ("system", get_step_back_prompt()),
            ("user", get_query_generation_prompt("Step-Back-Prompting", "{original_query}"))
        ])

        generate_step_back_queries_chain = (
                step_back_prompt | chat_llm | StrOutputParser() | (lambda x: x.split("\n"))
        )
        step_back_queries = generate_step_back_queries_chain.invoke({"original_query": query_text})
        print(f"{step_back_queries=}")

        if isinstance(step_back_queries, list):
            generate_query1 = re.sub(r'^1\. ', '', step_back_queries[0])
            generate_query2 = re.sub(r'^2\. ', '', step_back_queries[1])
            generate_query3 = re.sub(r'^3\. ', '', step_back_queries[2])
    elif generate_query_radio == "Customized-Multi-Step-Query":
        region = get_region()
        select_multi_step_query_sql = f"""
                SELECT json_value(dc.cmetadata, '$.file_name') name, de.embed_id embed_id, de.embed_data embed_data, de.doc_id doc_id
                FROM {DEFAULT_COLLECTION_NAME}_embedding de, {DEFAULT_COLLECTION_NAME}_collection dc
                WHERE de.doc_id = dc.id
                ORDER BY vector_distance(de.embed_vector , (
                        SELECT to_vector(et.embed_vector) embed_vector
                        FROM
                            dbms_vector_chain.utl_to_embeddings(:query_text, JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}}')) t,
                            JSON_TABLE ( t.column_value, '$[*]'
                                    COLUMNS (
                                        embed_id NUMBER PATH '$.embed_id',
                                        embed_data VARCHAR2 ( 4000 ) PATH '$.embed_data',
                                        embed_vector CLOB PATH '$.embed_vector'
                                    )
                                )
                            et), COSINE)
            """
        select_multi_step_query_sql += "FETCH FIRST 3 ROWS ONLY"
        # Prepare parameters for SQL execution
        multi_step_query_params = {
            "query_text": query_text
        }

        # For debugging: Print the final SQL command.
        # Assuming complete_sql is your SQL string with placeholders like :extend_around_chunk_size, :doc_ids, etc.
        query_sql_output = select_multi_step_query_sql

        # Manually replace placeholders with parameter values for debugging
        for key, value in multi_step_query_params.items():
            placeholder = f":{key}"
            # For the purpose of display, ensure the value is properly quoted if it's a string
            display_value = f"'{value}'" if isinstance(value, str) else str(value)
            query_sql_output = query_sql_output.replace(placeholder, display_value)

        # Now query_sql_output contains the SQL command with parameter values inserted
        print(f"\nQUERY_SQL_OUTPUT:\n{query_sql_output}")
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_multi_step_query_sql, multi_step_query_params)
                multi_step_queries = []
                for row in cursor:
                    # print(f"row: {row}")
                    multi_step_queries.append(row[2])
                print(f"{multi_step_queries=}")

        if isinstance(multi_step_queries, list):
            generate_query1 = multi_step_queries[0]
            generate_query2 = multi_step_queries[1]
            generate_query3 = multi_step_queries[2]

    return (
        gr.Textbox(value=generate_query1),
        gr.Textbox(value=generate_query2),
        gr.Textbox(value=generate_query3)
    )


def search_document(
        reranker_model_radio_input,
        reranker_top_k_slider_input,
        reranker_threshold_slider_input,
        threshold_value_slider_input,
        top_k_slider_input,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        text_search_checkbox_input,
        text_search_k_slider_input,
        document_metadata_text_input,
        query_text_input,
        sub_query1_text_input,
        sub_query2_text_input,
        sub_query3_text_input,
        partition_by_k_slider_input,
        answer_by_one_checkbox_input,
        extend_first_chunk_size_input,
        extend_around_chunk_size_input,
        use_image
):
    """
    Retrieve relevant splits for any question using similarity search.
    This is simply "top K" retrieval where we select documents based on embedding similarity to the query.
    """
    # 画像を使って回答がオンの場合、特定のパラメータ値を強制使用
    if use_image:
        answer_by_one_checkbox_input = False
        extend_first_chunk_size_input = 0
        extend_around_chunk_size_input = 0
        print(
            f"画像回答モード: answer_by_one_checkbox={answer_by_one_checkbox_input}, extend_first_chunk_size={extend_first_chunk_size_input}, extend_around_chunk_size={extend_around_chunk_size_input}")

    has_error = False
    if not query_text_input:
        has_error = True
        # gr.Warning("クエリを入力してください")
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        has_error = True
        gr.Warning("ドキュメントを選択してください")
    if document_metadata_text_input:
        document_metadata_text_input = document_metadata_text_input.strip()
        if "=" not in document_metadata_text_input or '"' in document_metadata_text_input or "'" in document_metadata_text_input or '\\' in document_metadata_text_input:
            has_error = True
            gr.Warning("メタデータの形式が正しくありません。key1=value1,key2=value2,... の形式で入力してください。")
        else:
            metadatas = document_metadata_text_input.split(",")
            for metadata in metadatas:
                if "=" not in metadata:
                    has_error = True
                    gr.Warning(
                        "メタデータの形式が正しくありません。key1=value1,key2=value2,... の形式で入力してください。")
                    break
    if has_error:
        return (
            gr.Textbox(value=""),
            gr.Markdown(
                "**検索結果数**: 0   |   **検索キーワード**: (0)[]",
                visible=True
            ),
            gr.Dataframe(
                value=None,
                row_count=(1, "fixed")
            )
        )

    def cut_lists(lists, limit=10):
        if not lists or len(lists) == 0:
            return lists
        if len(lists) > limit:
            # 文字列の長さでソート、長い文字列を優先的に削除
            sorted_indices = sorted(range(len(lists)), key=lambda i: len(lists[i]))
            lists = [lists[i] for i in sorted_indices[:limit]]

        return lists

    def generate_combinations(words_list):
        sampled_list = []
        n = len(words_list)
        threshold = 0.75
        if n < 3:
            sampled_list = words_list
            return sampled_list
        min_required = int(n * threshold)
        print(f"{n=}")
        print(f"{min_required=}")
        sampled_list += list(combinations(words_list, min_required))
        return sampled_list

    def process_lists(lists):
        contains_sql_list = []
        for lst in lists:
            if isinstance(lst, str):
                contains_sql_list.append(f"contains(de.embed_data, '{lst}') > 0")
            else:
                temp = " and ".join(str(item) for item in lst)
                if temp and temp not in contains_sql_list:
                    contains_sql_list.append(f"contains(de.embed_data, '{temp}') > 0")
        return ' OR '.join(contains_sql_list)

    def format_keywords(x):
        if len(x) > 0:
            formatted = '[' + ', '.join(x) + ']'
            return f"({len(x)}){formatted}"
        else:
            return "(0)[No Match]"

    def replace_newlines(text):
        if '\n\n' in text:
            text = text.replace('\n\n', '\n')
            return replace_newlines(text)
        else:
            return text

    region = get_region()

    query_text_input = query_text_input.strip()
    sub_query1_text_input = sub_query1_text_input.strip()
    sub_query2_text_input = sub_query2_text_input.strip()
    sub_query3_text_input = sub_query3_text_input.strip()

    # Use OracleAIVector
    unranked_docs = []
    threshold_value = threshold_value_slider_input
    top_k = top_k_slider_input
    doc_ids_str = "'" + "','".join([str(doc_id) for doc_id in doc_id_checkbox_group_input if doc_id]) + "'"
    print(f"{doc_ids_str=}")
    with_sql = """
               -- Select data
               WITH offsets AS (SELECT level - (:extend_around_chunk_size / 2 + 1) AS offset
                                FROM dual
               CONNECT BY level <= (:extend_around_chunk_size + 1)
                   )
                        , selected_embed_ids AS
                        ( \
               """
    where_sql = """
                    WHERE 1 = 1
                    AND de.doc_id = dc.id """
    where_metadata_sql = ""
    if document_metadata_text_input:
        metadata_conditions = []
        metadatas = document_metadata_text_input.split(",")
        for i, metadata in enumerate(metadatas):
            if "=" not in metadata:
                continue
            key, value = metadata.split("=", 1)
            # 使用正确的JSON路径语法和参数绑定
            metadata_conditions.append(f"json_value(dc.cmetadata, '$.\"{key}\"') = '{value}'")

        if metadata_conditions:
            where_metadata_sql = " AND (" + " AND ".join(metadata_conditions) + ") "
        print(f"{where_metadata_sql=}")

    # 注意：where_threshold_sql和where_sql现在在base_sql中直接构建，因为需要根据use_image使用不同的表别名
    # v4
    region = get_region()

    # 画像を使って回答がオンの場合、image_embeddingテーブルを使用、オフの場合はembeddingテーブルを使用
    if use_image:
        # 画像を使って回答がオンの場合、image_embeddingテーブルのみを使用
        base_sql = f"""
                        SELECT ie.doc_id doc_id, ie.embed_id embed_id, vector_distance(ie.embed_vector, (
                                SELECT to_vector(et.embed_vector) embed_vector
                                FROM
                                    dbms_vector_chain.utl_to_embeddings(
                                            :query_text,
                                            JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}}')) t,
                                    JSON_TABLE ( t.column_value, '$[*]'
                                            COLUMNS (
                                                embed_id NUMBER PATH '$.embed_id',
                                                embed_data VARCHAR2 ( 4000 ) PATH '$.embed_data',
                                                embed_vector CLOB PATH '$.embed_vector'
                                            )
                                        )
                                    et), COSINE
                                ) vector_distance
                        FROM {DEFAULT_COLLECTION_NAME}_image_embedding ie, {DEFAULT_COLLECTION_NAME}_collection dc
                        WHERE 1 = 1
                        AND ie.doc_id = dc.id """ + ("""
                        AND ie.doc_id IN (
                            SELECT TRIM(BOTH '''' FROM REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL)) AS doc_id
                            FROM DUAL
                            CONNECT BY REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL) IS NOT NULL
                        ) """ if not doc_id_all_checkbox_input else "") + where_metadata_sql + f"""
                        AND vector_distance(ie.embed_vector, (
                                SELECT to_vector(et.embed_vector) embed_vector
                                FROM
                                    dbms_vector_chain.utl_to_embeddings(
                                            :query_text,
                                            JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}}')) t,
                                    JSON_TABLE ( t.column_value, '$[*]'
                                            COLUMNS (
                                                embed_id NUMBER PATH '$.embed_id',
                                                embed_data VARCHAR2 ( 4000 ) PATH '$.embed_data',
                                                embed_vector CLOB PATH '$.embed_vector'
                                            )
                                        )
                                    et), COSINE
                                ) <= :threshold_value
                        ORDER BY vector_distance """
    else:
        # 画像を使って回答がオフの場合、embeddingテーブルのみを使用
        base_sql = f"""
                        SELECT de.doc_id doc_id, de.embed_id embed_id, vector_distance(de.embed_vector, (
                                SELECT to_vector(et.embed_vector) embed_vector
                                FROM
                                    dbms_vector_chain.utl_to_embeddings(
                                            :query_text,
                                            JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}}')) t,
                                    JSON_TABLE ( t.column_value, '$[*]'
                                            COLUMNS (
                                                embed_id NUMBER PATH '$.embed_id',
                                                embed_data VARCHAR2 ( 4000 ) PATH '$.embed_data',
                                                embed_vector CLOB PATH '$.embed_vector'
                                            )
                                        )
                                    et), COSINE
                                ) vector_distance
                        FROM {DEFAULT_COLLECTION_NAME}_embedding de, {DEFAULT_COLLECTION_NAME}_collection dc
                        WHERE 1 = 1
                        AND de.doc_id = dc.id """ + ("""
                        AND de.doc_id IN (
                            SELECT TRIM(BOTH '''' FROM REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL)) AS doc_id
                            FROM DUAL
                            CONNECT BY REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL) IS NOT NULL
                        ) """ if not doc_id_all_checkbox_input else "") + where_metadata_sql + f"""
                        AND vector_distance(de.embed_vector, (
                                SELECT to_vector(et.embed_vector) embed_vector
                                FROM
                                    dbms_vector_chain.utl_to_embeddings(
                                            :query_text,
                                            JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}}')) t,
                                    JSON_TABLE ( t.column_value, '$[*]'
                                            COLUMNS (
                                                embed_id NUMBER PATH '$.embed_id',
                                                embed_data VARCHAR2 ( 4000 ) PATH '$.embed_data',
                                                embed_vector CLOB PATH '$.embed_vector'
                                            )
                                        )
                                    et), COSINE
                                ) <= :threshold_value
                        ORDER BY vector_distance """
    base_sql += """
                    FETCH FIRST :partition_by PARTITIONS BY doc_id, :top_k ROWS ONLY
    """ if partition_by_k_slider_input > 0 else """
                    FETCH FIRST :top_k ROWS ONLY
    """
    select_sql = f"""
    ),
    selected_embed_id_doc_ids AS
    (
            SELECT DISTINCT s.embed_id + o.offset adjusted_embed_id, s.doc_id doc_id
            FROM selected_embed_ids s
            CROSS JOIN offsets o
    ),
    selected_results AS
    (
            SELECT s1.adjusted_embed_id adjusted_embed_id, s1.doc_id doc_id, NVL(s2.vector_distance, 999999.0) vector_distance
            FROM selected_embed_id_doc_ids s1
            LEFT JOIN selected_embed_ids s2
            ON s1.adjusted_embed_id = s2.embed_id AND s1.doc_id = s2.doc_id
    ),"""
    if use_image:
        # 画像を使って回答がオンの場合、image_embeddingテーブルを使用
        select_sql += f"""
    aggregated_results AS
    (
            SELECT json_value(dc.cmetadata, '$.file_name') name, ie.embed_id embed_id, ie.embed_data embed_data, ie.doc_id doc_id, MIN(s.vector_distance) vector_distance
            FROM selected_results s, {DEFAULT_COLLECTION_NAME}_image_embedding ie, {DEFAULT_COLLECTION_NAME}_collection dc
            WHERE s.adjusted_embed_id = ie.embed_id AND s.doc_id = dc.id and ie.doc_id = dc.id
            GROUP BY ie.doc_id, name, ie.embed_id, ie.embed_data"""
    else:
        # 画像を使って回答がオフの場合、embeddingテーブルを使用
        select_sql += f"""
    aggregated_results AS
    (
            SELECT json_value(dc.cmetadata, '$.file_name') name, de.embed_id embed_id, de.embed_data embed_data, de.doc_id doc_id, MIN(s.vector_distance) vector_distance
            FROM selected_results s, {DEFAULT_COLLECTION_NAME}_embedding de, {DEFAULT_COLLECTION_NAME}_collection dc
            WHERE s.adjusted_embed_id = de.embed_id AND s.doc_id = dc.id and de.doc_id = dc.id
            GROUP BY de.doc_id, name, de.embed_id, de.embed_data"""

    select_sql += """
            ORDER BY vector_distance
    ),
    ranked_data AS (
        SELECT
            ar.embed_data,
            ar.doc_id,
            ar.embed_id,
            ar.vector_distance,
            ar.name,
            LAG(ar.embed_id) OVER (PARTITION BY ar.doc_id ORDER BY ar.embed_id) AS prev_embed_id,
            LEAD(ar.embed_id) OVER (PARTITION BY ar.doc_id ORDER BY ar.embed_id) AS next_embed_id
        FROM
            aggregated_results ar
    ),
    grouped_data AS (
        SELECT
            rd.embed_data,
            rd.doc_id,
            rd.embed_id,
            rd.vector_distance,
            rd.name,
            CASE
                WHEN rd.prev_embed_id IS NULL OR rd.embed_id - rd.prev_embed_id > 1 THEN 1
                ELSE 0
            END AS new_group
        FROM
            ranked_data rd
    ),
    groups_marked AS (
        SELECT
            gd.embed_data,
            gd.doc_id,
            gd.embed_id,
            gd.vector_distance,
            gd.name,
            SUM(gd.new_group) OVER (PARTITION BY gd.doc_id ORDER BY gd.embed_id) AS group_id
        FROM
            grouped_data gd
    ),
    aggregated_data AS (
        SELECT
            RTRIM(XMLCAST(XMLAGG(XMLELEMENT(e, ad.embed_data || ',') ORDER BY ad.embed_id) AS CLOB), ',') AS combined_embed_data,
            ad.doc_id,
            MIN(ad.embed_id) AS min_embed_id,
            MIN(ad.vector_distance) AS min_vector_distance,
            ad.name
        FROM
            groups_marked ad
        GROUP BY
            ad.doc_id, ad.group_id, ad.name
    )
    SELECT
        ad.name,
        ad.min_embed_id AS embed_id,
        ad.combined_embed_data,
        ad.doc_id,
        ad.min_vector_distance AS vector_distance
    FROM
        aggregated_data ad
    ORDER BY
        ad.min_vector_distance
    """

    query_texts = [":query_text"]
    if sub_query1_text_input:
        query_texts.append(":sub_query1")
    if sub_query2_text_input:
        query_texts.append(":sub_query2")
    if sub_query3_text_input:
        query_texts.append(":sub_query3")
    print(f"{query_texts=}")

    search_texts = []
    if text_search_checkbox_input:
        # Generate the combined SQL based on available query texts
        search_texts = requests.post(os.environ["GINZA_API_ENDPOINT"],
                                     json={'query_text': query_text_input, 'language': 'ja'}).json()
        search_text = ""
        if search_texts and len(search_texts) > 0:
            search_texts = cut_lists(search_texts, text_search_k_slider_input)
            generated_combinations = generate_combinations(search_texts)
            search_text = process_lists(generated_combinations)
            # search_texts = cut_lists(search_texts, text_search_k_slider_input)
            # search_text = process_lists(search_texts)
        if len(search_text) > 0:
            # where_sql += """
            #             AND (""" + search_text + """) """
            # 注意：ここのwhere_sqlは異なるテーブルクエリで使用されるため、テーブル別名をハードコードできません
            # この部分のロジックはfull_text_search_sqlで処理されます
            region = get_region()
            if use_image:
                # 画像を使って回答がオンの場合、image_embeddingテーブルのみを使用
                full_text_search_sql = f"""
                            SELECT ie.doc_id doc_id, ie.embed_id embed_id, vector_distance(ie.embed_vector, (
                                    SELECT to_vector(et.embed_vector) embed_vector
                                    FROM
                                        dbms_vector_chain.utl_to_embeddings(
                                                :query_text,
                                                JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}}')) t,
                                        JSON_TABLE ( t.column_value, '$[*]'
                                                COLUMNS (
                                                    embed_id NUMBER PATH '$.embed_id',
                                                    embed_data VARCHAR2 ( 4000 ) PATH '$.embed_data',
                                                    embed_vector CLOB PATH '$.embed_vector'
                                                )
                                            )
                                        et), COSINE
                                    ) vector_distance
                            FROM {DEFAULT_COLLECTION_NAME}_image_embedding ie, {DEFAULT_COLLECTION_NAME}_collection dc
                            WHERE 1 = 1
                            AND ie.doc_id = dc.id
                            AND CONTAINS(ie.embed_data, :search_texts, 1) > 0 """ + ("""
                            AND ie.doc_id IN (
                                SELECT TRIM(BOTH '''' FROM REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL)) AS doc_id
                                FROM DUAL
                                CONNECT BY REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL) IS NOT NULL
                            ) """ if not doc_id_all_checkbox_input else "") + where_metadata_sql + """
                            ORDER BY SCORE(1) DESC FETCH FIRST :top_k ROWS ONLY """
            else:
                # 画像を使って回答がオフの場合、embeddingテーブルのみを使用
                full_text_search_sql = f"""
                            SELECT de.doc_id doc_id, de.embed_id embed_id, vector_distance(de.embed_vector, (
                                    SELECT to_vector(et.embed_vector) embed_vector
                                    FROM
                                        dbms_vector_chain.utl_to_embeddings(
                                                :query_text,
                                                JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-multilingual-v3.0"}}')) t,
                                        JSON_TABLE ( t.column_value, '$[*]'
                                                COLUMNS (
                                                    embed_id NUMBER PATH '$.embed_id',
                                                    embed_data VARCHAR2 ( 4000 ) PATH '$.embed_data',
                                                    embed_vector CLOB PATH '$.embed_vector'
                                                )
                                            )
                                        et), COSINE
                                    ) vector_distance
                            FROM {DEFAULT_COLLECTION_NAME}_embedding de, {DEFAULT_COLLECTION_NAME}_collection dc
                            WHERE 1 = 1
                            AND de.doc_id = dc.id
                            AND CONTAINS(de.embed_data, :search_texts, 1) > 0 """ + ("""
                            AND de.doc_id IN (
                                SELECT TRIM(BOTH '''' FROM REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL)) AS doc_id
                                FROM DUAL
                                CONNECT BY REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL) IS NOT NULL
                            ) """ if not doc_id_all_checkbox_input else "") + where_metadata_sql + """
                            ORDER BY SCORE(1) DESC FETCH FIRST :top_k ROWS ONLY """
            complete_sql = (with_sql + """
                UNION
        """.join(
                f"        ({base_sql.replace(':query_text', one_query_text)}        )" for one_query_text in
                query_texts) + """
                UNION
                ( """
                            + full_text_search_sql + """
                ) """
                            + select_sql)
    else:
        print(f"{query_texts=}")
        complete_sql = with_sql + """
            UNION
    """.join(
            f"        ({base_sql.replace(':query_text', one_query_text)}        )" for one_query_text in
            query_texts) + select_sql
        print(f"{complete_sql=}")

    # Prepare parameters for SQL execution
    params = {
        "extend_around_chunk_size": extend_around_chunk_size_input,
        "query_text": query_text_input,
        "threshold_value": threshold_value,
        "top_k": top_k
    }
    if not doc_id_all_checkbox_input:
        params["doc_ids"] = doc_ids_str
    if partition_by_k_slider_input > 0:
        params["partition_by"] = partition_by_k_slider_input
    if text_search_checkbox_input and search_texts:
        params["search_texts"] = " ACCUM ".join(search_texts)

    # Add sub-query texts if they exist
    if sub_query1_text_input:
        params["sub_query1"] = sub_query1_text_input
    if sub_query2_text_input:
        params["sub_query2"] = sub_query2_text_input
    if sub_query3_text_input:
        params["sub_query3"] = sub_query3_text_input

    # For debugging: Print the final SQL command.
    # Assuming complete_sql is your SQL string with placeholders like :extend_around_chunk_size, :doc_ids, etc.
    query_sql_output = complete_sql

    # Manually replace placeholders with parameter values for debugging
    for key, value in params.items():
        if key == "doc_ids":
            value = "''" + "'',''".join([str(doc_id) for doc_id in doc_id_checkbox_group_input if doc_id]) + "''"
        placeholder = f":{key}"
        # For the purpose of display, ensure the value is properly quoted if it's a string
        display_value = f"'{value}'" if isinstance(value, str) else str(value)
        query_sql_output = query_sql_output.replace(placeholder, display_value)
    query_sql_output = query_sql_output.strip() + ";"

    # Now query_sql_output contains the SQL command with parameter values inserted
    print(f"\nQUERY_SQL_OUTPUT:\n{query_sql_output}")

    with (pool.acquire() as conn):
        with conn.cursor() as cursor:
            max_retries = 3
            retries = 0
            while retries < max_retries:
                try:
                    cursor.execute(complete_sql, params)
                    break
                except Exception as e:
                    print(f"Exception: {e}")
                    retries += 1
                    print(f"Error executing SQL query: {e}. Retrying ({retries}/{max_retries})...")
                    time.sleep(10 * retries)
                    if retries == max_retries:
                        gr.Warning("データベース処理中にエラーが発生しました。しばらくしてから再度お試しください。")
                        return (
                            gr.Textbox(value=""),
                            gr.Markdown(
                                "**検索結果数**: 0   |   **検索キーワード**: (0)[]",
                                visible=True
                            ),
                            gr.Dataframe(
                                value=None,
                                row_count=(1, "fixed")
                            )
                        )
            for row in cursor:
                print(f"row: {row}")
                unranked_docs.append([row[0], row[1], row[2].read(), row[3], row[4]])

            if len(unranked_docs) == 0:
                docs_dataframe = pd.DataFrame(
                    columns=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"])

                return (
                    gr.Textbox(value=query_sql_output.strip()),
                    gr.Markdown(
                        "**検索結果数**: " + str(len(docs_dataframe)) + "   |   **検索キーワード**: (" + str(
                            len(search_texts)) + ")[" + ', '.join(search_texts) + "]",
                        visible=True),
                    gr.Dataframe(
                        value=docs_dataframe,
                        wrap=True,
                        headers=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"],
                        column_widths=["4%", "50%", "6%", "8%", "6%", "8%"],
                        row_count=(1, "fixed"),
                    ),
                )

            # ToDo: In case of error
            if 'cohere/rerank' in reranker_model_radio_input:
                unranked = []
                for doc in unranked_docs:
                    unranked.append(doc[2])
                ranked_results = rerank_documents_response(query_text_input, unranked, reranker_model_radio_input)
                ranked_scores = [0.0] * len(unranked_docs)
                for result in ranked_results:
                    ranked_scores[result['index']] = result['relevance_score']
                docs_data = [{'CONTENT': doc[2],
                              'EMBED_ID': doc[1],
                              'SOURCE': str(doc[3]) + ":" + doc[0],
                              'DISTANCE': '-' if str(doc[4]) == '999999.0' else str(doc[4]),
                              'SCORE': ce_score} for doc, ce_score in zip(unranked_docs, ranked_scores)]
                docs_dataframe = pd.DataFrame(docs_data)
                docs_dataframe = docs_dataframe[
                    docs_dataframe['SCORE'] >= float(reranker_threshold_slider_input)
                    ].sort_values(by='SCORE', ascending=False).head(
                    reranker_top_k_slider_input)
            else:
                docs_data = [{'CONTENT': doc[2],
                              'EMBED_ID': doc[1],
                              'SOURCE': str(doc[3]) + ":" + doc[0],
                              'DISTANCE': '-' if str(doc[4]) == '999999.0' else str(doc[4]),
                              'SCORE': '-'} for doc in unranked_docs]
                docs_dataframe = pd.DataFrame(docs_data)

            print(f"{extend_first_chunk_size_input=}")
            if extend_first_chunk_size_input > 0 and len(docs_dataframe) > 0:
                filtered_doc_ids = "'" + "','".join(
                    [source.split(':')[0] for source in docs_dataframe['SOURCE'].tolist()]) + "'"
                print(f"{filtered_doc_ids=}")
                # 画像を使って回答の設定に応じて適切なテーブルを選択
                if use_image:
                    # 画像を使って回答がオンの場合、image_embeddingテーブルを使用
                    select_extend_first_chunk_sql = f"""
SELECT
        json_value(dc.cmetadata, '$.file_name') name,
        MIN(ie.embed_id) embed_id,
        RTRIM(XMLCAST(XMLAGG(XMLELEMENT(e, ie.embed_data || ',') ORDER BY ie.embed_id) AS CLOB), ',') AS embed_data,
        ie.doc_id doc_id,
        '999999.0' vector_distance
FROM
        {DEFAULT_COLLECTION_NAME}_image_embedding ie, {DEFAULT_COLLECTION_NAME}_collection dc
WHERE
        ie.doc_id = dc.id AND
        ie.doc_id IN (:filtered_doc_ids) AND
        ie.embed_id <= :extend_first_chunk_size
GROUP BY
        ie.doc_id, name
ORDER
        BY ie.doc_id
            """
                else:
                    # 画像を使って回答がオフの場合、embeddingテーブルを使用
                    select_extend_first_chunk_sql = f"""
SELECT
        json_value(dc.cmetadata, '$.file_name') name,
        MIN(de.embed_id) embed_id,
        RTRIM(XMLCAST(XMLAGG(XMLELEMENT(e, de.embed_data || ',') ORDER BY de.embed_id) AS CLOB), ',') AS embed_data,
        de.doc_id doc_id,
        '999999.0' vector_distance
FROM
        {DEFAULT_COLLECTION_NAME}_embedding de, {DEFAULT_COLLECTION_NAME}_collection dc
WHERE
        de.doc_id = dc.id AND
        de.doc_id IN (:filtered_doc_ids) AND
        de.embed_id <= :extend_first_chunk_size
GROUP BY
        de.doc_id, name
ORDER
        BY de.doc_id
            """
                select_extend_first_chunk_sql = (select_extend_first_chunk_sql
                                                 .replace(':filtered_doc_ids', filtered_doc_ids)
                                                 .replace(':extend_first_chunk_size',
                                                          str(extend_first_chunk_size_input)))
                print(f"{select_extend_first_chunk_sql=}")
                query_sql_output += "\n" + select_extend_first_chunk_sql.strip()
                cursor.execute(select_extend_first_chunk_sql)
                first_chunks_df = pd.DataFrame(columns=docs_dataframe.columns)
                for row in cursor:
                    new_data = pd.DataFrame(
                        {'CONTENT': row[2].read(), 'EMBED_ID': row[1], 'SOURCE': str(row[3]) + ":" + row[0],
                         'DISTANCE': '-', 'SCORE': '-'},
                        index=[2])
                    first_chunks_df = pd.concat([new_data, first_chunks_df], ignore_index=True)
                print(f"{first_chunks_df=}")

                # 创建一个空的DataFrame,用于存储更新后的数据
                updated_df = pd.DataFrame(columns=docs_dataframe.columns)

                # 记录每个SOURCE的初始插入位置
                insert_positions = {}

                # 遍历原始数据的每一行
                for index, row in docs_dataframe.iterrows():
                    source = row['SOURCE']

                    # 如果当前SOURCE还没有记录初始插入位置,则将其初始化为当前位置
                    if source not in insert_positions:
                        insert_positions[source] = len(updated_df)

                    # 找到新数据中与当前SOURCE相同的行
                    same_source_new_data = first_chunks_df[first_chunks_df['SOURCE'] == source]

                    # 遍历新数据中与当前SOURCE相同的行
                    for _, new_row in same_source_new_data.iterrows():
                        # 在当前行之前插入新数据
                        updated_df = pd.concat([updated_df[:insert_positions[source]],
                                                pd.DataFrame(new_row).T,
                                                updated_df[insert_positions[source]:]])

                        # 更新当前SOURCE的插入位置
                        insert_positions[source] += 1

                    # 将当前行添加到updated_df中
                    updated_df = pd.concat([updated_df[:insert_positions[source]],
                                            pd.DataFrame(row).T,
                                            updated_df[insert_positions[source]:]])

                    # 更新当前SOURCE的插入位置
                    insert_positions[source] += 1
                docs_dataframe = updated_df

            conn.commit()

            docs_dataframe['KEY_WORDS'] = docs_dataframe['CONTENT'].apply(
                lambda x: [word for word in search_texts if word.lower() in x.lower()])
            docs_dataframe['KEY_WORDS'] = docs_dataframe['KEY_WORDS'].apply(format_keywords)
            docs_dataframe['CONTENT'] = docs_dataframe['CONTENT'].apply(replace_newlines)

            if answer_by_one_checkbox_input:
                docs_dataframe = docs_dataframe[docs_dataframe['SOURCE'] == docs_dataframe['SOURCE'].iloc[0]]

            docs_dataframe = docs_dataframe.reset_index(drop=True)
            docs_dataframe.insert(0, 'NO', pd.Series(range(1, len(docs_dataframe) + 1)))
            print(f"{docs_dataframe=}")

            if len(docs_dataframe) == 0:
                docs_dataframe = pd.DataFrame(
                    columns=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"])

            return (
                gr.Textbox(value=query_sql_output.strip()),
                gr.Markdown(
                    "**検索結果数**: " + str(len(docs_dataframe)) + "   |   **検索キーワード**: (" + str(
                        len(search_texts)) + ")[" + ', '.join(search_texts) + "]",
                    visible=True
                ),
                gr.Dataframe(
                    value=docs_dataframe,
                    wrap=True,
                    headers=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"],
                    column_widths=["4%", "50%", "6%", "8%", "6%", "6%", "8%"],
                    row_count=(len(docs_dataframe) if len(docs_dataframe) > 0 else 1, "fixed")
                )
            )


def extract_and_format(input_str, search_result_df):
    json_arrays = re.findall(r'\[\n.*?\{.*?}\n.*?]', input_str, flags=re.DOTALL)
    if not json_arrays:
        return (
                input_str +
                f"\n"
                f"---回答内で参照されているコンテキスト---"
                f"\n"
                f"回答にコンテキストが存在しないか、コンテキストの形式が正しくありません。"
        )

    extracted = []
    for json_str in json_arrays:
        input_str = input_str.replace(json_str, '')
        json_str = json_str.replace('\n', '').replace('\r', '')
        data = json.loads(json_str)

        for item in data:
            print(f"{item=}")
            if isinstance(item, dict):
                if "EMBED_ID" in item and "SOURCE" in item:
                    extracted.append({
                        "EMBED_ID": item["EMBED_ID"],
                        "SOURCE": item["SOURCE"]
                    })

    formatted = (
            input_str +
            f"\n"
            f"---回答内で参照されているコンテキスト---"
            f"\n"
    )
    formatted += "[\n"
    for item in extracted:
        content = "N/A"
        if item["EMBED_ID"] and isinstance(item["EMBED_ID"], int) and item["SOURCE"]:
            content = search_result_df.loc[
                (search_result_df["EMBED_ID"].astype(int) == int(item["EMBED_ID"])) &
                (search_result_df["SOURCE"] == item["SOURCE"]),
                "CONTENT"
            ].values
            if len(content) > 0:
                content = content[0]
                content = content.replace('"', '\'')
                content = content.replace('\n', ' ').replace('\r', ' ')
        formatted += (
            '    {{\n'
            '        "EMBED_ID": {},\n'
            '        "SOURCE": "{}",\n'
            '        "CONTENT": "{}"\n'
            '    }},\n'
        ).format(item["EMBED_ID"], item["SOURCE"], content)
    if extracted:
        formatted = formatted.rstrip(",\n") + "\n"
    formatted += "]"

    return formatted


def extract_citation(input_str):
    # 2つの部分の内容をマッチング
    pattern = '^(.*?)\n---回答内で参照されているコンテキスト---\n(.*?)$'
    match = re.search(pattern, input_str, re.DOTALL)
    if match:
        part1 = match.group(1).strip()
        part2 = match.group(2).strip()
        return part1, part2
    else:
        return None, None


#
# def generate_langgpt_prompt_ja(context, query_text, include_citation=False, include_current_time=False):
#     # 固定するエラーメッセージ
#     error_message = "申し訳ありませんが、コンテキストから適切な回答を見つけることができませんでした。別の LLM モデルをお試しいただくか、クエリの内容や設定を少し調整していただくことで解決できるかもしれません。"
#
#     # LangGPTテンプレートの基本構造
#     prompt = f"""
# # Role: 厳格コンテキストQA
#
# ## Profile
#
# - Author: User
# - Version: 0.2
# - Language: 日本語
# - Description: 厳密なコンテキストベースの質問応答システム。提供された文脈データのみを使用し、一切の改変を加えずに回答します。
#
# ### Core Skills
# 1. コンテキストの完全一致検索
# 2. 文脈改変の完全排除
# 3. 回答不能時の定型通知
# 4. マルチフォーマット出力対応
#
# ## Rules
# 1. {error_message}
# 2. 回答は<context>の内容に100%依存
# 3. 部分一致や推測を一切行わない
# 4. 日付情報がある場合の時系列処理（最新情報優先）
# 5. 引用情報の厳密なフォーマット保持
#
# ## Workflow
# 1. コンテキスト解析フェーズ
#    - UTF-8エンコーディングで厳密解析
#    - メタデータ（EMBED_ID/SOURCE）の抽出
# 2. クエリマッチングフェーズ
#    - 完全文字列マッチングアルゴリズム適用
#    - 複数候補がある場合は最新日付を優先
# 3. 回答生成フェーズ
#    - マッチデータの直接引用
#    - 引用情報の構造化出力（要求時）
# 4. エラーハンドリング
#    - マッチなし → 定型エラーメッセージ
#    - 矛盾データ → 事実関係を列挙
#
# ## Initialization
# As a/an <Role>, you must follow the <Rules> in <Language>.
# コンテキストQAシステムが起動しました。以下の要素を提供ください：
#
# <context>
# {context}
# </context>
#
# <query>
# {query_text}
# </query>
# """
#
#     # 引用情報の条件付き追加
#     if include_citation:
#         prompt += """
# ### 引用フォーマット規約
# - 出力直後にJSON配列を追加
# - 厳密な構造保持（```json不使用）：
# [
#     {
#         "EMBED_ID": <一意な識別子>,
#         "SOURCE": "<情報の出典>",
#         "EXTRACT_TEXT": "<引用部分の原文>"
#     }
# ]
# """
#
#     # 時間処理の条件付き追加
#     if include_current_time:
#         current_time = datetime.now().strftime('%Y%m%d%H%M%S')
#         prompt += f"""
# ### 時系列処理規則
# - 基準時刻: {current_time}
# - 最新情報判定アルゴリズム：
#   1. 日付データの正規化（YYYYMMDDHHMMSS）
#   2. 時刻近接順にソート
#   3. 同一情報のバージョン管理
# - 期間指定クエリ対応：
#   /period:start=YYYYMMDD,end=YYYYMMDD
# """
#
#     return prompt.strip()


async def chat_document(
        search_result,
        llm_answer_checkbox,
        include_citation,
        include_current_time,
        use_image,
        query_text,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        rag_prompt_template
):
    has_error = False
    if not query_text:
        has_error = True
        # gr.Warning("クエリを入力してください")
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        has_error = True
        # gr.Warning("ドキュメントを選択してください")
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        has_error = True
        gr.Warning("検索結果が見つかりませんでした。設定もしくはクエリを変更して再度ご確認ください。")
    if has_error:
        yield (
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value="")
        )
        return

    query_text = query_text.strip()

    xai_grok_3_response = ""
    command_a_response = ""
    llama_4_maverick_response = ""
    llama_4_scout_response = ""
    llama_3_3_70b_response = ""
    llama_3_2_90b_vision_response = ""
    openai_gpt4o_response = ""
    openai_gpt4_response = ""
    azure_openai_gpt4o_response = ""
    azure_openai_gpt4_response = ""

    xai_grok_3_checkbox = False
    command_a_checkbox = False
    llama_4_maverick_checkbox = False
    llama_4_scout_checkbox = False
    llama_3_3_70b_checkbox = False
    llama_3_2_90b_vision_checkbox = False
    openai_gpt4o_checkbox = False
    openai_gpt4_checkbox = False
    azure_openai_gpt4o_checkbox = False
    azure_openai_gpt4_checkbox = False
    if "xai/grok-3" in llm_answer_checkbox:
        xai_grok_3_checkbox = True
    if "cohere/command-a" in llm_answer_checkbox:
        command_a_checkbox = True
    if "meta/llama-4-maverick-17b-128e-instruct-fp8" in llm_answer_checkbox:
        llama_4_maverick_checkbox = True
    if "meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox:
        llama_4_scout_checkbox = True
    if "meta/llama-3-3-70b" in llm_answer_checkbox:
        llama_3_3_70b_checkbox = True
    if "meta/llama-3-2-90b-vision" in llm_answer_checkbox:
        llama_3_2_90b_vision_checkbox = True
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt4o_checkbox = True
    if "openai/gpt-4" in llm_answer_checkbox:
        openai_gpt4_checkbox = True
    if "azure_openai/gpt-4o" in llm_answer_checkbox:
        azure_openai_gpt4o_checkbox = True
    if "azure_openai/gpt-4" in llm_answer_checkbox:
        azure_openai_gpt4_checkbox = True

    # context = '\n'.join(search_result['CONTENT'].astype(str).values)
    context = search_result[['EMBED_ID', 'SOURCE', 'CONTENT']].to_dict('records')
    #     print(f"{context=}")
    #     # system_text = f"""
    #     #         Use the following pieces of Context to answer the question at the end.
    #     #         If you don't know the answer, just say that you don't know, don't try to make up an answer.
    #     #         Use the EXACT TEXT from the Context WITHOUT ANY MODIFICATIONS, REORGANIZATION or EMBELLISHMENT.
    #     #         Don't try to answer anything that isn't in Context.
    #     #         Context:
    #     #         ```
    #     #         {context}
    #     #         ```
    #     #         """
    #     system_text = f"""
    # ---目標：---
    # 次のコンテキストを使用して、最後にある質問に答えてください。
    # コンテキストにないことについては答えようとしないでください。
    # もし答えがわからない場合は、「申し訳ありませんが、コンテキストから適切な回答を見つけることができませんでした。別の LLM モデルをお試しいただくか、クエリの内容や設定を少し調整していただくことで解決できるかもしれません。」と言ってください。
    # 答えをでっち上げようとしないでください。
    # コンテキストの正確なテキストを使用し、**一切の修正、再構成、または脚色を加えずに**使用してください。
    # \n
    # """
    #
    #     if include_citation:
    #         system_text += f"""
    # After providing the answer, **include the 'EMBED_ID' and 'SOURCE' of the 'CONTENT' used to formulate the answer in JSON format**.
    # The JSON array must strictly follow this structure without '```json' and '```' around it:
    #
    # [
    #     {{
    #         "EMBED_ID": <A unique identifier for the content piece.>,
    #         "SOURCE": "<A string indicating the origin of the content.>"
    #     }}
    # ]
    #
    # If multiple pieces of CONTENT are used, include all relevant EMBED_IDs and SOURCEs in the JSON array.
    # \n
    # """
    #
    #     current_time = datetime.now()
    #     formatted_time = current_time.strftime('%Y%m%d%H%M%S')
    #
    #     if include_current_time:
    #         system_text += f"""
    # The current time is {formatted_time} in format '%Y%m%d%H%M%S', When the Context contains multiple entries with dates,
    # please consider these dates carefully when answering the question:
    # - If the question asks about the "latest" or "most recent" information, use data from the most recent date
    # - If the question asks about a specific time period, use data from that corresponding time period
    # - If comparing different time periods, clearly specify which date's data you are referencing
    # \n
    # """
    #
    #     system_text += f"""
    # ---コンテキスト：--- \n
    # <context>
    # {context}
    # </context>
    # \n
    # """
    #
    #     user_text = f"""
    # ---質問：--- \n
    # <query>
    # {query_text}
    # </query>
    # \n
    # ---役に立つ回答：--- \n
    # """

    system_text = ""
    user_text = get_langgpt_rag_prompt(context, query_text, include_citation, include_current_time, rag_prompt_template)

    xai_grok_3_user_text = user_text
    command_a_user_text = user_text

    llama_4_maverick_user_text = user_text
    llama_4_scout_user_text = user_text
    llama_3_3_70b_user_text = user_text
    llama_3_2_90b_vision_user_text = user_text
    openai_gpt4o_user_text = user_text
    openai_gpt4_user_text = user_text
    azure_openai_gpt4o_user_text = user_text
    azure_openai_gpt4_user_text = user_text

    async for xai_grok_3, command_a, llama_4_maverick, llama_4_scout, llama_3_3_70b, llama_3_2_90b_vision, gpt4o, gpt4, azure_gpt4o, azure_gpt4 in chat(
            system_text,
            xai_grok_3_user_text,
            command_a_user_text,
            None,
            llama_4_maverick_user_text,
            None,
            llama_4_scout_user_text,
            llama_3_3_70b_user_text,
            None,
            llama_3_2_90b_vision_user_text,
            openai_gpt4o_user_text,
            openai_gpt4_user_text,
            azure_openai_gpt4o_user_text,
            azure_openai_gpt4_user_text,
            xai_grok_3_checkbox,
            command_a_checkbox,
            llama_4_maverick_checkbox,
            llama_4_scout_checkbox,
            llama_3_3_70b_checkbox,
            llama_3_2_90b_vision_checkbox,
            openai_gpt4o_checkbox,
            openai_gpt4_checkbox,
            azure_openai_gpt4o_checkbox,
            azure_openai_gpt4_checkbox
    ):
        xai_grok_3_response += xai_grok_3
        command_a_response += command_a
        llama_4_maverick_response += llama_4_maverick
        llama_4_scout_response += llama_4_scout
        llama_3_3_70b_response += llama_3_3_70b
        llama_3_2_90b_vision_response += llama_3_2_90b_vision
        openai_gpt4o_response += gpt4o
        openai_gpt4_response += gpt4
        azure_openai_gpt4o_response += azure_gpt4o
        azure_openai_gpt4_response += azure_gpt4
        yield (
            gr.Markdown(value=xai_grok_3_response),
            gr.Markdown(value=command_a_response),
            gr.Markdown(value=llama_4_maverick_response),
            gr.Markdown(value=llama_4_scout_response),
            gr.Markdown(value=llama_3_3_70b_response),
            gr.Markdown(value=llama_3_2_90b_vision_response),
            gr.Markdown(value=openai_gpt4o_response),
            gr.Markdown(value=openai_gpt4_response),
            gr.Markdown(value=azure_openai_gpt4o_response),
            gr.Markdown(value=azure_openai_gpt4_response)
        )


async def append_citation(
        search_result,
        llm_answer_checkbox,
        include_citation,
        query_text,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        xai_grok_3_answer_text,
        command_a_answer_text,
        llama_4_maverick_answer_text,
        llama_4_scout_answer_text,
        llama_3_3_70b_answer_text,
        llama_3_2_90b_vision_answer_text,
        openai_gpt4o_answer_text,
        openai_gpt4_answer_text,
        azure_openai_gpt4o_answer_text,
        azure_openai_gpt4_answer_text
):
    has_error = False
    if not query_text:
        has_error = True
        # gr.Warning("クエリを入力してください")
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        has_error = True
        # gr.Warning("ドキュメントを選択してください")
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        has_error = True
        # gr.Warning("検索結果が見つかりませんでした。設定もしくはクエリを変更して再度ご確認ください。")
    if has_error:
        yield (
            gr.Markdown(value=xai_grok_3_answer_text),
            gr.Markdown(value=command_a_answer_text),
            gr.Markdown(value=llama_4_maverick_answer_text),
            gr.Markdown(value=llama_4_scout_answer_text),
            gr.Markdown(value=llama_3_3_70b_answer_text),
            gr.Markdown(value=llama_3_2_90b_vision_answer_text),
            gr.Markdown(value=openai_gpt4o_answer_text),
            gr.Markdown(value=openai_gpt4_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_answer_text),
            gr.Markdown(value=azure_openai_gpt4_answer_text)
        )
        return

    if not include_citation:
        yield (
            gr.Markdown(value=xai_grok_3_answer_text),
            gr.Markdown(value=command_a_answer_text),
            gr.Markdown(value=llama_4_maverick_answer_text),
            gr.Markdown(value=llama_4_scout_answer_text),
            gr.Markdown(value=llama_3_3_70b_answer_text),
            gr.Markdown(value=llama_3_2_90b_vision_answer_text),
            gr.Markdown(value=openai_gpt4o_answer_text),
            gr.Markdown(value=openai_gpt4_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_answer_text),
            gr.Markdown(value=azure_openai_gpt4_answer_text)
        )
        return

    if "xai/grok-3" in llm_answer_checkbox:
        xai_grok_3_answer_text = extract_and_format(xai_grok_3_answer_text, search_result)
    if "cohere/command-a" in llm_answer_checkbox:
        command_a_answer_text = extract_and_format(command_a_answer_text, search_result)
    if "meta/llama-4-maverick-17b-128e-instruct-fp8" in llm_answer_checkbox:
        llama_4_maverick_answer_text = extract_and_format(llama_4_maverick_answer_text, search_result)
    if "meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox:
        llama_4_scout_answer_text = extract_and_format(llama_4_scout_answer_text, search_result)
    if "meta/llama-3-3-70b" in llm_answer_checkbox:
        llama_3_3_70b_answer_text = extract_and_format(llama_3_3_70b_answer_text, search_result)
    if "meta/llama-3-2-90b-vision" in llm_answer_checkbox:
        llama_3_2_90b_vision_answer_text = extract_and_format(llama_3_2_90b_vision_answer_text, search_result)
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt4o_answer_text = extract_and_format(openai_gpt4o_answer_text, search_result)
    if "openai/gpt-4" in llm_answer_checkbox:
        openai_gpt4_answer_text = extract_and_format(openai_gpt4_answer_text, search_result)
    if "azure_openai/gpt-4o" in llm_answer_checkbox:
        azure_openai_gpt4o_answer_text = extract_and_format(azure_openai_gpt4o_answer_text, search_result)
    if "azure_openai/gpt-4" in llm_answer_checkbox:
        azure_openai_gpt4_answer_text = extract_and_format(azure_openai_gpt4_answer_text, search_result)
    yield (
        gr.Markdown(value=xai_grok_3_answer_text),
        gr.Markdown(value=command_a_answer_text),
        gr.Markdown(value=llama_4_maverick_answer_text),
        gr.Markdown(value=llama_4_scout_answer_text),
        gr.Markdown(value=llama_3_3_70b_answer_text),
        gr.Markdown(value=llama_3_2_90b_vision_answer_text),
        gr.Markdown(value=openai_gpt4o_answer_text),
        gr.Markdown(value=openai_gpt4_answer_text),
        gr.Markdown(value=azure_openai_gpt4o_answer_text),
        gr.Markdown(value=azure_openai_gpt4_answer_text)
    )
    return


async def process_single_image_streaming(image_url, query_text, llm_answer_checkbox_group, target_models, image_index,
                                         doc_id, img_id, custom_image_prompt=None):
    """
    単一画像を選択されたLLMモデルで処理し、ストリーミング形式で回答を返す

    Args:
        image_url: 画像のURL
        query_text: クエリテキスト
        llm_answer_checkbox_group: 選択されたLLMモデルのリスト
        target_models: 対象モデルのリスト
        image_index: 画像のインデックス
        doc_id: ドキュメントID
        img_id: 画像ID
        custom_image_prompt: カスタム画像プロンプトテンプレート

    Yields:
        dict: 各モデルの部分的な回答を含む辞書
    """
    if custom_image_prompt:
        custom_image_prompt = custom_image_prompt.replace('{{query_text}}', '{query_text}')

    region = get_region()

    # 各モデルのタスクジェネレーターを作成
    async def create_model_task(model):
        llm = None  # LLMインスタンスを初期化
        try:
            if model not in llm_answer_checkbox_group:
                # 選択されていないモデルは即座に完了を通知
                yield "TASK_DONE"
                return

            print(f"\n=== 画像 {image_index} (doc_id: {doc_id}, img_id: {img_id}) - {model} での処理 ===")

            if model == "meta/llama-4-maverick-17b-128e-instruct-fp8":
                llm = ChatOCIGenAI(
                    model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
                    provider="meta",
                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                    compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
                    model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
                )
            elif model == "meta/llama-4-scout-17b-16e-instruct":
                llm = ChatOCIGenAI(
                    model_id="meta.llama-4-scout-17b-16e-instruct",
                    provider="meta",
                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                    compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
                    model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
                )
            elif model == "meta/llama-3-2-90b-vision":
                llm = ChatOCIGenAI(
                    model_id="meta.llama-3.2-90b-vision-instruct",
                    provider="meta",
                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                    compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
                    model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
                )
            elif model == "openai/gpt-4o":
                load_dotenv(find_dotenv())
                llm = ChatOpenAI(
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
            elif model == "azure_openai/gpt-4o":
                load_dotenv(find_dotenv())
                llm = AzureChatOpenAI(
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
            else:
                # 未対応のモデルは即座に完了を通知
                yield "TASK_DONE"
                return

            # メッセージを作成
            prompt_text = get_image_qa_prompt(query_text, custom_image_prompt)
            prompt_text = prompt_text.replace('{{query_text}}', '{query_text}')
            human_message = HumanMessage(content=[
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ])
            messages = [human_message]

            # LLMに送信して回答を取得
            start_time = time.time()
            # langfuse_handler = CallbackHandler(
            #     secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            #     public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            #     host=os.environ["LANGFUSE_HOST"],
            # )

            # 表示用に画像を圧縮
            compressed_image_url = compress_image_for_display(image_url)

            # ヘッダー情報を最初にyield（圧縮された画像を使用）
            header_text = f"\n\n---\n\n![画像]({compressed_image_url})\n\n**画像 {image_index} (doc_id: {doc_id}, img_id: {img_id}) による回答：**\n\n"
            # header_text = f"\n\n**画像 {image_index} (doc_id: {doc_id}, img_id: {img_id}) による回答：**\n\n"
            yield header_text

            # ストリーミングで回答を取得
            # Avoid for: Error uploading media: HTTPConnectionPool(host='minio', port=9000)
            # async for chunk in llm.astream(messages, config={"callbacks": [langfuse_handler],
            #                                                  "metadata": {"ls_model_name": model}}):
            async for chunk in llm.astream(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    yield chunk.content

            end_time = time.time()
            inference_time = end_time - start_time
            print(f"\n\n推論時間: {inference_time:.2f}秒")
            print(f"=== {model} での処理完了 ===\n")

            # 推論時間を追加
            yield f"\n\n推論時間: {inference_time:.2f}秒\n\n"
            yield "TASK_DONE"

        except Exception as e:
            print(f"エラーが発生しました ({model}): {e}")
            # 表示用に画像を圧縮
            compressed_image_url = compress_image_for_display(image_url)
            error_text = f"\n\n---\n\n![画像]({compressed_image_url})\n\n**画像 {image_index} (doc_id: {doc_id}, img_id: {img_id}) による回答：**\n\nエラーが発生しました: {e}\n\n"
            # error_text = f"\n\n**画像 {image_index} (doc_id: {doc_id}, img_id: {img_id}) による回答：**\n\nエラーが発生しました: {e}\n\n"
            yield error_text
            yield "TASK_DONE"
        finally:
            # リソースクリーンアップ：LLMクライアントの接続を適切に閉じる
            try:
                await cleanup_llm_client_async(llm)
                print(f"モデル {model} のLLMクライアントクリーンアップが完了しました")
            except Exception as cleanup_error:
                print(f"モデル {model} のLLMクライアントクリーンアップ中にエラー: {cleanup_error}")
            finally:
                llm = None  # 参照をクリア

                # 追加の強制クリーンアップ
                import gc
                gc.collect()  # ガベージコレクションを実行

    # 各モデルのジェネレーターを作成
    llama_4_maverick_gen = create_model_task("meta/llama-4-maverick-17b-128e-instruct-fp8")
    llama_4_scout_gen = create_model_task("meta/llama-4-scout-17b-16e-instruct")
    llama_3_2_90b_vision_gen = create_model_task("meta/llama-3-2-90b-vision")
    openai_gpt4o_gen = create_model_task("openai/gpt-4o")
    azure_openai_gpt4o_gen = create_model_task("azure_openai/gpt-4o")

    # 各モデルの応答を蓄積
    llama_4_maverick_response = ""
    llama_4_scout_response = ""
    llama_3_2_90b_vision_response = ""
    openai_gpt4o_response = ""
    azure_openai_gpt4o_response = ""

    # 各モデルの状態を追跡
    responses_status = ["", "", "", "", ""]

    # タイムアウト設定（最大5分）
    import asyncio
    timeout_seconds = 300
    start_time = time.time()

    try:
        while True:
            # タイムアウトチェック
            if time.time() - start_time > timeout_seconds:
                print(f"画像処理がタイムアウトしました（{timeout_seconds}秒）")
                break

            responses = ["", "", "", "", ""]
            generators = [llama_4_maverick_gen, llama_4_scout_gen, llama_3_2_90b_vision_gen, openai_gpt4o_gen,
                          azure_openai_gpt4o_gen]

            for i, gen in enumerate(generators):
                if responses_status[i] == "TASK_DONE":
                    continue

                try:
                    # タイムアウト付きでanextを実行
                    response = await asyncio.wait_for(anext(gen), timeout=30.0)
                    if response:
                        if response == "TASK_DONE":
                            responses_status[i] = response
                        else:
                            responses[i] = response
                except StopAsyncIteration:
                    responses_status[i] = "TASK_DONE"
                except asyncio.TimeoutError:
                    print(f"モデル {i} の処理がタイムアウトしました")
                    responses_status[i] = "TASK_DONE"
                except Exception as e:
                    print(f"モデル {i} の処理中にエラーが発生しました: {e}")
                    responses_status[i] = "TASK_DONE"

            # 応答を蓄積
            llama_4_maverick_response += responses[0]
            llama_4_scout_response += responses[1]
            llama_3_2_90b_vision_response += responses[2]
            openai_gpt4o_response += responses[3]
            azure_openai_gpt4o_response += responses[4]

            # 現在の状態をyield
            yield {
                "meta/llama-4-maverick-17b-128e-instruct-fp8": llama_4_maverick_response,
                "meta/llama-4-scout-17b-16e-instruct": llama_4_scout_response,
                "meta/llama-3-2-90b-vision": llama_3_2_90b_vision_response,
                "openai/gpt-4o": openai_gpt4o_response,
                "azure_openai/gpt-4o": azure_openai_gpt4o_response
            }

            # すべてのタスクが完了したかチェック
            if all(response_status == "TASK_DONE" for response_status in responses_status):
                print("All image processing tasks completed")
                break

    finally:
        # 最終的なリソースクリーンアップ：すべてのジェネレーターを適切に閉じる
        generators = [llama_4_maverick_gen, llama_4_scout_gen, llama_3_2_90b_vision_gen, openai_gpt4o_gen,
                      azure_openai_gpt4o_gen]
        generator_names = ["llama_4_maverick", "llama_4_scout", "llama_3_2_90b_vision", "openai_gpt4o",
                           "azure_openai_gpt4o"]

        for i, gen in enumerate(generators):
            try:
                # 非同期ジェネレーターの場合
                if hasattr(gen, 'aclose'):
                    await gen.aclose()
                    print(f"ジェネレーター {generator_names[i]} の非同期クローズが完了しました")
                elif hasattr(gen, 'close'):
                    gen.close()
                    print(f"ジェネレーター {generator_names[i]} の同期クローズが完了しました")

                # ジェネレーター内のLLMクライアントも確実にクリーンアップ
                if hasattr(gen, 'gi_frame') and gen.gi_frame is not None:
                    # ジェネレーターがまだ実行中の場合、強制終了
                    try:
                        gen.close()
                        print(f"実行中のジェネレーター {generator_names[i]} を強制終了しました")
                    except GeneratorExit:
                        pass  # 正常な終了
                    except Exception as force_close_error:
                        print(f"ジェネレーター {generator_names[i]} の強制終了中にエラー: {force_close_error}")

            except Exception as cleanup_error:
                print(f"ジェネレーター {generator_names[i]} のクリーンアップ中にエラーが発生しました: {cleanup_error}")

        # 軽量なHTTP接続クリーンアップ
        await lightweight_cleanup()

        # ガベージコレクションを強制実行してリソースを解放
        import gc
        gc.collect()
        print("単一画像処理のリソースクリーンアップが完了しました")


async def process_multiple_images_streaming(image_data_list, query_text, llm_answer_checkbox_group, target_models,
                                            custom_image_prompt=None):
    """
    複数の画像を一度にVLMに送信して処理し、ストリーミング形式で回答を返す

    Args:
        image_data_list: 画像データのリスト [(base64_data, doc_id, img_id), ...]
        query_text: クエリテキスト
        llm_answer_checkbox_group: 選択されたLLMモデルのリスト
        target_models: 対象モデルのリスト
        custom_image_prompt: カスタム画像プロンプトテンプレート

    Yields:
        dict: 各モデルの部分的な回答を含む辞書
    """
    if custom_image_prompt:
        custom_image_prompt = custom_image_prompt.replace('{{query_text}}', '{query_text}')

    region = get_region()

    # 画像URLリストを作成
    image_urls = []
    for base64_data, doc_id, img_id in image_data_list:
        image_url = f"data:image/png;base64,{base64_data}"
        image_urls.append(image_url)

    print(f"複数画像処理開始: {len(image_urls)}枚の画像を一括処理")

    # 各モデルのタスクジェネレーターを作成
    async def create_model_task(model):
        llm = None  # LLMインスタンスを初期化
        try:
            if model not in llm_answer_checkbox_group:
                # 選択されていないモデルは即座に完了を通知
                yield "TASK_DONE"
                return

            print(f"\n=== 複数画像 ({len(image_urls)}枚) - {model} での処理 ===")

            if model == "meta/llama-4-maverick-17b-128e-instruct-fp8":
                llm = ChatOCIGenAI(
                    model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
                    provider="meta",
                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                    compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
                    model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
                )
            elif model == "meta/llama-4-scout-17b-16e-instruct":
                llm = ChatOCIGenAI(
                    model_id="meta.llama-4-scout-17b-16e-instruct",
                    provider="meta",
                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                    compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
                    model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
                )
            elif model == "meta/llama-3-2-90b-vision":
                llm = ChatOCIGenAI(
                    model_id="meta.llama-3.2-90b-vision-instruct",
                    provider="meta",
                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                    compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
                    model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
                )
            elif model == "openai/gpt-4o":
                load_dotenv(find_dotenv())
                llm = ChatOpenAI(
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
            elif model == "azure_openai/gpt-4o":
                load_dotenv(find_dotenv())
                llm = AzureChatOpenAI(
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
            else:
                # 未対応のモデルは即座に完了を通知
                yield "TASK_DONE"
                return

            # メッセージを作成（複数画像対応）
            prompt_text = get_image_qa_prompt(query_text, custom_image_prompt)
            prompt_text = prompt_text.replace('{{query_text}}', '{query_text}')

            # メッセージコンテンツを構築
            message_content = [{"type": "text", "text": prompt_text}]

            # 各画像を追加
            for i, image_url in enumerate(image_urls):
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })

            human_message = HumanMessage(content=message_content)
            messages = [human_message]

            # LLMに送信して回答を取得
            start_time = time.time()

            # 表示用に画像を圧縮してヘッダー情報を作成
            compressed_images_text = ""
            for i, (image_url, (_, doc_id, img_id)) in enumerate(zip(image_urls, image_data_list), 1):
                compressed_image_url = compress_image_for_display(image_url)
                compressed_images_text += f"\n\n![画像{i}]({compressed_image_url})\n"

            # ヘッダー情報を最初にyield
            header_text = f"\n\n---\n{compressed_images_text}\n**{len(image_urls)}枚の画像による回答：**\n\n"
            yield header_text

            # ストリーミングで回答を取得
            async for chunk in llm.astream(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    yield chunk.content

            end_time = time.time()
            inference_time = end_time - start_time
            print(f"\n\n推論時間: {inference_time:.2f}秒")
            print(f"=== {model} での処理完了 ===\n")

            # 推論時間を追加
            yield f"\n\n推論時間: {inference_time:.2f}秒\n\n"
            yield "TASK_DONE"

        except Exception as e:
            print(f"エラーが発生しました ({model}): {e}")
            # 表示用に画像を圧縮してエラーメッセージを作成
            compressed_images_text = ""
            for i, (image_url, (_, doc_id, img_id)) in enumerate(zip(image_urls, image_data_list), 1):
                compressed_image_url = compress_image_for_display(image_url)
                compressed_images_text += f"\n\n![画像{i}]({compressed_image_url})\n"

            error_text = f"\n\n---\n{compressed_images_text}\n**{len(image_urls)}枚の画像による回答：**\n\nエラーが発生しました: {e}\n\n"
            yield error_text
            yield "TASK_DONE"
        finally:
            # リソースクリーンアップ：LLMクライアントの接続を適切に閉じる
            await cleanup_llm_client_async(llm)
            llm = None  # 参照をクリア

    # 各モデルのジェネレーターを作成
    llama_4_maverick_gen = create_model_task("meta/llama-4-maverick-17b-128e-instruct-fp8")
    llama_4_scout_gen = create_model_task("meta/llama-4-scout-17b-16e-instruct")
    llama_3_2_90b_vision_gen = create_model_task("meta/llama-3-2-90b-vision")
    openai_gpt4o_gen = create_model_task("openai/gpt-4o")
    azure_openai_gpt4o_gen = create_model_task("azure_openai/gpt-4o")

    # 各モデルの応答を蓄積
    llama_4_maverick_response = ""
    llama_4_scout_response = ""
    llama_3_2_90b_vision_response = ""
    openai_gpt4o_response = ""
    azure_openai_gpt4o_response = ""

    # 各モデルの状態を追跡
    responses_status = ["", "", "", "", ""]

    # タイムアウト設定（最大5分）
    import asyncio
    timeout_seconds = 300
    start_time = time.time()

    try:
        while True:
            # タイムアウトチェック
            if time.time() - start_time > timeout_seconds:
                print(f"複数画像処理がタイムアウトしました（{timeout_seconds}秒）")
                break

            responses = ["", "", "", "", ""]
            generators = [llama_4_maverick_gen, llama_4_scout_gen, llama_3_2_90b_vision_gen, openai_gpt4o_gen,
                          azure_openai_gpt4o_gen]

            for i, gen in enumerate(generators):
                if responses_status[i] == "TASK_DONE":
                    continue

                try:
                    # タイムアウト付きでanextを実行
                    response = await asyncio.wait_for(anext(gen), timeout=30.0)
                    if response:
                        if response == "TASK_DONE":
                            responses_status[i] = response
                        else:
                            responses[i] = response
                except StopAsyncIteration:
                    responses_status[i] = "TASK_DONE"
                except asyncio.TimeoutError:
                    print(f"モデル {i} の処理がタイムアウトしました")
                    responses_status[i] = "TASK_DONE"
                except Exception as e:
                    print(f"モデル {i} の処理中にエラーが発生しました: {e}")
                    responses_status[i] = "TASK_DONE"

            # 応答を蓄積
            llama_4_maverick_response += responses[0]
            llama_4_scout_response += responses[1]
            llama_3_2_90b_vision_response += responses[2]
            openai_gpt4o_response += responses[3]
            azure_openai_gpt4o_response += responses[4]

            # 現在の状態をyield
            yield {
                "meta/llama-4-maverick-17b-128e-instruct-fp8": llama_4_maverick_response,
                "meta/llama-4-scout-17b-16e-instruct": llama_4_scout_response,
                "meta/llama-3-2-90b-vision": llama_3_2_90b_vision_response,
                "openai/gpt-4o": openai_gpt4o_response,
                "azure_openai/gpt-4o": azure_openai_gpt4o_response
            }

            # すべてのタスクが完了したかチェック
            if all(response_status == "TASK_DONE" for response_status in responses_status):
                print("All multiple image processing tasks completed")
                break

    finally:
        # 最終的なリソースクリーンアップ：すべてのジェネレーターを適切に閉じる
        generators = [llama_4_maverick_gen, llama_4_scout_gen, llama_3_2_90b_vision_gen, openai_gpt4o_gen,
                      azure_openai_gpt4o_gen]
        generator_names = ["llama_4_maverick", "llama_4_scout", "llama_3_2_90b_vision", "openai_gpt4o",
                           "azure_openai_gpt4o"]

        for i, gen in enumerate(generators):
            try:
                # 非同期ジェネレーターの場合
                if hasattr(gen, 'aclose'):
                    await gen.aclose()
                    print(f"複数画像処理: ジェネレーター {generator_names[i]} の非同期クローズが完了しました")
                elif hasattr(gen, 'close'):
                    gen.close()
                    print(f"複数画像処理: ジェネレーター {generator_names[i]} の同期クローズが完了しました")

                # ジェネレーター内のLLMクライアントも確実にクリーンアップ
                if hasattr(gen, 'gi_frame') and gen.gi_frame is not None:
                    # ジェネレーターがまだ実行中の場合、強制終了
                    try:
                        gen.close()
                        print(f"複数画像処理: 実行中のジェネレーター {generator_names[i]} を強制終了しました")
                    except GeneratorExit:
                        pass  # 正常な終了
                    except Exception as force_close_error:
                        print(
                            f"複数画像処理: ジェネレーター {generator_names[i]} の強制終了中にエラー: {force_close_error}")

            except Exception as cleanup_error:
                print(
                    f"複数画像処理: ジェネレーター {generator_names[i]} のクリーンアップ中にエラーが発生しました: {cleanup_error}")

        # 軽量なHTTP接続クリーンアップ
        await lightweight_cleanup()

        # ガベージコレクションを強制実行してリソースを解放
        import gc
        gc.collect()
        print("複数画像処理のリソースクリーンアップが完了しました")


async def process_image_answers_streaming(
        search_result,
        use_image,
        single_image_processing,
        llm_answer_checkbox_group,
        query_text,
        llama_4_maverick_image_answer_text,
        llama_4_scout_image_answer_text,
        llama_3_2_90b_vision_image_answer_text,
        openai_gpt4o_image_answer_text,
        azure_openai_gpt4o_image_answer_text,
        custom_image_prompt=None
):
    """
    画像を使って回答がオンの場合、検索結果から画像データを取得し、
    選択されたLLMモデルで画像処理を行い、ストリーミング形式で回答を出力する

    処理の流れ：
    1. 検索結果からdoc_idとembed_idのペアを抽出
    2. データベースから対応する画像のbase64データを取得（最大10個まで）
    3. 取得した画像を各選択されたLLMモデルで並行処理
    4. ストリーミング形式で回答を出力

    注意：パフォーマンスと応答時間を考慮し、処理する画像数は最大10個に制限されています。

    Args:
        search_result: 検索結果
        use_image: 画像を使って回答するかどうか
        llm_answer_checkbox_group: 選択されたLLMモデルのリスト
        query_text: クエリテキスト
        llama_4_maverick_image_answer_text: Llama 4 Maverick の画像回答テキスト
        llama_4_scout_image_answer_text: Llama 4 Scout の画像回答テキスト
        llama_3_2_90b_vision_image_answer_text: Llama 3.2 90B Vision の画像回答テキスト
        openai_gpt4o_image_answer_text: OpenAI GPT-4o の画像回答テキスト
        azure_openai_gpt4o_image_answer_text: Azure OpenAI GPT-4o の画像回答テキスト
        custom_image_prompt: カスタム画像プロンプトテンプレート

    Yields:
        tuple: 各モデルの更新された画像回答を含むGradio Markdownコンポーネントのタプル
    """
    print("process_image_answers_streaming() 開始...")

    # データベース接続プールの健康状態をチェック
    if not check_database_pool_health():
        print("データベース接続プールに問題があります")
        yield (
            gr.Markdown(value=llama_4_maverick_image_answer_text),
            gr.Markdown(value=llama_4_scout_image_answer_text),
            gr.Markdown(value=llama_3_2_90b_vision_image_answer_text),
            gr.Markdown(value=openai_gpt4o_image_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
        )
        return

    # 画像を使って回答がオフの場合は何もしない
    if not use_image:
        print("画像を使って回答がオフのため、base64_data取得をスキップします")
        yield (
            gr.Markdown(value=llama_4_maverick_image_answer_text),
            gr.Markdown(value=llama_4_scout_image_answer_text),
            gr.Markdown(value=llama_3_2_90b_vision_image_answer_text),
            gr.Markdown(value=openai_gpt4o_image_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
        )
        return

    # 検索結果が空の場合は何もしない
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        print("検索結果が空のため、base64_data取得をスキップします")
        yield (
            gr.Markdown(value=llama_4_maverick_image_answer_text),
            gr.Markdown(value=llama_4_scout_image_answer_text),
            gr.Markdown(value=llama_3_2_90b_vision_image_answer_text),
            gr.Markdown(value=openai_gpt4o_image_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
        )
        return

    # 指定されたLLMモデルがチェックされているかを確認
    target_models = [
        "meta/llama-4-maverick-17b-128e-instruct-fp8",
        "meta/llama-4-scout-17b-16e-instruct",
        "meta/llama-3-2-90b-vision",
        "openai/gpt-4o",
        "azure_openai/gpt-4o"
    ]

    # llm_answer_checkbox_groupに指定されたモデルのいずれかが含まれているかチェック
    has_target_model = any(model in llm_answer_checkbox_group for model in target_models)

    if not has_target_model:
        print(
            "対象のLLMモデル（llama-4-maverick, llama-4-scout, llama-3-2-90b-vision, gpt-4o）がチェックされていないため、base64_data取得をスキップします")
        yield (
            gr.Markdown(value=llama_4_maverick_image_answer_text),
            gr.Markdown(value=llama_4_scout_image_answer_text),
            gr.Markdown(value=llama_3_2_90b_vision_image_answer_text),
            gr.Markdown(value=openai_gpt4o_image_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
        )
        return

    print("条件を満たしているため、base64_dataを取得します...")

    try:
        # 検索結果からdoc_idとembed_idを取得
        doc_embed_pairs = []
        for _, row in search_result.iterrows():
            source = row['SOURCE']
            embed_id = row['EMBED_ID']
            if ':' in source:
                doc_id = source.split(':')[0]
                doc_embed_pair = (doc_id, embed_id)
                if doc_embed_pair not in doc_embed_pairs:
                    doc_embed_pairs.append(doc_embed_pair)

        if not doc_embed_pairs:
            print("検索結果からdoc_idとembed_idを取得できませんでした")
            yield (
                gr.Markdown(value=llama_4_maverick_image_answer_text),
                gr.Markdown(value=llama_4_scout_image_answer_text),
                gr.Markdown(value=llama_3_2_90b_vision_image_answer_text),
                gr.Markdown(value=openai_gpt4o_image_answer_text),
                gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
            )
            return

        print(f"取得したdoc_id, embed_idペア数: {len(doc_embed_pairs)}")
        print(f"最初の5ペア: {doc_embed_pairs[:5]}")

        # データベースからdistinct base64_dataを取得
        try:
            with pool.acquire() as conn:
                with conn.cursor() as cursor:
                    # まず_image_embeddingテーブルからimg_idを取得
                    embed_where_conditions = []
                    for doc_id, embed_id in doc_embed_pairs:
                        embed_where_conditions.append(f"(doc_id = '{doc_id}' AND embed_id = {embed_id})")

                    embed_where_clause = " OR ".join(embed_where_conditions)

                    # _image_embeddingテーブルからimg_idを取得
                    get_img_ids_sql = f"""
                        SELECT doc_id, embed_id, img_id
                        FROM (
                            SELECT
                                doc_id,
                                embed_id,
                                img_id,
                                ROW_NUMBER() OVER (PARTITION BY doc_id, img_id ORDER BY embed_id ASC) as rn
                            FROM {DEFAULT_COLLECTION_NAME}_image_embedding
                            WHERE ({embed_where_clause})
                            AND img_id IS NOT NULL
                        ) subquery
                        WHERE rn = 1
                        ORDER BY doc_id, img_id ASC
                    """

                    print(f"img_id取得SQL: {get_img_ids_sql}")
                    cursor.execute(get_img_ids_sql)

                    doc_img_pairs = []
                    for row in cursor:
                        doc_id = row[0]
                        embed_id = row[1]
                        img_id = row[2]
                        doc_img_pair = (doc_id, img_id)
                        if doc_img_pair not in doc_img_pairs:
                            doc_img_pairs.append(doc_img_pair)
                            print(f"見つかったペア: doc_id={doc_id}, embed_id={embed_id}, img_id={img_id}")

                    if not doc_img_pairs:
                        print("_image_embeddingテーブルからimg_idを取得できませんでした")
                        yield (
                            gr.Markdown(value=llama_4_maverick_image_answer_text),
                            gr.Markdown(value=llama_4_scout_image_answer_text),
                            gr.Markdown(value=llama_3_2_90b_vision_image_answer_text),
                            gr.Markdown(value=openai_gpt4o_image_answer_text),
                            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
                        )
                        return

                    print(f"取得したdoc_id, img_idペア数: {len(doc_img_pairs)}")

                    # 次に_imageテーブルからbase64_dataを取得
                    img_where_conditions = []
                    for doc_id, img_id in doc_img_pairs:
                        img_where_conditions.append(f"(doc_id = '{doc_id}' AND img_id = {img_id})")

                    img_where_clause = " OR ".join(img_where_conditions)

                    # base64画像データを取得（最大10件まで）
                    select_sql = f"""
                    SELECT base64_data, doc_id, img_id
                    FROM {DEFAULT_COLLECTION_NAME}_image
                    WHERE ({img_where_clause})
                    AND base64_data IS NOT NULL
                    FETCH FIRST 10 ROWS ONLY
                    """

                    print(f"実行するSQL: {select_sql}")
                    cursor.execute(select_sql)

                    base64_data_list = []
                    for row in cursor:
                        if row[0] is not None:
                            try:
                                # CLOBオブジェクトの読み取り
                                if hasattr(row[0], 'read'):
                                    base64_string = row[0].read()
                                    # 10MB制限チェック
                                    if len(base64_string) > 10 * 1024 * 1024:
                                        print(f"Base64データが大きすぎます（{len(base64_string)}文字）、スキップします")
                                        continue
                                else:
                                    base64_string = str(row[0])

                                base64_data_list.append((base64_string, row[1], row[2]))

                            except Exception as e:
                                print(f"CLOB読み取りエラー: {e}")
                                continue

                    print(f"取得したbase64_dataの数: {len(base64_data_list)}")

                    # 初期化：現在の画像回答テキストを保持（累積用）
                    accumulated_llama_4_maverick_text = llama_4_maverick_image_answer_text
                    accumulated_llama_4_scout_text = llama_4_scout_image_answer_text
                    accumulated_llama_3_2_90b_vision_text = llama_3_2_90b_vision_image_answer_text
                    accumulated_openai_gpt4o_text = openai_gpt4o_image_answer_text
                    accumulated_azure_openai_gpt4o_text = azure_openai_gpt4o_image_answer_text

                    # 処理方式を選択：1枚ずつ処理 vs 一括処理
                    if base64_data_list:
                        if single_image_processing:
                            # 1枚ずつ処理モード
                            print(f"単一画像処理開始: {len(base64_data_list)}枚の画像を1枚ずつ処理中...")

                            # 各base64_dataに対してLLMで処理
                            for i, (base64_data, doc_id, img_id) in enumerate(base64_data_list, 1):
                                print(f"画像 {i} (doc_id: {doc_id}, img_id: {img_id}) を処理中...")

                                # base64データをdata:image/png;base64,{base64_data}形式に変換
                                image_url = f"data:image/png;base64,{base64_data}"

                                # 各モデルの現在の画像に対する回答を保持
                                current_image_llama_4_maverick = ""
                                current_image_llama_4_scout = ""
                                current_image_llama_3_2_90b_vision = ""
                                current_image_openai_gpt4o = ""
                                current_image_azure_openai_gpt4o = ""

                                # 選択されたLLMモデルに対して処理を実行し、結果をストリーミングで取得
                                async for llm_results in process_single_image_streaming(
                                        image_url,
                                        query_text,
                                        llm_answer_checkbox_group,
                                        target_models,
                                        i,
                                        doc_id,
                                        img_id,
                                        custom_image_prompt
                                ):
                                    # 各LLMの結果を現在の画像の回答として更新
                                    if "meta/llama-4-maverick-17b-128e-instruct-fp8" in llm_results:
                                        current_image_llama_4_maverick = llm_results[
                                            "meta/llama-4-maverick-17b-128e-instruct-fp8"]

                                    if "meta/llama-4-scout-17b-16e-instruct" in llm_results:
                                        current_image_llama_4_scout = llm_results["meta/llama-4-scout-17b-16e-instruct"]

                                    if "meta/llama-3-2-90b-vision" in llm_results:
                                        current_image_llama_3_2_90b_vision = llm_results["meta/llama-3-2-90b-vision"]

                                    if "openai/gpt-4o" in llm_results:
                                        current_image_openai_gpt4o = llm_results["openai/gpt-4o"]

                                    if "azure_openai/gpt-4o" in llm_results:
                                        current_image_azure_openai_gpt4o = llm_results["azure_openai/gpt-4o"]

                                    # 累積テキストと現在の画像の回答を結合して表示
                                    current_llama_4_maverick_text = accumulated_llama_4_maverick_text + current_image_llama_4_maverick
                                    current_llama_4_scout_text = accumulated_llama_4_scout_text + current_image_llama_4_scout
                                    current_llama_3_2_90b_vision_text = accumulated_llama_3_2_90b_vision_text + current_image_llama_3_2_90b_vision
                                    current_openai_gpt4o_text = accumulated_openai_gpt4o_text + current_image_openai_gpt4o
                                    current_azure_openai_gpt4o_text = accumulated_azure_openai_gpt4o_text + current_image_azure_openai_gpt4o

                                    # 更新された画像回答結果をyield
                                    yield (
                                        gr.Markdown(value=current_llama_4_maverick_text),
                                        gr.Markdown(value=current_llama_4_scout_text),
                                        gr.Markdown(value=current_llama_3_2_90b_vision_text),
                                        gr.Markdown(value=current_openai_gpt4o_text),
                                        gr.Markdown(value=current_azure_openai_gpt4o_text)
                                    )

                                # 現在の画像の処理が完了したら、累積テキストに追加
                                accumulated_llama_4_maverick_text += current_image_llama_4_maverick
                                accumulated_llama_4_scout_text += current_image_llama_4_scout
                                accumulated_llama_3_2_90b_vision_text += current_image_llama_3_2_90b_vision
                                accumulated_openai_gpt4o_text += current_image_openai_gpt4o
                                accumulated_azure_openai_gpt4o_text += current_image_azure_openai_gpt4o
                        else:
                            # 一括処理モード
                            print(f"複数画像一括処理開始: {len(base64_data_list)}枚の画像を一括処理中...")

                            # 各モデルの回答を保持
                            current_llama_4_maverick = ""
                            current_llama_4_scout = ""
                            current_llama_3_2_90b_vision = ""
                            current_openai_gpt4o = ""
                            current_azure_openai_gpt4o = ""

                            # 複数画像を一括処理
                            async for llm_results in process_multiple_images_streaming(
                                    base64_data_list,
                                    query_text,
                                    llm_answer_checkbox_group,
                                    target_models,
                                    custom_image_prompt
                            ):
                                # 各LLMの結果を複数画像の回答として更新
                                if "meta/llama-4-maverick-17b-128e-instruct-fp8" in llm_results:
                                    current_llama_4_maverick = llm_results[
                                        "meta/llama-4-maverick-17b-128e-instruct-fp8"]

                                if "meta/llama-4-scout-17b-16e-instruct" in llm_results:
                                    current_llama_4_scout = llm_results["meta/llama-4-scout-17b-16e-instruct"]

                                if "meta/llama-3-2-90b-vision" in llm_results:
                                    current_llama_3_2_90b_vision = llm_results["meta/llama-3-2-90b-vision"]

                                if "openai/gpt-4o" in llm_results:
                                    current_openai_gpt4o = llm_results["openai/gpt-4o"]

                                if "azure_openai/gpt-4o" in llm_results:
                                    current_azure_openai_gpt4o = llm_results["azure_openai/gpt-4o"]

                                # 累積テキストと複数画像の回答を結合して表示
                                current_llama_4_maverick_text = accumulated_llama_4_maverick_text + current_llama_4_maverick
                                current_llama_4_scout_text = accumulated_llama_4_scout_text + current_llama_4_scout
                                current_llama_3_2_90b_vision_text = accumulated_llama_3_2_90b_vision_text + current_llama_3_2_90b_vision
                                current_openai_gpt4o_text = accumulated_openai_gpt4o_text + current_openai_gpt4o
                                current_azure_openai_gpt4o_text = accumulated_azure_openai_gpt4o_text + current_azure_openai_gpt4o

                                # 更新された画像回答結果をyield
                                yield (
                                    gr.Markdown(value=current_llama_4_maverick_text),
                                    gr.Markdown(value=current_llama_4_scout_text),
                                    gr.Markdown(value=current_llama_3_2_90b_vision_text),
                                    gr.Markdown(value=current_openai_gpt4o_text),
                                    gr.Markdown(value=current_azure_openai_gpt4o_text)
                                )

        except Exception as db_e:
            print(f"データベース操作中にエラーが発生しました: {db_e}")
            # データベースエラー時も現在の状態をyield
            yield (
                gr.Markdown(value=llama_4_maverick_image_answer_text),
                gr.Markdown(value=llama_4_scout_image_answer_text),
                gr.Markdown(value=llama_3_2_90b_vision_image_answer_text),
                gr.Markdown(value=openai_gpt4o_image_answer_text),
                gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
            )
            return

    except Exception as e:
        print(f"base64_data取得中にエラーが発生しました: {e}")
        # エラー時も現在の状態をyield
        yield (
            gr.Markdown(value=llama_4_maverick_image_answer_text),
            gr.Markdown(value=llama_4_scout_image_answer_text),
            gr.Markdown(value=llama_3_2_90b_vision_image_answer_text),
            gr.Markdown(value=openai_gpt4o_image_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
        )

    finally:
        # 画像処理完了後の軽量なリソースクリーンアップ
        try:
            await lightweight_cleanup()
        except Exception as cleanup_error:
            print(f"軽量リソースクリーンアップ中にエラー: {cleanup_error}")

    print("process_image_answers_streaming() 完了")


async def eval_by_ragas(
        query_text,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        search_result,
        llm_answer_checkbox_group,
        llm_evaluation_checkbox,
        system_text,
        standard_answer_text,
        xai_grok_3_response,
        command_a_response,
        llama_4_maverick_response,
        llama_4_scout_response,
        llama_3_3_70b_response,
        llama_3_2_90b_vision_response,
        openai_gpt4o_response,
        openai_gpt4_response,
        azure_openai_gpt4o_response,
        azure_openai_gpt4_response
):
    has_error = False
    if not query_text:
        has_error = True
        # gr.Warning("クエリを入力してください")
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        has_error = True
        # gr.Warning("ドキュメントを選択してください")
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        has_error = True
        # gr.Warning("検索結果が見つかりませんでした。設定もしくはクエリを変更して再度ご確認ください。")
    if llm_evaluation_checkbox and (not llm_answer_checkbox_group or llm_answer_checkbox_group == [""]):
        has_error = True
        gr.Warning("LLM 評価をオンにする場合、少なくとも1つのLLM モデルを選択してください")
    if llm_evaluation_checkbox and not system_text:
        has_error = True
        gr.Warning("LLM 評価をオンにする場合、LLM 評価のシステム・メッセージを入力してください")
    if llm_evaluation_checkbox and not standard_answer_text:
        has_error = True
        gr.Warning("LLM 評価をオンにする場合、LLM 評価の標準回答を入力してください")
    if has_error:
        yield (
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value="")
        )
        return

    def remove_last_line(text):
        if text:
            lines = text.splitlines()
            if lines[-1].startswith("推論時間"):
                lines.pop()
            return '\n'.join(lines)
        else:
            return text

    if standard_answer_text:
        standard_answer_text = standard_answer_text.strip()
    else:
        standard_answer_text = "入力されていません。"
    print(f"{llm_evaluation_checkbox=}")
    if not llm_evaluation_checkbox:
        yield (
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value="")
        )
    else:
        xai_grok_3_checkbox = False
        command_a_checkbox = False
        llama_4_maverick_checkbox = False
        llama_4_scout_checkbox = False
        llama_3_3_70b_checkbox = False
        llama_3_2_90b_vision_checkbox = False
        openai_gpt4o_checkbox = False
        openai_gpt4_checkbox = False
        azure_openai_gpt4o_checkbox = False
        azure_openai_gpt4_checkbox = False
        if "xai/grok-3" in llm_answer_checkbox_group:
            xai_grok_3_checkbox = True
        if "cohere/command-a" in llm_answer_checkbox_group:
            command_a_checkbox = True
        if "meta/llama-4-maverick-17b-128e-instruct-fp8" in llm_answer_checkbox_group:
            llama_4_maverick_checkbox = True
        if "meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox_group:
            llama_4_scout_checkbox = True
        if "meta/llama-3-3-70b" in llm_answer_checkbox_group:
            llama_3_3_70b_checkbox = True
        if "meta/llama-3-2-90b-vision" in llm_answer_checkbox_group:
            llama_3_2_90b_vision_checkbox = True
        if "openai/gpt-4o" in llm_answer_checkbox_group:
            openai_gpt4o_checkbox = True
        if "openai/gpt-4" in llm_answer_checkbox_group:
            openai_gpt4_checkbox = True
        if "azure_openai/gpt-4o" in llm_answer_checkbox_group:
            azure_openai_gpt4o_checkbox = True
        if "azure_openai/gpt-4" in llm_answer_checkbox_group:
            azure_openai_gpt4_checkbox = True

        xai_grok_3_response = remove_last_line(xai_grok_3_response)
        command_a_response = remove_last_line(command_a_response)
        llama_4_maverick_response = remove_last_line(llama_4_maverick_response)
        llama_4_scout_response = remove_last_line(llama_4_scout_response)
        llama_3_3_70b_response = remove_last_line(llama_3_3_70b_response)
        llama_3_2_90b_vision_response = remove_last_line(llama_3_2_90b_vision_response)
        openai_gpt4o_response = remove_last_line(openai_gpt4o_response)
        openai_gpt4_response = remove_last_line(openai_gpt4_response)
        azure_openai_gpt4o_response = remove_last_line(azure_openai_gpt4o_response)
        azure_openai_gpt4_response = remove_last_line(azure_openai_gpt4_response)

        xai_grok_3_user_text = f"""
-標準回答-
 {standard_answer_text}

-与えられた回答-
 {xai_grok_3_response}

-出力-\n　"""

        command_a_user_text = f"""
-標準回答-
 {standard_answer_text}

-与えられた回答-
 {command_a_response}

-出力-\n　"""



        llama_4_maverick_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{llama_4_maverick_response}

-出力-\n　"""

        llama_4_scout_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{llama_4_scout_response}

-出力-\n　"""

        llama_3_3_70b_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{llama_3_3_70b_response}

-出力-\n　"""

        llama_3_2_90b_vision_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{llama_3_2_90b_vision_response}

-出力-\n　"""

        openai_gpt4o_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{openai_gpt4o_response}

-出力-\n　"""

        openai_gpt4_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{openai_gpt4_response}

-出力-\n　"""

        azure_openai_gpt4o_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{azure_openai_gpt4o_response}

-出力-\n　"""

        azure_openai_gpt4_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{azure_openai_gpt4_response}

-出力-\n　"""

        eval_xai_grok_3_response = ""
        eval_command_a_response = ""
        eval_llama_4_maverick_response = ""
        eval_llama_4_scout_response = ""
        eval_llama_3_3_70b_response = ""
        eval_llama_3_2_90b_vision_response = ""
        eval_openai_gpt4o_response = ""
        eval_openai_gpt4_response = ""
        eval_azure_openai_gpt4o_response = ""
        eval_azure_openai_gpt4_response = ""

        async for xai_grok_3, command_a, llama_4_maverick, llama_4_scout, llama_3_3_70b, llama_3_2_90b_vision, gpt4o, gpt4, azure_gpt4o, azure_gpt4 in chat(
                system_text,
                xai_grok_3_user_text,
                command_a_user_text,
                None,
                llama_4_maverick_user_text,
                None,
                llama_4_scout_user_text,
                llama_3_3_70b_user_text,
                None,
                llama_3_2_90b_vision_user_text,
                openai_gpt4o_user_text,
                openai_gpt4_user_text,
                azure_openai_gpt4o_user_text,
                azure_openai_gpt4_user_text,
                xai_grok_3_checkbox,
                command_a_checkbox,
                llama_4_maverick_checkbox,
                llama_4_scout_checkbox,
                llama_3_3_70b_checkbox,
                llama_3_2_90b_vision_checkbox,
                openai_gpt4o_checkbox,
                openai_gpt4_checkbox,
                azure_openai_gpt4o_checkbox,
                azure_openai_gpt4_checkbox
        ):
            eval_xai_grok_3_response += xai_grok_3
            eval_command_a_response += command_a
            eval_llama_4_maverick_response += llama_4_maverick
            eval_llama_4_scout_response += llama_4_scout
            eval_llama_3_3_70b_response += llama_3_3_70b
            eval_llama_3_2_90b_vision_response += llama_3_2_90b_vision
            eval_openai_gpt4o_response += gpt4o
            eval_openai_gpt4_response += gpt4
            eval_azure_openai_gpt4o_response += azure_gpt4o
            eval_azure_openai_gpt4_response += azure_gpt4
            yield (
                gr.Markdown(value=eval_xai_grok_3_response),
                gr.Markdown(value=eval_command_a_response),
                gr.Markdown(value=eval_llama_4_maverick_response),
                gr.Markdown(value=eval_llama_4_scout_response),
                gr.Markdown(value=eval_llama_3_3_70b_response),
                gr.Markdown(value=eval_llama_3_2_90b_vision_response),
                gr.Markdown(value=eval_openai_gpt4o_response),
                gr.Markdown(value=eval_openai_gpt4_response),
                gr.Markdown(value=eval_azure_openai_gpt4o_response),
                gr.Markdown(value=eval_azure_openai_gpt4_response)
            )


def generate_download_file(
        search_result,
        llm_answer_checkbox_group,
        include_citation,
        llm_evaluation_checkbox,
        query_text,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        standard_answer_text,
        xai_grok_3_response,
        command_a_response,
        llama_4_maverick_response,
        llama_4_scout_response,
        llama_3_3_70b_response,
        llama_3_2_90b_vision_response,
        openai_gpt4o_response,
        openai_gpt4_response,
        azure_openai_gpt4o_response,
        azure_openai_gpt4_response,
        xai_grok_3_evaluation,
        command_a_evaluation,
        llama_4_maverick_evaluation,
        llama_4_scout_evaluation,
        llama_3_3_70b_evaluation,
        llama_3_2_90b_vision_evaluation,
        openai_gpt4o_evaluation,
        openai_gpt4_evaluation,
        azure_openai_gpt4o_evaluation,
        azure_openai_gpt4_evaluation
):
    if not query_text:
        return gr.DownloadButton(value=None, visible=False)
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        return gr.DownloadButton(value=None, visible=False)
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        return gr.DownloadButton(value=None, visible=False)
    # サンプルDataFrameを作成
    if llm_evaluation_checkbox:
        standard_answer_text = standard_answer_text
    else:
        standard_answer_text = ""
    df1 = pd.DataFrame({'クエリ': [query_text], '標準回答': [standard_answer_text]})

    df2 = search_result

    if "xai/grok-3" in llm_answer_checkbox_group:
        xai_grok_3_response = xai_grok_3_response
        xai_grok_3_referenced_contexts = ""
        if include_citation:
            xai_grok_3_response, xai_grok_3_referenced_contexts = extract_citation(xai_grok_3_response)
        if llm_evaluation_checkbox:
            xai_grok_3_evaluation = xai_grok_3_evaluation
        else:
            xai_grok_3_evaluation = ""
    else:
        xai_grok_3_response = ""
        xai_grok_3_evaluation = ""
        xai_grok_3_referenced_contexts = ""

    if "cohere/command-a" in llm_answer_checkbox_group:
        command_a_response = command_a_response
        command_a_referenced_contexts = ""
        if include_citation:
            command_a_response, command_a_referenced_contexts = extract_citation(command_a_response)
        if llm_evaluation_checkbox:
            command_a_evaluation = command_a_evaluation
        else:
            command_a_evaluation = ""
    else:
        command_a_response = ""
        command_a_evaluation = ""
        command_a_referenced_contexts = ""



    if "meta/llama-4-maverick-17b-128e-instruct-fp8" in llm_answer_checkbox_group:
        llama_4_maverick_response = llama_4_maverick_response
        llama_4_maverick_referenced_contexts = ""
        if include_citation:
            llama_4_maverick_response, llama_4_maverick_referenced_contexts = extract_citation(
                llama_4_maverick_response)
        if llm_evaluation_checkbox:
            llama_4_maverick_evaluation = llama_4_maverick_evaluation
        else:
            llama_4_maverick_evaluation = ""
    else:
        llama_4_maverick_response = ""
        llama_4_maverick_evaluation = ""
        llama_4_maverick_referenced_contexts = ""

    if "meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox_group:
        llama_4_scout_response = llama_4_scout_response
        llama_4_scout_referenced_contexts = ""
        if include_citation:
            llama_4_scout_response, llama_4_scout_referenced_contexts = extract_citation(llama_4_scout_response)
        if llm_evaluation_checkbox:
            llama_4_scout_evaluation = llama_4_scout_evaluation
        else:
            llama_4_scout_evaluation = ""
    else:
        llama_4_scout_response = ""
        llama_4_scout_evaluation = ""
        llama_4_scout_referenced_contexts = ""

    if "meta/llama-3-3-70b" in llm_answer_checkbox_group:
        llama_3_3_70b_response = llama_3_3_70b_response
        llama_3_3_70b_referenced_contexts = ""
        if include_citation:
            llama_3_3_70b_response, llama_3_3_70b_referenced_contexts = extract_citation(llama_3_3_70b_response)
        if llm_evaluation_checkbox:
            llama_3_3_70b_evaluation = llama_3_3_70b_evaluation
        else:
            llama_3_3_70b_evaluation = ""
    else:
        llama_3_3_70b_response = ""
        llama_3_3_70b_evaluation = ""
        llama_3_3_70b_referenced_contexts = ""

    if "meta/llama-3-2-90b-vision" in llm_answer_checkbox_group:
        llama_3_2_90b_vision_response = llama_3_2_90b_vision_response
        llama_3_2_90b_vision_referenced_contexts = ""
        if include_citation:
            llama_3_2_90b_vision_response, llama_3_2_90b_vision_referenced_contexts = extract_citation(
                llama_3_2_90b_vision_response)
        if llm_evaluation_checkbox:
            llama_3_2_90b_vision_evaluation = llama_3_2_90b_vision_evaluation
        else:
            llama_3_2_90b_vision_evaluation = ""
    else:
        llama_3_2_90b_vision_response = ""
        llama_3_2_90b_vision_evaluation = ""
        llama_3_2_90b_vision_referenced_contexts = ""

    if "openai/gpt-4o" in llm_answer_checkbox_group:
        openai_gpt4o_response = openai_gpt4o_response
        openai_gpt4o_referenced_contexts = ""
        if include_citation:
            openai_gpt4o_response, openai_gpt4o_referenced_contexts = extract_citation(openai_gpt4o_response)
        if llm_evaluation_checkbox:
            openai_gpt4o_evaluation = openai_gpt4o_evaluation
        else:
            openai_gpt4o_evaluation = ""
    else:
        openai_gpt4o_response = ""
        openai_gpt4o_evaluation = ""
        openai_gpt4o_referenced_contexts = ""

    if "openai/gpt-4" in llm_answer_checkbox_group:
        openai_gpt4_response = openai_gpt4_response
        openai_gpt4_referenced_contexts = ""
        if include_citation:
            openai_gpt4_response, openai_gpt4_referenced_contexts = extract_citation(openai_gpt4_response)
        if llm_evaluation_checkbox:
            openai_gpt4_evaluation = openai_gpt4_evaluation
        else:
            openai_gpt4_evaluation = ""
    else:
        openai_gpt4_response = ""
        openai_gpt4_evaluation = ""
        openai_gpt4_referenced_contexts = ""

    if "azure_openai/gpt-4o" in llm_answer_checkbox_group:
        azure_openai_gpt4o_response = azure_openai_gpt4o_response
        azure_openai_gpt4o_referenced_contexts = ""
        if include_citation:
            azure_openai_gpt4o_response, azure_openai_gpt4o_referenced_contexts = extract_citation(
                azure_openai_gpt4o_response)
        if llm_evaluation_checkbox:
            azure_openai_gpt4o_evaluation = azure_openai_gpt4o_evaluation
        else:
            azure_openai_gpt4o_evaluation = ""
    else:
        azure_openai_gpt4o_response = ""
        azure_openai_gpt4o_evaluation = ""
        azure_openai_gpt4o_referenced_contexts = ""

    if "azure_openai/gpt-4" in llm_answer_checkbox_group:
        azure_openai_gpt4_response = azure_openai_gpt4_response
        azure_openai_gpt4_referenced_contexts = ""
        if include_citation:
            azure_openai_gpt4_response, azure_openai_gpt4_referenced_contexts = extract_citation(
                azure_openai_gpt4_response)
        if llm_evaluation_checkbox:
            azure_openai_gpt4_evaluation = azure_openai_gpt4_evaluation
        else:
            azure_openai_gpt4_evaluation = ""
    else:
        azure_openai_gpt4_response = ""
        azure_openai_gpt4_evaluation = ""
        azure_openai_gpt4_referenced_contexts = ""

    df3 = pd.DataFrame(
        {
            'LLM モデル':
                [
                    "xai/grok-3",
                    "cohere/command-a",
                    "meta/llama-4-maverick-17b-128e-instruct-fp8",
                    "meta/llama-4-scout-17b-16e-instruct",
                    "meta/llama-3-3-70b",
                    "meta/llama-3-2-90b-vision",
                    "openai/gpt-4o",
                    "openai/gpt-4",
                    "azure_openai/gpt-4o",
                    "azure_openai/gpt-4"
                ],
            'LLM メッセージ': [
                xai_grok_3_response,
                command_a_response,
                llama_4_maverick_response,
                llama_4_scout_response,
                llama_3_3_70b_response,
                llama_3_2_90b_vision_response,
                openai_gpt4o_response,
                openai_gpt4_response,
                azure_openai_gpt4o_response,
                azure_openai_gpt4_response
            ],
            '引用 Contexts': [
                xai_grok_3_referenced_contexts,
                command_a_referenced_contexts,
                llama_4_maverick_referenced_contexts,
                llama_4_scout_referenced_contexts,
                llama_3_3_70b_referenced_contexts,
                llama_3_2_90b_vision_referenced_contexts,
                openai_gpt4o_referenced_contexts,
                openai_gpt4_referenced_contexts,
                azure_openai_gpt4o_referenced_contexts,
                azure_openai_gpt4_referenced_contexts
            ],
            'LLM 評価結果': [
                xai_grok_3_evaluation,
                command_a_evaluation,
                llama_4_maverick_evaluation,
                llama_4_scout_evaluation,
                llama_3_3_70b_evaluation,
                llama_3_2_90b_vision_evaluation,
                openai_gpt4o_evaluation,
                openai_gpt4_evaluation,
                azure_openai_gpt4o_evaluation,
                azure_openai_gpt4_evaluation
            ]
        }
    )

    # ファイルパスを定義
    filepath = '/tmp/query_result.xlsx'

    # ExcelWriterを使用して複数のDataFrameを異なるシートに書き込み
    with pd.ExcelWriter(filepath) as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
        df3.to_excel(writer, sheet_name='Sheet3', index=False)

    print(f"Excelファイルが {filepath} に保存されました")
    return gr.DownloadButton(value=filepath, visible=True)


def generate_eval_result_file():
    print("in generate_eval_result_file() start...")

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            select_sql = """
                         SELECT r.query_id,
                                r.query,
                                r.standard_answer,
                                r.sql,
                                f.llm_name,
                                f.llm_answer,
                                f.ragas_evaluation_result,
                                f.human_evaluation_result,
                                f.user_comment,
                                TO_CHAR(r.created_date, 'YYYY-MM-DD HH24:MI:SS') AS created_date
                         FROM RAG_QA_RESULT r
                                  JOIN
                              RAG_QA_FEEDBACK f
                              ON
                                  r.query_id = f.query_id \
                         """

            cursor.execute(select_sql)

            # 列名を取得
            columns = [col[0] for col in cursor.description]

            # データを取得
            data = cursor.fetchall()

            print(f"{columns=}")

            # データをDataFrameに変換
            result_df = pd.DataFrame(data, columns=columns)

            print(f"{result_df=}")

            # 列名を日文に変更
            result_df.rename(columns={
                'QUERY_ID': 'クエリID',
                'QUERY': 'クエリ',
                'STANDARD_ANSWER': '標準回答',
                'SQL': '使用されたSQL',
                'LLM_NAME': 'LLM モデル',
                'LLM_ANSWER': 'LLM メッセージ',
                'RAGAS_EVALUATION_RESULT': 'LLM 評価結果',
                'HUMAN_EVALUATION_RESULT': 'Human 評価結果',
                'USER_COMMENT': 'Human コメント',
                'CREATED_DATE': '作成日時'
            }, inplace=True)

            print(f"{result_df=}")

            # 必要に応じてcreated_date列をdatetime型に変換
            result_df['作成日時'] = pd.to_datetime(result_df['作成日時'], format='%Y-%m-%d %H:%M:%S')

            # ファイルパスを定義
            filepath = '/tmp/evaluation_result.xlsx'

            # ExcelWriterを使用して複数のDataFrameを異なるシートに書き込み
            with pd.ExcelWriter(filepath) as writer:
                result_df.to_excel(writer, sheet_name='Sheet1', index=False)

            print(f"Excelファイルが {filepath} に保存されました")
            gr.Info("評価レポートの生成が完了しました")
            return gr.DownloadButton(value=filepath, visible=True)


def set_query_id_state():
    print("in set_query_id_state() start...")
    return generate_unique_id("query_")


def insert_query_result(
        search_result,
        query_id,
        query,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        sql,
        llm_answer_checkbox_group,
        llm_evaluation_checkbox,
        standard_answer_text,
        xai_grok_3_response,
        command_a_response,

        llama_4_maverick_response,
        llama_4_scout_response,
        llama_3_3_70b_response,
        llama_3_2_90b_vision_response,
        openai_gpt4o_response,
        openai_gpt4_response,
        azure_openai_gpt4o_response,
        azure_openai_gpt4_response,
        xai_grok_3_evaluation,
        command_a_evaluation,

        llama_4_maverick_evaluation,
        llama_4_scout_evaluation,
        llama_3_3_70b_evaluation,
        llama_3_2_90b_vision_evaluation,
        openai_gpt4o_evaluation,
        openai_gpt4_evaluation,
        azure_openai_gpt4o_evaluation,
        azure_openai_gpt4_evaluation
):
    print("in insert_query_result() start...")
    if not query:
        return
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        return
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        return
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            # レコードが存在しない場合、挿入操作を実行
            insert_sql = """
                         INSERT INTO RAG_QA_RESULT (query_id,
                                                    query,
                                                    standard_answer,
                                                    sql)
                         VALUES (:1,
                                 :2,
                                 :3,
                                 :4) \
                         """
            cursor.setinputsizes(None, None, None, oracledb.CLOB)
            cursor.execute(
                insert_sql,
                [
                    query_id,
                    query,
                    standard_answer_text,
                    sql
                ]
            )

            if "xai/grok-3" in llm_answer_checkbox_group:
                xai_grok_3_response = xai_grok_3_response
                if llm_evaluation_checkbox:
                    xai_grok_3_evaluation = xai_grok_3_evaluation
                else:
                    xai_grok_3_evaluation = ""

                insert_sql = """
                             INSERT INTO RAG_QA_FEEDBACK (query_id,
                                                          llm_name,
                                                          llm_answer,
                                                          ragas_evaluation_result)
                             VALUES (:1,
                                     :2,
                                     :3,
                                     :4) \
                             """
                cursor.setinputsizes(None, None, oracledb.CLOB, oracledb.CLOB)
                cursor.execute(
                    insert_sql,
                    [
                        query_id,
                        "xai/grok-3",
                        xai_grok_3_response,
                        xai_grok_3_evaluation
                    ]
                )

            if "cohere/command-a" in llm_answer_checkbox_group:
                command_a_response = command_a_response
                if llm_evaluation_checkbox:
                    command_a_evaluation = command_a_evaluation
                else:
                    command_a_evaluation = ""

                insert_sql = """
                             INSERT INTO RAG_QA_FEEDBACK (query_id,
                                                          llm_name,
                                                          llm_answer,
                                                          ragas_evaluation_result)
                             VALUES (:1,
                                     :2,
                                     :3,
                                     :4) \
                             """
                cursor.setinputsizes(None, None, oracledb.CLOB, oracledb.CLOB)
                cursor.execute(
                    insert_sql,
                    [
                        query_id,
                        "cohere/command-a",
                        command_a_response,
                        command_a_evaluation
                    ]
                )


            if "meta/llama-4-maverick-17b-128e-instruct-fp8" in llm_answer_checkbox_group:
                llama_4_maverick_response = llama_4_maverick_response
                if llm_evaluation_checkbox:
                    llama_4_maverick_evaluation = llama_4_maverick_evaluation
                else:
                    llama_4_maverick_evaluation = ""

                insert_sql = """
                             INSERT INTO RAG_QA_FEEDBACK (query_id,
                                                          llm_name,
                                                          llm_answer,
                                                          ragas_evaluation_result)
                             VALUES (:1,
                                     :2,
                                     :3,
                                     :4) \
                             """
                cursor.setinputsizes(None, None, oracledb.CLOB, oracledb.CLOB)
                cursor.execute(
                    insert_sql,
                    [
                        query_id,
                        "meta/llama-4-maverick-17b-128e-instruct-fp8",
                        llama_4_maverick_response,
                        llama_4_maverick_evaluation
                    ]
                )

            if "meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox_group:
                llama_4_scout_response = llama_4_scout_response
                if llm_evaluation_checkbox:
                    llama_4_scout_evaluation = llama_4_scout_evaluation
                else:
                    llama_4_scout_evaluation = ""

                insert_sql = """
                             INSERT INTO RAG_QA_FEEDBACK (query_id,
                                                          llm_name,
                                                          llm_answer,
                                                          ragas_evaluation_result)
                             VALUES (:1,
                                     :2,
                                     :3,
                                     :4) \
                             """
                cursor.setinputsizes(None, None, oracledb.CLOB, oracledb.CLOB)
                cursor.execute(
                    insert_sql,
                    [
                        query_id,
                        "meta/llama-4-scout-17b-16e-instruct",
                        llama_4_scout_response,
                        llama_4_scout_evaluation
                    ]
                )

            if "meta/llama-3-3-70b" in llm_answer_checkbox_group:
                llama_3_3_70b_response = llama_3_3_70b_response
                if llm_evaluation_checkbox:
                    llama_3_3_70b_evaluation = llama_3_3_70b_evaluation
                else:
                    llama_3_3_70b_evaluation = ""

                insert_sql = """
                             INSERT INTO RAG_QA_FEEDBACK (query_id,
                                                          llm_name,
                                                          llm_answer,
                                                          ragas_evaluation_result)
                             VALUES (:1,
                                     :2,
                                     :3,
                                     :4) \
                             """
                cursor.setinputsizes(None, None, oracledb.CLOB, oracledb.CLOB)
                cursor.execute(
                    insert_sql,
                    [
                        query_id,
                        "meta/llama-3-3-70b",
                        llama_3_3_70b_response,
                        llama_3_3_70b_evaluation
                    ]
                )

            if "meta/llama-3-2-90b-vision" in llm_answer_checkbox_group:
                llama_3_2_90b_vision_response = llama_3_2_90b_vision_response
                if llm_evaluation_checkbox:
                    llama_3_2_90b_vision_evaluation = llama_3_2_90b_vision_evaluation
                else:
                    llama_3_2_90b_vision_evaluation = ""

                insert_sql = """
                             INSERT INTO RAG_QA_FEEDBACK (query_id,
                                                          llm_name,
                                                          llm_answer,
                                                          ragas_evaluation_result)
                             VALUES (:1,
                                     :2,
                                     :3,
                                     :4) \
                             """
                cursor.setinputsizes(None, None, oracledb.CLOB, oracledb.CLOB)
                cursor.execute(
                    insert_sql,
                    [
                        query_id,
                        "meta/llama-3-2-90b-vision",
                        llama_3_2_90b_vision_response,
                        llama_3_2_90b_vision_evaluation
                    ]
                )

            if "openai/gpt-4o" in llm_answer_checkbox_group:
                openai_gpt4o_response = openai_gpt4o_response
                if llm_evaluation_checkbox:
                    openai_gpt4o_evaluation = openai_gpt4o_evaluation
                else:
                    openai_gpt4o_evaluation = ""

                insert_sql = """
                             INSERT INTO RAG_QA_FEEDBACK (query_id,
                                                          llm_name,
                                                          llm_answer,
                                                          ragas_evaluation_result)
                             VALUES (:1,
                                     :2,
                                     :3,
                                     :4) \
                             """
                cursor.setinputsizes(None, None, oracledb.CLOB, oracledb.CLOB)
                cursor.execute(
                    insert_sql,
                    [
                        query_id,
                        "openai/gpt-4o",
                        openai_gpt4o_response,
                        openai_gpt4o_evaluation
                    ]
                )

            if "openai/gpt-4" in llm_answer_checkbox_group:
                openai_gpt4_response = openai_gpt4_response
                if llm_evaluation_checkbox:
                    openai_gpt4_evaluation = openai_gpt4_evaluation
                else:
                    openai_gpt4_evaluation = ""

                insert_sql = """
                             INSERT INTO RAG_QA_FEEDBACK (query_id,
                                                          llm_name,
                                                          llm_answer,
                                                          ragas_evaluation_result)
                             VALUES (:1,
                                     :2,
                                     :3,
                                     :4) \
                             """
                cursor.setinputsizes(None, None, oracledb.CLOB, oracledb.CLOB)
                cursor.execute(
                    insert_sql,
                    [
                        query_id,
                        "openai/gpt-4",
                        openai_gpt4_response,
                        openai_gpt4_evaluation
                    ]
                )

            if "azure_openai/gpt-4o" in llm_answer_checkbox_group:
                azure_openai_gpt4o_response = azure_openai_gpt4o_response
                if llm_evaluation_checkbox:
                    azure_openai_gpt4o_evaluation = azure_openai_gpt4o_evaluation
                else:
                    azure_openai_gpt4o_evaluation = ""

                insert_sql = """
                             INSERT INTO RAG_QA_FEEDBACK (query_id,
                                                          llm_name,
                                                          llm_answer,
                                                          ragas_evaluation_result)
                             VALUES (:1,
                                     :2,
                                     :3,
                                     :4) \
                             """
                cursor.setinputsizes(None, None, oracledb.CLOB, oracledb.CLOB)
                cursor.execute(
                    insert_sql,
                    [
                        query_id,
                        "azure_openai/gpt-4o",
                        azure_openai_gpt4o_response,
                        azure_openai_gpt4o_evaluation
                    ]
                )

            if "azure_openai/gpt-4" in llm_answer_checkbox_group:
                azure_openai_gpt4_response = azure_openai_gpt4_response
                if llm_evaluation_checkbox:
                    azure_openai_gpt4_evaluation = azure_openai_gpt4_evaluation
                else:
                    azure_openai_gpt4_evaluation = ""

                insert_sql = """
                             INSERT INTO RAG_QA_FEEDBACK (query_id,
                                                          llm_name,
                                                          llm_answer,
                                                          ragas_evaluation_result)
                             VALUES (:1,
                                     :2,
                                     :3,
                                     :4) \
                             """
                cursor.setinputsizes(None, None, oracledb.CLOB, oracledb.CLOB)
                cursor.execute(
                    insert_sql,
                    [
                        query_id,
                        "azure_openai/gpt-4",
                        azure_openai_gpt4_response,
                        azure_openai_gpt4_evaluation
                    ]
                )

        conn.commit()


def delete_document(server_directory, doc_ids):
    has_error = False
    if not server_directory:
        has_error = True
        gr.Warning("サーバー・ディレクトリを入力してください")
    print(f"{doc_ids=}")
    if not doc_ids or len(doc_ids) == 0 or (len(doc_ids) == 1 and doc_ids[0] == ''):
        has_error = True
        gr.Warning("ドキュメントを選択してください")
    if has_error:
        return (
            gr.Textbox(value=""),
            gr.Radio(),
            gr.CheckboxGroup()
        )

    output_sql = ""
    with pool.acquire() as conn, conn.cursor() as cursor:
        for doc_id in filter(bool, doc_ids):
            server_path = get_server_path(doc_id)
            if os.path.exists(server_path):
                os.remove(server_path)
                print(f"File {doc_id} deleted successfully")
            else:
                print(f"File {doc_id} not found")

            delete_embedding_sql = f"""
DELETE FROM {DEFAULT_COLLECTION_NAME}_embedding
WHERE doc_id = :doc_id """
            delete_collection_sql = f"""
DELETE FROM {DEFAULT_COLLECTION_NAME}_collection
WHERE id = :doc_id """
            delete_image_sql = f"""
DELETE FROM {DEFAULT_COLLECTION_NAME}_image
WHERE doc_id = :doc_id """
            delete_image_embedding_sql = f"""
DELETE FROM {DEFAULT_COLLECTION_NAME}_image_embedding
WHERE doc_id = :doc_id """

            output_sql += delete_embedding_sql.strip().replace(":doc_id", "'" + doc_id + "'") + ";\n"
            output_sql += delete_collection_sql.strip().replace(":doc_id", "'" + doc_id + "'") + ";\n"
            output_sql += delete_image_sql.strip().replace(":doc_id", "'" + doc_id + "'") + ";\n"
            output_sql += delete_image_embedding_sql.strip().replace(":doc_id", "'" + doc_id + "'") + ";\n"

            cursor.execute(delete_embedding_sql, doc_id=doc_id)
            cursor.execute(delete_collection_sql, doc_id=doc_id)
            cursor.execute(delete_image_sql, doc_id=doc_id)
            cursor.execute(delete_image_embedding_sql, doc_id=doc_id)

            conn.commit()

    doc_list = get_doc_list()
    return (
        gr.Textbox(value=output_sql),
        gr.Radio(doc_list),
        gr.CheckboxGroup(choices=doc_list, value=[]),
        gr.CheckboxGroup(choices=doc_list)
    )


theme = gr.themes.Default(
    spacing_size="sm",
    font=[GoogleFont(name="Noto Sans JP"), GoogleFont(name="Noto Sans SC"), GoogleFont(name="Roboto")]
).set(
)

with gr.Blocks(css=custom_css, theme=theme) as app:
    gr.Markdown(value="# RAG精度あげたろう", elem_classes="main_Header")
    gr.Markdown(value="### LLM＆RAG精度評価ツール",
                elem_classes="sub_Header")

    query_id_state = gr.State()

    with gr.Tabs() as tabs:
        with gr.TabItem(label="環境設定") as tab_setting:
            with gr.TabItem(label="OCI GenAIの設定*") as tab_create_oci_cred:
                with gr.Accordion(label="使用されたSQL", open=False) as tab_create_oci_cred_sql_accordion:
                    tab_create_oci_cred_sql_text = gr.Textbox(
                        label="SQL",
                        show_label=False,
                        lines=25,
                        max_lines=50,
                        autoscroll=False,
                        interactive=False,
                        show_copy_button=True
                    )
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_user_ocid_text = gr.Textbox(
                            label="User OCID*",
                            lines=1,
                            interactive=True
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_tenancy_ocid_text = gr.Textbox(
                            label="Tenancy OCID*",
                            lines=1,
                            interactive=True
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_fingerprint_text = gr.Textbox(
                            label="Fingerprint*",
                            lines=1,
                            interactive=True
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_private_key_file = gr.File(
                            label="Private Key*",
                            file_types=[".pem"],
                            type="filepath",
                            interactive=True
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_region_text = gr.Dropdown(
                            choices=["ap-osaka-1", "us-chicago-1"],
                            label="Region*",
                            interactive=True,
                            value="ap-osaka-1",
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_clear_button = gr.ClearButton(value="クリア")
                    with gr.Column():
                        tab_create_oci_cred_button = gr.Button(value="設定/再設定", variant="primary")
                with gr.Accordion(label="OCI GenAIのテスト", open=False) as tab_create_oci_cred_test_accordion:
                    with gr.Row():
                        with gr.Column():
                            tab_create_oci_cred_test_query_text = gr.Textbox(
                                label="テキスト",
                                lines=1,
                                max_lines=1,
                                value="こんにちわ"
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_create_oci_cred_test_vector_text = gr.Textbox(
                                label="ベクトル",
                                lines=10,
                                max_lines=10,
                                autoscroll=False
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_create_oci_cred_test_button = gr.Button(value="テスト", variant="primary")
            with gr.TabItem(label="テーブルの作成*") as tab_create_table:
                with gr.Accordion(label="使用されたSQL", open=False) as tab_create_table_sql_accordion:
                    tab_create_table_sql_text = gr.Textbox(
                        label="SQL",
                        show_label=False,
                        lines=25,
                        max_lines=50,
                        autoscroll=False,
                        interactive=False,
                        show_copy_button=True
                    )
                with gr.Row():
                    with gr.Column():
                        tab_create_table_button = gr.Button(value="作成/再作成", variant="primary")
            with gr.TabItem(label="Cohereの設定(オプション)") as tab_create_cohere_cred:
                with gr.Row():
                    with gr.Column():
                        tab_create_cohere_cred_api_key_text = gr.Textbox(
                            label="API Key*",
                            type="password",
                            lines=1,
                            interactive=True
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_cohere_cred_button = gr.Button(value="設定/再設定", variant="primary")
            with gr.TabItem(label="OpenAIの設定(オプション)") as tab_create_openai_cred:
                with gr.Row():
                    with gr.Column():
                        tab_create_openai_cred_base_url_text = gr.Textbox(
                            label="Base URL*", lines=1, interactive=True
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_openai_cred_api_key_text = gr.Textbox(
                            label="API Key*",
                            type="password",
                            lines=1,
                            interactive=True
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_openai_cred_button = gr.Button(value="設定/再設定", variant="primary")
            with gr.TabItem(label="Azure OpenAIの設定(オプション)") as tab_create_azure_openai_cred:
                with gr.Row():
                    with gr.Column():
                        tab_create_azure_openai_cred_api_key_text = gr.Textbox(
                            label="API Key*",
                            type="password",
                            lines=1,
                            interactive=True
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_azure_openai_cred_endpoint_gpt_4o_text = gr.Textbox(
                            label="GPT-4O Endpoint*",
                            lines=1,
                            interactive=True
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_azure_openai_cred_endpoint_gpt_4_text = gr.Textbox(
                            label="GPT-4 Endpoint(オプション)",
                            lines=1,
                            interactive=True
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_azure_openai_cred_button = gr.Button(value="設定/再設定", variant="primary")

            with gr.TabItem(label="Langfuseの設定(オプション)") as tab_create_langfuse_cred:
                with gr.Row():
                    with gr.Column():
                        tab_create_langfuse_cred_secret_key_text = gr.Textbox(
                            label="LANGFUSE_SECRET_KEY*",
                            lines=1,
                            interactive=True,
                            placeholder="sk-lf-..."
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_langfuse_cred_public_key_text = gr.Textbox(
                            label="LANGFUSE_PUBLIC_KEY*",
                            lines=1,
                            interactive=True,
                            placeholder="pk-lf-..."
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_langfuse_cred_host_text = gr.Textbox(
                            label="LANGFUSE_HOST*", lines=1,
                            interactive=True,
                            placeholder="http://xxx.xxx.xxx.xxx:3000"
                        )
                with gr.Row():
                    with gr.Column():
                        tab_create_langfuse_cred_button = gr.Button(value="設定/再設定", variant="primary")
        with gr.TabItem(label="LLM評価") as tab_llm_evaluation:
            with gr.TabItem(label="LLMとチャット") as tab_chat_with_llm:
                with gr.Row():
                    with gr.Column():
                        tab_chat_with_llm_answer_checkbox_group = gr.CheckboxGroup(
                            [
                                "xai/grok-3",
                                "cohere/command-a",
                                "meta/llama-4-maverick-17b-128e-instruct-fp8",
                                "meta/llama-4-scout-17b-16e-instruct",
                                "meta/llama-3-3-70b",
                                "meta/llama-3-2-90b-vision",
                                "openai/gpt-4o",
                                "openai/gpt-4",
                                "azure_openai/gpt-4o",
                                "azure_openai/gpt-4"],
                            label="LLM モデル*",
                            value=[]
                        )
                with gr.Accordion(
                        label="XAI Grok-3 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_xai_grok_3_accordion:
                    tab_chat_with_xai_grok_3_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="Command-A メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_command_a_accordion:
                    tab_chat_with_command_a_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )


                with gr.Accordion(
                        label="Llama 4 Maverick 17b メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_llama_4_maverick_accordion:
                    tab_chat_with_llama_4_maverick_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="Llama 4 Scout 17b メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_llama_4_scout_accordion:
                    tab_chat_with_llama_4_scout_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="Llama 3.3 70b メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_llama_3_3_70b_accordion:
                    tab_chat_with_llama_3_3_70b_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="Llama 3.2 90b Vision メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_llama_3_2_90b_vision_accordion:
                    tab_chat_with_llama_3_2_90b_vision_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="OpenAI gpt-4o メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_openai_gpt4o_accordion:
                    tab_chat_with_openai_gpt4o_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="OpenAI gpt-4 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_openai_gpt4_accordion:
                    tab_chat_with_openai_gpt4_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="Azure OpenAI gpt-4o メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_azure_openai_gpt4o_accordion:
                    tab_chat_with_azure_openai_gpt4o_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="Azure OpenAI gpt-4 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_azure_openai_gpt4_accordion:
                    tab_chat_with_azure_openai_gpt4_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )

                with gr.Accordion(open=False, label="システム・メッセージ"):
                    #                     tab_chat_with_llm_system_text = gr.Textbox(label="システム・メッセージ*", show_label=False, lines=5,
                    #                                                                max_lines=15,
                    #                                                                value="You are a helpful assistant. \n\
                    # Please respond to me in the same language I use for my messages. \n\
                    # If I switch languages, please switch your responses accordingly.")
                    tab_chat_with_llm_system_text = gr.Textbox(
                        label="システム・メッセージ*",
                        show_label=False,
                        lines=5,
                        max_lines=15,
                        value=get_chat_system_message()
                    )
                with gr.Accordion(open=False,
                                  label="画像ファイル(オプション) - Llama-4-Maverick、Llama-4-Scout、Llama-3.2-90B-Visionモデルを利用する場合に限り、この画像入力が適用されます。"):
                    tab_chat_with_llm_query_image = gr.Image(
                        label="",
                        interactive=True,
                        type="filepath",
                        height=300,
                        show_label=False,
                    )
                with gr.Row():
                    with gr.Column():
                        tab_chat_with_llm_query_text = gr.Textbox(
                            label="ユーザー・メッセージ*",
                            lines=2,
                            max_lines=5
                        )
                with gr.Row():
                    with gr.Column():
                        tab_chat_with_llm_clear_button = gr.ClearButton(value="クリア")
                    with gr.Column():
                        tab_chat_with_llm_chat_button = gr.Button(value="送信", variant="primary")
        with gr.TabItem(label="RAG評価", elem_classes="inner_tab") as tab_rag_evaluation:
            with gr.TabItem(label="Step-0.前処理") as tab_convert_document:
                with gr.TabItem(label="MarkItDown") as tab_convert_by_markitdown_document:
                    with gr.Row():
                        with gr.Column():
                            tab_convert_document_convert_by_markitdown_file_text = gr.File(
                                label="変換前のファイル*",
                                file_types=[
                                    ".csv", ".xls", ".xlsx", ".json", ".xml", ".ppt", ".pptx", ".doc", ".docx", ".pdf",
                                    ".txt", ".png", ".jpg", ".jpeg"
                                ],
                                type="filepath",
                                interactive=True,
                            )
                    with gr.Accordion("Advanced Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                tab_convert_document_convert_by_markitdown_use_llm_checkbox = gr.Checkbox(
                                    label="LLMによる処理を有効にする",
                                    value=True,
                                    info=(
                                        f"OCI Generative AI Visionモデルが利用されます。['.jpg','.jpeg','.png','.ppt','.pptx']に対応しています。"
                                    ),
                                    interactive=True,
                                )
                        with gr.Row():
                            with gr.Column():
                                tab_convert_document_convert_by_markitdown_llm_prompt_text = gr.Textbox(
                                    label="LLM ユーザー・メッセージ",
                                    value=get_markitdown_llm_prompt(),
                                    interactive=True,
                                    lines=2,
                                    max_lines=5,
                                )
                    with gr.Row():
                        with gr.Column():
                            tab_convert_document_convert_by_markitdown_button = gr.Button(
                                value="Markdownへ変換",
                                variant="primary")
                with gr.TabItem(label="Excel2Text") as tab_convert_excel_to_text_document:
                    with gr.Row():
                        with gr.Column():
                            tab_convert_document_convert_excel_to_text_file_text = gr.File(
                                label="変換前のファイル*",
                                file_types=[
                                    ".csv", ".xls", ".xlsx"
                                ],
                                type="filepath",
                                interactive=True,
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_convert_document_convert_button = gr.Button(
                                value="ExcelをTextへ変換",
                                variant="primary")
            with gr.TabItem(label="Step-1.読込み") as tab_load_document:
                with gr.Accordion(label="使用されたSQL", open=False) as tab_load_document_sql_accordion:
                    tab_load_document_output_sql_text = gr.Textbox(
                        label="使用されたSQL",
                        show_label=False,
                        lines=10,
                        autoscroll=False,
                        show_copy_button=True
                    )
                with gr.Row():
                    with gr.Column():
                        tab_load_document_doc_id_text = gr.Textbox(
                            label="Doc ID",
                            lines=1,
                            interactive=False
                        )
                with gr.Row(visible=False):
                    with gr.Column():
                        tab_load_document_page_count_text = gr.Textbox(
                            label="ページ数",
                            lines=1,
                            interactive=False
                        )
                with gr.Row():
                    with gr.Column():
                        tab_load_document_page_content_text = gr.Textbox(
                            label="コンテンツ",
                            lines=15,
                            max_lines=15,
                            autoscroll=False,
                            show_copy_button=True,
                            interactive=False
                        )
                with gr.Row():
                    with gr.Column():
                        tab_load_document_file_text = gr.File(
                            label="ファイル*",
                            file_types=[
                                ".txt", ".csv", ".doc", ".docx", ".epub", ".image",
                                ".md", ".msg", ".odt", ".org", ".pdf", ".ppt",
                                ".pptx",
                                ".rtf", ".rst", ".tsv", ".xls", ".xlsx"
                            ],
                            type="filepath")
                    with gr.Column():
                        tab_load_document_server_directory_text = gr.Text(
                            label="サーバー・ディレクトリ*",
                            value="/u01/data/no1rag/"
                        )
                with gr.Row():
                    with gr.Column():
                        tab_load_document_metadata_text = gr.Textbox(
                            label="メタデータ(オプション)",
                            lines=1,
                            max_lines=1,
                            autoscroll=True,
                            show_copy_button=False,
                            interactive=True,
                            info="key1=value1,key2=value2,... の形式で入力してください。\"'などの記号はサポートしていません。",
                            placeholder="key1=value1,key2=value2,..."
                        )
                with gr.Row():
                    with gr.Column():
                        tab_load_document_load_button = gr.Button(value="読込み", variant="primary")
            with gr.TabItem(label="Step-2.分割・ベクトル化・保存") as tab_split_document:
                # with gr.Accordion("UTL_TO_CHUNKS 設定*"):
                with gr.Accordion(
                        "CHUNKS 設定*(<FIXED_DELIMITER>という分割符がドキュメントに含まれている場合、チャンクは<FIXED_DELIMITER>分割され、MaxおよびOverlapの設定は無視されます。)"):
                    with gr.Row():
                        with gr.Column():
                            tab_split_document_chunks_language_radio = gr.Radio(
                                label="LANGUAGE",
                                choices=[
                                    ("JAPANESE", "JAPANESE"),
                                    ("AMERICAN", "AMERICAN")
                                ],
                                value="JAPANESE",
                                visible=False,
                                info="Default value: JAPANESE。入力テキストの言語を指定。テキストに言語によって解釈が異なる可能性のある特定の文字(例えば、句読点や略語)が含まれる場合、この言語の指定は特に重要です。Oracle Database Globalization Support Guideに記載されているNLSでサポートされている言語名または略称を指定できる。"
                            )
                        with gr.Column():
                            tab_split_document_chunks_by_radio = gr.Radio(
                                label="BY",
                                choices=[
                                    ("BY CHARACTERS (or BY CHARS )",
                                     "CHARACTERS"),
                                    ("BY WORDS", "WORDS"),
                                    ("BY VOCABULARY", "VOCABULARY")
                                ],
                                value="WORDS",
                                visible=False,
                                info="Default value: BY WORDS。データ分割の方法を文字、単語、語彙トークンで指定。BY CHARACTERS: 文字数で計算して分割。BY WORDS: 単語数を計算して分割、単語ごとに空白文字が入る言語が対象、日本語、中国語、タイ語などの場合、各ネイティブ文字は単語（ユニグラム）としてみなされる。BY VOCABULARY: 語彙のトークン数を計算して分割、CREATE_VOCABULARYパッケージを使って、語彙登録が可能。",
                            )
                    with gr.Row():
                        with gr.Column(scale=1):
                            # tab_split_document_chunks_max_text = gr.Text(label="Max",
                            #                                              value="256",
                            #                                              lines=1,
                            #                                              info="",
                            #                                              )
                            tab_split_document_chunks_max_slider = gr.Slider(
                                label="Max",
                                value=320,
                                minimum=64,
                                maximum=512,
                                step=1,
                            )
                        with gr.Column(scale=1):
                            tab_split_document_chunks_overlap_slider = gr.Slider(
                                label="Overlap(Percentage of Max)",
                                minimum=0,
                                maximum=20,
                                step=5,
                                value=0,
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_split_document_chunks_split_by_radio = gr.Radio(
                                label="SPLIT [BY]",
                                choices=[
                                    ("NONE", "NONE"),
                                    ("NEWLINE", "NEWLINE"),
                                    ("BLANKLINE", "BLANKLINE"),
                                    ("SPACE", "SPACE"),
                                    ("RECURSIVELY", "RECURSIVELY"),
                                    ("SENTENCE", "SENTENCE"),
                                    ("CUSTOM", "CUSTOM")
                                ],
                                value="RECURSIVELY",
                                visible=False,
                                info="Default value: RECURSIVELY。テキストが最大サイズに達したときに、どうやって分割するかを指定。チャンクの適切な境界を定義するために使用する。NONE: MAX指定されている文字数、単語数、語彙トークン数に達したら分割。NEWLINE: MAX指定サイズを超えてテキストの行末で分割。BLANKLINE: 指定サイズを超えてBLANKLINE（2回の改行）の末尾で分割。SPACE: MAX指定サイズを超えて空白の行末で分割。RECURSIVELY: BLANKLINE, NEWLINE, SPACE, NONEの順に条件に応じて分割する。1.入力テキストがmax値以上の場合、最初の分割文字で分割、2.1.が失敗した場合に、2番目の分割文字で分割、3.分割文字が存在しない場合、テキスト中のどの位置においてもMAXで分割。SENTENCE: 文末の句読点で分割BY WORDSとBY VOCABULARYでのみ指定可能。MAX設定の仕方によっては必ず句読点で区切られるわけではないので注意。例えば、文がMAX値よりも大きい場合、MAX値で区切られる。MAX値よりも小さな文の場合で、2文以上がMAXの制限内に収まるのであれば1つに収める。CUSTOM: カスタム分割文字リストに基づいて分割、分割文字は最大16個まで、長さはそれぞれ10文字までで指定可能。"
                            )
                        with gr.Column():
                            tab_split_document_chunks_split_by_custom_text = gr.Text(
                                label="CUSTOM SPLIT CHARACTERS(SPLIT [BY] = CUSTOMの場合のみ有効)",
                                # value="'\u3002', '.'",
                                visible=False,
                                info="カンマ区切りのカスタム分割文字リストに基づいて分割、分割文字は最大16個まで、長さはそれぞれ10文字までで指定可能。タブ (\t)、改行 (\n)、およびラインフィード (\r) のシーケンスのみを省略できる。たとえば、'<html>','</html>'"
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_split_document_chunks_normalize_radio = gr.Radio(
                                label="NORMALIZE",
                                choices=[("NONE", "NONE"),
                                         ("ALL", "ALL"),
                                         ("OPTIONS", "OPTIONS")],
                                value="ALL",
                                visible=False,
                                info="Default value: ALL。ドキュメントをテキストに変換する際にありがちな問題について、自動的に前処理、後処理を実行し高品質なチャンクとして格納するために使用。NONE: 処理を行わない。ALL: マルチバイトの句読点をシングルバイトに正規化。OPTIONS: PUNCTUATION: スマート引用符、スマートハイフン、マルチバイト句読点をテキストに含める。WHITESPACE: 不要な文字を削除して空白を最小限に抑える例えば空白行はそのままに、余分な改行やスペース、タブを削除する。WIDECHAR: マルチバイト数字とローマ字をシングルバイトに正規化する"
                            )
                        with gr.Column():
                            tab_split_document_chunks_normalize_options_checkbox_group = gr.CheckboxGroup(
                                label="NORMALIZE OPTIONS(NORMALIZE = OPTIONSの場合のみ有効かつ必須)",
                                choices=[
                                    ("PUNCTUATION", "PUNCTUATION"),
                                    ("WHITESPACE", "WHITESPACE"),
                                    ("WIDECHAR", "WIDECHAR")],
                                visible=False,
                                info="ドキュメントをテキストに変換する際にありがちな問題について、自動的に前処理、後処理を実行し高品質なチャンクとして格納するために使用。PUNCTUATION: スマート引用符、スマートハイフン、マルチバイト句読点をテキストに含める。WHITESPACE: 不要な文字を削除して空白を最小限に抑える例えば空白行はそのままに、余分な改行やスペース、タブを削除する。WIDECHAR: マルチバイト数字とローマ字をシングルバイトに正規化する"
                            )

                with gr.Row():
                    with gr.Column():
                        # doc_id_text = gr.Textbox(label="Doc ID*", lines=1)
                        tab_split_document_doc_id_radio = gr.Radio(
                            choices=get_doc_list(),
                            label="ドキュメント*",
                        )

                with gr.Row():
                    with gr.Column():
                        tab_split_document_split_button = gr.Button(value="分割", variant="primary")
                    with gr.Column():
                        tab_split_document_embed_save_button = gr.Button(
                            value="ベクトル化して保存（データ量が多いと時間がかかる）",
                            variant="primary"
                        )

                with gr.Accordion(label="使用されたSQL", open=False) as tab_split_document_sql_accordion:
                    tab_split_document_output_sql_text = gr.Textbox(
                        label="使用されたSQL",
                        show_label=False,
                        lines=10,
                        autoscroll=False,
                        show_copy_button=True
                    )
                with gr.Row():
                    tab_split_document_chunks_count = gr.Textbox(label="チャンク数", lines=1)
                with gr.Row():
                    tab_split_document_chunks_result_dataframe = gr.Dataframe(
                        label="チャンク結果",
                        headers=["CHUNK_ID", "CHUNK_OFFSET", "CHUNK_LENGTH", "CHUNK_DATA"],
                        datatype=["str", "str", "str", "str"],
                        row_count=(1, "fixed"),
                        col_count=(4, "fixed"),
                        wrap=True,
                        column_widths=["10%", "10%", "10%", "70%"]
                    )
                with gr.Accordion(label="チャンク詳細",
                                  open=True) as tab_split_document_chunks_result_detail:
                    with gr.Row():
                        with gr.Column():
                            tab_split_document_chunks_result_detail_chunk_id = gr.Textbox(
                                label="CHUNK_ID",
                                interactive=False,
                                lines=1
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_split_document_chunks_result_detail_chunk_data = gr.Textbox(
                                label="CHUNK_DATA",
                                interactive=True,
                                lines=5,
                                max_lines=5,
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_split_document_chunks_result_detail_update_button = gr.Button(
                                value="更新",
                                variant="primary"
                            )

            with gr.TabItem(label="Step-3.削除(オプション)") as tab_delete_document:
                with gr.Accordion(
                        label="使用されたSQL",
                        open=False
                ) as tab_delete_document_sql_accordion:
                    tab_delete_document_delete_sql = gr.Textbox(
                        label="生成されたSQL",
                        show_label=False,
                        lines=10,
                        autoscroll=False,
                        show_copy_button=True
                    )
                with gr.Row():
                    with gr.Column():
                        tab_delete_document_server_directory_text = gr.Text(
                            label="サーバー・ディレクトリ*",
                            value="/u01/data/no1rag/"
                        )
                with gr.Row():
                    with gr.Column():
                        # doc_id_text = gr.Textbox(label="Doc ID*", lines=1)
                        tab_delete_document_doc_ids_checkbox_group = gr.CheckboxGroup(
                            choices=get_doc_list(),
                            label="ドキュメント*",
                            type="value",
                            value=[],
                        )
                with gr.Row():
                    with gr.Column():
                        tab_delete_document_delete_button = gr.Button(value="削除", variant="primary")
            with gr.TabItem(label="Step-4.ドキュメントとチャット") as tab_chat_document:
                with gr.Row():
                    with gr.Column():
                        tab_chat_document_llm_answer_checkbox_group = gr.CheckboxGroup(
                            [
                                "xai/grok-3",
                                "cohere/command-a",
                                "meta/llama-4-maverick-17b-128e-instruct-fp8",
                                "meta/llama-4-scout-17b-16e-instruct",
                                "meta/llama-3-3-70b",
                                "meta/llama-3-2-90b-vision",
                                "openai/gpt-4o",
                                "openai/gpt-4",
                                "azure_openai/gpt-4o",
                                "azure_openai/gpt-4"
                            ],
                            label="LLM モデル",
                            value=[]
                        )
                with gr.Row():
                    with gr.Column():
                        tab_chat_document_question_embedding_model_checkbox_group = gr.CheckboxGroup(
                            ["cohere/embed-multilingual-v3.0"],
                            label="Embedding モデル*",
                            value="cohere/embed-multilingual-v3.0",
                            interactive=False
                        )
                    with gr.Column():
                        tab_chat_document_reranker_model_radio = gr.Radio(
                            [
                                "None",
                                # "cohere/rerank-multilingual-v3.1",
                                # "cohere/rerank-english-v3.1",
                                "cohere/rerank-multilingual-v3.0",
                                "cohere/rerank-english-v3.0",
                            ],
                            label="Rerank モデル*", value="None")
                with gr.Row():
                    with gr.Column():
                        tab_chat_document_top_k_slider = gr.Slider(
                            label="類似検索 Top-K*",
                            minimum=1,
                            maximum=100,
                            step=1,
                            info="Default value: 20。類似度距離の低い順（=類似度の高い順）で上位K件のみを抽出する。",
                            interactive=True,
                            value=20
                        )
                    with gr.Column():
                        tab_chat_document_threshold_value_slider = gr.Slider(
                            label="類似検索閾値*",
                            minimum=0.10,
                            info="Default value: 0.55。類似度距離が閾値以下のデータのみを抽出する。",
                            maximum=0.95,
                            step=0.05,
                            value=0.55
                        )
                with gr.Row():
                    with gr.Column():
                        tab_chat_document_reranker_top_k_slider = gr.Slider(
                            label="Rerank Top-K*",
                            minimum=1,
                            maximum=50,
                            step=1,
                            info="Default value: 5。Rerank Scoreの高い順で上位K件のみを抽出する。",
                            interactive=True,
                            value=5
                        )
                    with gr.Column():
                        tab_chat_document_reranker_threshold_slider = gr.Slider(
                            label="Rerank Score 閾値*",
                            minimum=0.0,
                            info="Default value: 0.0045。Rerank Scoreが閾値以上のデータのみを抽出する。",
                            maximum=0.99,
                            step=0.0005,
                            value=0.0045,
                            interactive=True
                        )
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_answer_by_one_checkbox = gr.Checkbox(
                                label="Highest-Ranked-One 文書による回答",
                                value=False,
                                info="他のすべての文書を無視し、最上位にランクされた1つの文書のみによって回答する。"
                            )
                        with gr.Column():
                            pass
                        with gr.Column(visible=False):
                            tab_chat_document_partition_by_k_slider = gr.Slider(
                                label="Partition-By-K",
                                minimum=0,
                                maximum=20,
                                step=1,
                                interactive=True,
                                info="Default value: 0。類似検索の対象ドキュメント数を指定。0: 全部。1: 1個。2：2個。... n: n個。"
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_extend_first_chunk_size = gr.Slider(
                                label="Extend-First-K", minimum=0,
                                maximum=50,
                                step=1,
                                interactive=True,
                                value=0,
                                info="Default value: 0。DISTANCE計算対象外。検索されたチャンクを拡張する数を指定。0: 拡張しない。 1: 最初の1個を拡張。2: 最初の2個を拡張。 ... n: 最初のn個を拡張。"
                            )
                        with gr.Column():
                            tab_chat_document_extend_around_chunk_size = gr.Slider(
                                label="Extend-Around-K",
                                minimum=0,
                                maximum=50, step=2,
                                interactive=True,
                                value=2,
                                info="Default value: 2。DISTANCE計算対象外。検索されたチャンクを拡張する数を指定。0: 拡張しない。 2: 2個で前後それぞれ1個を拡張。4: 4個で前後それぞれ2個を拡張。... n: n個で前後それぞれn/2個を拡張。"
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_text_search_checkbox = gr.Checkbox(
                                label="テキスト検索",
                                value=False,
                                info="テキスト検索は元のクエリに限定され、クエリの拡張で生成されたクエリは無視される。"
                            )
                        with gr.Column():
                            tab_chat_document_text_search_k_slider = gr.Slider(
                                label="テキスト検索Limit-K",
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=6,
                                info="Default value: 6。テキスト検索に使用できる単語数の制限。"
                            )
                    with gr.Accordion(label="RAG Prompt 設定", open=False) as tab_chat_document_rag_prompt_accordion:
                        with gr.Row():
                            with gr.Column():
                                tab_chat_document_rag_prompt_text = gr.Textbox(
                                    label="RAG Prompt テンプレート",
                                    lines=15,
                                    max_lines=25,
                                    interactive=True,
                                    show_copy_button=True,
                                    value=get_langgpt_rag_prompt("{{context}}", "{{query_text}}", False, False),
                                    info="RAGで使用されるpromptテンプレートです。{{context}}と{{query_text}}は実行時に置換されます。"
                                )
                        with gr.Row():
                            with gr.Column(scale=1):
                                tab_chat_document_rag_prompt_reset_button = gr.Button(
                                    value="デフォルトに戻す",
                                    variant="secondary"
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_rag_prompt_save_button = gr.Button(
                                    value="保存",
                                    variant="primary",
                                    visible=False,
                                )
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_include_citation_checkbox = gr.Checkbox(
                                label="回答に引用を含める",
                                value=False,
                                info="回答には引用を含め、使用したコンテキストのみを引用として出力する。"
                            )
                        with gr.Column():
                            tab_chat_document_include_current_time_checkbox = gr.Checkbox(
                                label="Promptに現在の時間を含める",
                                value=False,
                                info="Promptに回答時の現在の時間を含めます。"
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_use_image_checkbox = gr.Checkbox(
                                label="画像を使って回答",
                                value=False,
                                info="RAGの回答時に画像データを使用。ただし、処理する画像数を10個に制限。"
                            )
                        with gr.Column():
                            tab_chat_document_single_image_processing_checkbox = gr.Checkbox(
                                label="1枚ずつ処理",
                                value=True,
                                info="チェックすると画像を1枚ずつ処理、チェックしないと全ての画像を一括処理。"
                            )
                    with gr.Accordion(label="画像 Prompt 設定", open=False,
                                      visible=False) as tab_chat_document_image_prompt_accordion:
                        with gr.Row():
                            with gr.Column():
                                tab_chat_document_image_prompt_text = gr.Textbox(
                                    label="画像 Prompt テンプレート",
                                    lines=15,
                                    max_lines=25,
                                    interactive=True,
                                    show_copy_button=True,
                                    value=get_image_qa_prompt("{{query_text}}"),
                                    info="画像処理で使用されるpromptテンプレートです。{{query_text}}は実行時に置換されます。"
                                )
                        with gr.Row():
                            with gr.Column(scale=1):
                                tab_chat_document_image_prompt_reset_button = gr.Button(
                                    value="デフォルトに戻す",
                                    variant="secondary"
                                )
                            with gr.Column(scale=1):
                                gr.Markdown("&nbsp;")
                                # tab_chat_document_image_prompt_save_button = gr.Button(
                                #     value="保存",
                                #     variant="primary",
                                #     visible=False,
                                # )
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_document_metadata_text = gr.Textbox(
                                label="メタデータ",
                                lines=1,
                                max_lines=1,
                                autoscroll=True,
                                show_copy_button=False,
                                interactive=True,
                                info="key1=value1,key2=value2,... の形式で入力する。",
                                placeholder="key1=value1,key2=value2,..."
                            )
                with gr.Row(visible=False):
                    tab_chat_document_accuracy_plan_radio = gr.Radio(
                        [
                            "Somewhat Inaccurate",
                            "Decent Accuracy",
                            "Extremely Precise"
                        ],
                        label="Accuracy Plan*",
                        value="Somewhat Inaccurate",
                        interactive=True
                    )
                with gr.Row():
                    tab_chat_document_llm_evaluation_checkbox = gr.Checkbox(
                        label="LLM 評価",
                        show_label=True,
                        interactive=True,
                        value=False,
                    )
                with gr.Row():
                    #                     tab_chat_document_system_message_text = gr.Textbox(label="システム・メッセージ*", lines=15,
                    #                                                                        max_lines=20,
                    #                                                                        interactive=True,
                    #                                                                        visible=False,
                    #                                                                        value=f"""You are an ANSWER EVALUATOR.
                    # Your task is to compare a given answer to a standard answer and evaluate its quality.
                    # Respond with a score from 0 to 10 for each of the following criteria:
                    # 1. Correctness (0 being completely incorrect, 10 being perfectly correct)
                    # 2. Completeness (0 being entirely incomplete, 10 being fully complete)
                    # 3. Clarity (0 being very unclear, 10 being extremely clear)
                    # 4. Conciseness (0 being extremely verbose, 10 being optimally concise)
                    #
                    # After providing scores, give a brief explanation for each score.
                    # Finally, provide an overall score from 0 to 10 and a summary of the evaluation.
                    # Please respond to me in the same language I use for my messages.
                    # If I switch languages, please switch your responses accordingly.
                    #                 """)
                    tab_chat_document_system_message_text = gr.Textbox(
                        label="システム・メッセージ*",
                        lines=15,
                        max_lines=20,
                        interactive=True,
                        visible=False,
                        value=get_llm_evaluation_system_message())
                with gr.Row():
                    tab_chat_document_standard_answer_text = gr.Textbox(
                        label="標準回答*",
                        lines=2,
                        interactive=True,
                        visible=False
                    )
                with gr.Accordion("ドキュメント*", open=True):
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_doc_id_all_checkbox = gr.Checkbox(label="全部", value=True)
                    with gr.Row():
                        with gr.Column():
                            # doc_id_text = gr.Textbox(label="Doc ID*", lines=1)
                            tab_chat_document_doc_id_checkbox_group = gr.CheckboxGroup(
                                choices=get_doc_list(),
                                label="ドキュメント*",
                                show_label=False,
                                interactive=False,
                            )
                with gr.Row() as tab_chat_document_searched_query_row:
                    with gr.Column():
                        tab_chat_document_query_text = gr.Textbox(label="クエリ*", lines=2)
                # with gr.Accordion("Sub-Query/RAG-Fusion/HyDE/Step-Back-Prompting/Customized-Multi-Step-Query", open=True):
                with gr.Accordion("クエリの拡張", open=False):
                    with gr.Row():
                        # generate_query_radio = gr.Radio(
                        #     ["None", "Sub-Query", "RAG-Fusion", "HyDE", "Step-Back-Prompting",
                        #      "Customized-Multi-Step-Query"],
                        tab_chat_document_generate_query_radio = gr.Radio(
                            [
                                ("None", "None"),
                                ("サブクエリ", "Sub-Query"),
                                ("類似クエリ", "RAG-Fusion"),
                                ("仮回答", "HyDE"),
                                ("抽象化クエリ", "Step-Back-Prompting")
                            ],
                            label="LLMによって生成？",
                            value="None",
                            interactive=True
                        )
                    with gr.Row():
                        tab_chat_document_sub_query1_text = gr.Textbox(
                            # label="(Sub-Query)サブクエリ1/(RAG-Fusion)類似クエリ1/(HyDE)仮回答1/(Step-Back-Prompting)抽象化クエリ1/(Customized-Multi-Step-Query)マルチステップクエリ1",
                            label="生成されたクエリ1",
                            lines=1,
                            interactive=True,
                            info=""
                        )
                    with gr.Row():
                        tab_chat_document_sub_query2_text = gr.Textbox(
                            # label="(Sub-Query)サブクエリ2/(RAG-Fusion)類似クエリ2/(HyDE)仮回答2/(Step-Back-Prompting)抽象化クエリ2/(Customized-Multi-Step-Query)マルチステップクエリ2",
                            label="生成されたクエリ2",
                            lines=1,
                            interactive=True
                        )
                    with gr.Row():
                        tab_chat_document_sub_query3_text = gr.Textbox(
                            # label="(Sub-Query)サブクエリ3/(RAG-Fusion)類似クエリ3/(HyDE)仮回答3/(Step-Back-Prompting)抽象化クエリ3/(Customized-Multi-Step-Query)マルチステップクエリ3",
                            label="生成されたクエリ3",
                            lines=1,
                            interactive=True
                        )
                with gr.Row() as tab_chat_document_chat_document_row:
                    with gr.Column():
                        tab_chat_document_chat_document_button = gr.Button(value="送信", variant="primary")
                with gr.Accordion(label="使用されたSQL", open=False) as tab_chat_document_sql_accordion:
                    tab_chat_document_output_sql_text = gr.Textbox(
                        label="使用されたSQL",
                        show_label=False,
                        lines=25,
                        max_lines=25,
                        autoscroll=False,
                        show_copy_button=True
                    )
                with gr.Row() as tab_chat_document_searched_data_summary_row:
                    with gr.Column(scale=10):
                        tab_chat_document_searched_data_summary_text = gr.Markdown(
                            value="",
                            visible=False
                        )
                    with gr.Column(scale=2):
                        tab_chat_document_download_output_button = gr.DownloadButton(
                            label="ダウンロード",
                            visible=False,
                            variant="secondary",
                            size="sm"
                        )
                with gr.Row() as searched_result_row:
                    with gr.Column():
                        tab_chat_document_searched_result_dataframe = gr.Dataframe(
                            headers=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"],
                            datatype=["str", "str", "str", "str", "str", "str", "str"],
                            row_count=(5, "fixed"),
                            max_height=400,
                            col_count=(7, "fixed"),
                            wrap=True,
                            column_widths=["4%", "50%", "6%", "8%", "6%", "6%", "8%"],
                            interactive=False,
                        )
                with gr.Accordion(
                        label="XAI Grok-3 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_xai_grok_3_accordion:
                    tab_chat_document_xai_grok_3_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_xai_grok_3_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_xai_grok_3_answer_human_eval_feedback_radio = gr.Radio(
                                show_label=False,
                                choices=[
                                    ("Good response", "good"),
                                    ("Neutral response", "neutral"),
                                    ("Bad response", "bad"),
                                ],
                                value="good",
                                container=False,
                                interactive=True,
                            )
                        with gr.Row():
                            with gr.Column(scale=11):
                                tab_chat_document_xai_grok_3_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_xai_grok_3_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_xai_grok_3_evaluation_accordion:
                        tab_chat_document_xai_grok_3_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="Command-A メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_command_a_accordion:
                    tab_chat_document_command_a_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_command_a_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_command_a_answer_human_eval_feedback_radio = gr.Radio(
                                show_label=False,
                                choices=[
                                    ("Good response", "good"),
                                    ("Neutral response", "neutral"),
                                    ("Bad response", "bad"),
                                ],
                                value="good",
                                container=False,
                                interactive=True,
                            )
                        with gr.Row():
                            with gr.Column(scale=11):
                                tab_chat_document_command_a_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_command_a_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_command_a_evaluation_accordion:
                        tab_chat_document_command_a_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )


                with gr.Accordion(
                        label="Llama 4 Maverick 17b メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_llama_4_maverick_accordion:
                    tab_chat_document_llama_4_maverick_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="画像回答",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_llama_4_maverick_image_accordion:
                        tab_chat_document_llama_4_maverick_image_answer_text = gr.Markdown(
                            show_copy_button=True,
                            height=600,
                            min_height=600,
                            max_height=600
                        )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_llama_4_maverick_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_llama_4_maverick_answer_human_eval_feedback_radio = gr.Radio(
                                show_label=False,
                                choices=[
                                    ("Good response", "good"),
                                    ("Neutral response", "neutral"),
                                    ("Bad response", "bad"),
                                ],
                                value="good",
                                container=False,
                                interactive=True,
                            )
                        with gr.Row():
                            with gr.Column(scale=11):
                                tab_chat_document_llama_4_maverick_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_llama_4_maverick_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_llama_4_maverick_evaluation_accordion:
                        tab_chat_document_llama_4_maverick_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="Llama 4 Scout 17b メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_llama_4_scout_accordion:
                    tab_chat_document_llama_4_scout_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="画像回答",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_llama_4_scout_image_accordion:
                        tab_chat_document_llama_4_scout_image_answer_text = gr.Markdown(
                            show_copy_button=True,
                            height=600,
                            min_height=600,
                            max_height=600
                        )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_llama_4_scout_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_llama_4_scout_answer_human_eval_feedback_radio = gr.Radio(
                                show_label=False,
                                choices=[
                                    ("Good response", "good"),
                                    ("Neutral response", "neutral"),
                                    ("Bad response", "bad"),
                                ],
                                value="good",
                                container=False,
                                interactive=True,
                            )
                        with gr.Row():
                            with gr.Column(scale=11):
                                tab_chat_document_llama_4_scout_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_llama_4_scout_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_llama_4_scout_evaluation_accordion:
                        tab_chat_document_llama_4_scout_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="Llama 3.3 70b メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_llama_3_3_70b_accordion:
                    tab_chat_document_llama_3_3_70b_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_llama_3_3_70b_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_llama_3_3_70b_answer_human_eval_feedback_radio = gr.Radio(
                                show_label=False,
                                choices=[
                                    ("Good response", "good"),
                                    ("Neutral response", "neutral"),
                                    ("Bad response", "bad"),
                                ],
                                value="good",
                                container=False,
                                interactive=True,
                            )
                        with gr.Row():
                            with gr.Column(scale=11):
                                tab_chat_document_llama_3_3_70b_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_llama_3_3_70b_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_llama_3_3_70b_evaluation_accordion:
                        tab_chat_document_llama_3_3_70b_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="Llama 3.2 90b Vision メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_llama_3_2_90b_vision_accordion:
                    tab_chat_document_llama_3_2_90b_vision_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="画像回答",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_llama_3_2_90b_vision_image_accordion:
                        tab_chat_document_llama_3_2_90b_vision_image_answer_text = gr.Markdown(
                            show_copy_button=True,
                            height=600,
                            min_height=600,
                            max_height=600
                        )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_llama_3_2_90b_vision_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_llama_3_2_90b_vision_answer_human_eval_feedback_radio = gr.Radio(
                                show_label=False,
                                choices=[
                                    ("Good response", "good"),
                                    ("Neutral response", "neutral"),
                                    ("Bad response", "bad"),
                                ],
                                value="good",
                                container=False,
                                interactive=True,
                            )
                        with gr.Row():
                            with gr.Column(scale=11):
                                tab_chat_document_llama_3_2_90b_vision_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_llama_3_2_90b_vision_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_llama_3_2_90b_vision_evaluation_accordion:
                        tab_chat_document_llama_3_2_90b_vision_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(label="OpenAI gpt-4o メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_document_llm_openai_gpt4o_accordion:
                    tab_chat_document_openai_gpt4o_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="画像回答",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_openai_gpt4o_image_accordion:
                        tab_chat_document_openai_gpt4o_image_answer_text = gr.Markdown(
                            show_copy_button=True,
                            height=600,
                            min_height=600,
                            max_height=600
                        )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_openai_gpt4o_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_openai_gpt4o_answer_human_eval_feedback_radio = gr.Radio(
                                show_label=False,
                                choices=[
                                    ("Good response", "good"),
                                    ("Neutral response", "neutral"),
                                    ("Bad response", "bad"),
                                ],
                                value="good",
                                container=False,
                                interactive=True,
                            )
                        with gr.Row():
                            with gr.Column(scale=11):
                                tab_chat_document_openai_gpt4o_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_openai_gpt4o_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_openai_gpt4o_evaluation_accordion:
                        tab_chat_document_openai_gpt4o_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="OpenAI gpt-4 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_openai_gpt4_accordion:
                    tab_chat_document_openai_gpt4_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_openai_gpt4_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_openai_gpt4_answer_human_eval_feedback_radio = gr.Radio(
                                show_label=False,
                                choices=[
                                    ("Good response", "good"),
                                    ("Neutral response", "neutral"),
                                    ("Bad response", "bad"),
                                ],
                                value="good",
                                container=False,
                                interactive=True,
                            )
                        with gr.Row():
                            with gr.Column(scale=11):
                                tab_chat_document_openai_gpt4_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_openai_gpt4_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_openai_gpt4_evaluation_accordion:
                        tab_chat_document_openai_gpt4_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="Azure OpenAI gpt-4o メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_azure_openai_gpt4o_accordion:
                    tab_chat_document_azure_openai_gpt4o_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="画像回答",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_azure_openai_gpt4o_image_accordion:
                        tab_chat_document_azure_openai_gpt4o_image_answer_text = gr.Markdown(
                            show_copy_button=True,
                            height=600,
                            min_height=600,
                            max_height=600
                        )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_azure_openai_gpt4o_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_azure_openai_gpt4o_answer_human_eval_feedback_radio = gr.Radio(
                                show_label=False,
                                choices=[
                                    ("Good response", "good"),
                                    ("Neutral response", "neutral"),
                                    ("Bad response", "bad"),
                                ],
                                value="good",
                                container=False,
                                interactive=True,
                            )
                        with gr.Row():
                            with gr.Column(scale=11):
                                tab_chat_document_azure_openai_gpt4o_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_azure_openai_gpt4o_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_azure_openai_gpt4o_evaluation_accordion:
                        tab_chat_document_azure_openai_gpt4o_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="Azure OpenAI gpt-4 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_azure_openai_gpt4_accordion:
                    tab_chat_document_azure_openai_gpt4_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_azure_openai_gpt4_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_azure_openai_gpt4_answer_human_eval_feedback_radio = gr.Radio(
                                show_label=False,
                                choices=[
                                    ("Good response", "good"),
                                    ("Neutral response", "neutral"),
                                    ("Bad response", "bad"),
                                ],
                                value="good",
                                container=False,
                                interactive=True,
                            )
                        with gr.Row():
                            with gr.Column(scale=11):
                                tab_chat_document_azure_openai_gpt4_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_azure_openai_gpt4_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_azure_openai_gpt4_evaluation_accordion:
                        tab_chat_document_azure_openai_gpt4_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )

            with gr.TabItem(label="Step-5.評価レポートの取得") as tab_download_eval_result:
                with gr.Row():
                    tab_download_eval_result_generate_button = gr.Button(
                        value="評価レポートの生成",
                        variant="primary",
                    )

                    tab_download_eval_result_download_button = gr.DownloadButton(
                        label="評価レポートのダウンロード",
                        variant="primary",
                    )

    gr.Markdown(
        value="### 本ソフトウェアは検証評価用です。日常利用のための基本機能は備えていない点につきましてご理解をよろしくお願い申し上げます。",
        elem_classes="sub_Header")
    gr.Markdown(value="### Developed by Oracle Japan", elem_classes="sub_Header")
    tab_create_oci_clear_button.add(
        [
            tab_create_oci_cred_user_ocid_text,
            tab_create_oci_cred_tenancy_ocid_text,
            tab_create_oci_cred_fingerprint_text,
            tab_create_oci_cred_private_key_file
        ]
    )
    tab_create_oci_cred_button.click(
        create_oci_cred,
        inputs=[
            tab_create_oci_cred_user_ocid_text,
            tab_create_oci_cred_tenancy_ocid_text,
            tab_create_oci_cred_fingerprint_text,
            tab_create_oci_cred_private_key_file,
            tab_create_oci_cred_region_text,
        ],
        outputs=[
            tab_create_oci_cred_sql_accordion,
            tab_create_oci_cred_sql_text
        ]
    )
    tab_create_oci_cred_test_button.click(
        test_oci_cred,
        inputs=[
            tab_create_oci_cred_test_query_text
        ],
        outputs=[
            tab_create_oci_cred_test_vector_text
        ]
    )
    tab_create_cohere_cred_button.click(
        create_cohere_cred,
        inputs=[
            tab_create_cohere_cred_api_key_text
        ],
        outputs=[
            tab_create_cohere_cred_api_key_text
        ]
    )
    tab_create_openai_cred_button.click(
        create_openai_cred,
        inputs=[
            tab_create_openai_cred_base_url_text,
            tab_create_openai_cred_api_key_text
        ],
        outputs=[
            tab_create_openai_cred_base_url_text,
            tab_create_openai_cred_api_key_text
        ]
    )
    tab_create_azure_openai_cred_button.click(
        create_azure_openai_cred,
        inputs=[
            tab_create_azure_openai_cred_api_key_text,
            tab_create_azure_openai_cred_endpoint_gpt_4o_text,
            tab_create_azure_openai_cred_endpoint_gpt_4_text,
        ],
        outputs=[
            tab_create_azure_openai_cred_api_key_text,
            tab_create_azure_openai_cred_endpoint_gpt_4o_text,
            tab_create_azure_openai_cred_endpoint_gpt_4_text,
        ]
    )

    tab_create_langfuse_cred_button.click(
        create_langfuse_cred,
        inputs=[
            tab_create_langfuse_cred_secret_key_text,
            tab_create_langfuse_cred_public_key_text,
            tab_create_langfuse_cred_host_text
        ],
        outputs=[
            tab_create_langfuse_cred_secret_key_text,
            tab_create_langfuse_cred_public_key_text,
            tab_create_langfuse_cred_host_text
        ]
    )
    tab_chat_with_llm_answer_checkbox_group.change(
        set_chat_llm_answer,
        inputs=[
            tab_chat_with_llm_answer_checkbox_group
        ],
        outputs=[
            tab_chat_with_llm_xai_grok_3_accordion,
            tab_chat_with_llm_command_a_accordion,
            tab_chat_with_llm_llama_4_maverick_accordion,
            tab_chat_with_llm_llama_4_scout_accordion,
            tab_chat_with_llm_llama_3_3_70b_accordion,
            tab_chat_with_llm_llama_3_2_90b_vision_accordion,
            tab_chat_with_llm_openai_gpt4o_accordion,
            tab_chat_with_llm_openai_gpt4_accordion,
            tab_chat_with_llm_azure_openai_gpt4o_accordion,
            tab_chat_with_llm_azure_openai_gpt4_accordion
        ]
    )
    tab_chat_with_llm_clear_button.add(
        [
            tab_chat_with_llm_query_image,
            tab_chat_with_llm_query_text,
            tab_chat_with_llm_answer_checkbox_group,
            tab_chat_with_xai_grok_3_answer_text,
            tab_chat_with_command_a_answer_text,
            tab_chat_with_llama_4_maverick_answer_text,
            tab_chat_with_llama_4_scout_answer_text,
            tab_chat_with_llama_3_3_70b_answer_text,
            tab_chat_with_llama_3_2_90b_vision_answer_text,
            tab_chat_with_openai_gpt4o_answer_text,
            tab_chat_with_openai_gpt4_answer_text,
            tab_chat_with_azure_openai_gpt4o_answer_text,
            tab_chat_with_azure_openai_gpt4_answer_text
        ]
    )
    tab_chat_with_llm_chat_button.click(
        chat_stream,
        inputs=[
            tab_chat_with_llm_system_text,
            tab_chat_with_llm_query_image,
            tab_chat_with_llm_query_text,
            tab_chat_with_llm_answer_checkbox_group
        ],
        outputs=[
            tab_chat_with_xai_grok_3_answer_text,
            tab_chat_with_command_a_answer_text,
            tab_chat_with_llama_4_maverick_answer_text,
            tab_chat_with_llama_4_scout_answer_text,
            tab_chat_with_llama_3_3_70b_answer_text,
            tab_chat_with_llama_3_2_90b_vision_answer_text,
            tab_chat_with_openai_gpt4o_answer_text,
            tab_chat_with_openai_gpt4_answer_text,
            tab_chat_with_azure_openai_gpt4o_answer_text,
            tab_chat_with_azure_openai_gpt4_answer_text
        ]
    )

    tab_create_table_button.click(
        create_table,
        inputs=[],
        outputs=[
            tab_create_table_sql_accordion,
            tab_create_table_sql_text
        ]
    )

    tab_convert_document_convert_by_markitdown_button.click(
        convert_to_markdown_document,
        inputs=[
            tab_convert_document_convert_by_markitdown_file_text,
            tab_convert_document_convert_by_markitdown_use_llm_checkbox,
            tab_convert_document_convert_by_markitdown_llm_prompt_text,
        ],
        outputs=[
            tab_convert_document_convert_by_markitdown_file_text,
            tab_load_document_file_text,
        ],
    )

    tab_convert_document_convert_button.click(
        convert_excel_to_text_document,
        inputs=[
            tab_convert_document_convert_excel_to_text_file_text,
        ],
        outputs=[
            tab_convert_document_convert_excel_to_text_file_text,
            tab_load_document_file_text,
        ],
    )

    tab_load_document_load_button.click(
        load_document,
        inputs=[
            tab_load_document_file_text,
            tab_load_document_server_directory_text,
            tab_load_document_metadata_text,
        ],
        outputs=[
            tab_load_document_output_sql_text,
            tab_load_document_doc_id_text,
            tab_load_document_page_count_text,
            tab_load_document_page_content_text
        ],
    )
    tab_split_document.select(
        refresh_doc_list,
        inputs=[],
        outputs=[
            tab_split_document_doc_id_radio,
            tab_delete_document_doc_ids_checkbox_group,
            tab_chat_document_doc_id_checkbox_group
        ]
    ).then(
        reset_document_chunks_result_dataframe,
        inputs=[],
        outputs=[
            tab_split_document_chunks_result_dataframe,
        ]
    ).then(
        reset_document_chunks_result_detail,
        inputs=[],
        outputs=[
            tab_split_document_chunks_result_detail_chunk_id,
            tab_split_document_chunks_result_detail_chunk_data,
        ]
    )

    tab_split_document_split_button.click(
        split_document_by_unstructured,
        inputs=[
            tab_split_document_doc_id_radio,
            tab_split_document_chunks_by_radio,
            tab_split_document_chunks_max_slider,
            tab_split_document_chunks_overlap_slider,
            tab_split_document_chunks_split_by_radio,
            tab_split_document_chunks_split_by_custom_text,
            tab_split_document_chunks_language_radio,
            tab_split_document_chunks_normalize_radio,
            tab_split_document_chunks_normalize_options_checkbox_group
        ],
        outputs=[
            tab_split_document_output_sql_text,
            tab_split_document_chunks_count,
            tab_split_document_chunks_result_dataframe
        ]
    ).then(
        reset_document_chunks_result_detail,
        inputs=[],
        outputs=[
            tab_split_document_chunks_result_detail_chunk_id,
            tab_split_document_chunks_result_detail_chunk_data,
        ]
    )

    tab_split_document_chunks_result_dataframe.select(
        on_select_split_document_chunks_result,
        inputs=[
            tab_split_document_chunks_result_dataframe,
        ],
        outputs=[
            tab_split_document_chunks_result_detail_chunk_id,
            tab_split_document_chunks_result_detail_chunk_data,
        ]
    )

    tab_split_document_chunks_result_detail_update_button.click(
        update_document_chunks_result_detail,
        inputs=[
            tab_split_document_doc_id_radio,
            tab_split_document_chunks_result_dataframe,
            tab_split_document_chunks_result_detail_chunk_id,
            tab_split_document_chunks_result_detail_chunk_data,
        ],
        outputs=[
            tab_split_document_chunks_result_dataframe,
            tab_split_document_chunks_result_detail_chunk_id,
            tab_split_document_chunks_result_detail_chunk_data,
        ]
    )

    # tab_split_document_embed_save_button.click(
    #     split_document_by_unstructured,
    #     inputs=[
    #         tab_split_document_doc_id_radio,
    #         tab_split_document_chunks_by_radio,
    #         tab_split_document_chunks_max_slider,
    #         tab_split_document_chunks_overlap_slider,
    #         tab_split_document_chunks_split_by_radio,
    #         tab_split_document_chunks_split_by_custom_text,
    #         tab_split_document_chunks_language_radio,
    #         tab_split_document_chunks_normalize_radio,
    #         tab_split_document_chunks_normalize_options_checkbox_group
    #     ],
    #     outputs=[
    #         tab_split_document_output_sql_text,
    #         tab_split_document_chunks_count,
    #         tab_split_document_chunks_result_dataframe
    #     ],
    # ).then(
    tab_split_document_embed_save_button.click(
        embed_save_document_by_unstructured,
        inputs=[
            tab_split_document_doc_id_radio,
            tab_split_document_chunks_by_radio,
            tab_split_document_chunks_max_slider,
            tab_split_document_chunks_overlap_slider,
            tab_split_document_chunks_split_by_radio,
            tab_split_document_chunks_split_by_custom_text,
            tab_split_document_chunks_language_radio,
            tab_split_document_chunks_normalize_radio,
            tab_split_document_chunks_normalize_options_checkbox_group
        ],
        outputs=[
            tab_split_document_output_sql_text,
            tab_split_document_chunks_count,
            tab_split_document_chunks_result_dataframe
        ],
    )

    tab_delete_document.select(
        refresh_doc_list,
        inputs=[],
        outputs=[
            tab_split_document_doc_id_radio,
            tab_delete_document_doc_ids_checkbox_group,
            tab_chat_document_doc_id_checkbox_group
        ]
    )
    tab_delete_document_delete_button.click(
        delete_document,
        inputs=[
            tab_delete_document_server_directory_text,
            tab_delete_document_doc_ids_checkbox_group
        ],
        outputs=[
            tab_delete_document_delete_sql,
            tab_split_document_doc_id_radio,
            tab_delete_document_doc_ids_checkbox_group
        ]
    )
    tab_chat_document.select(
        refresh_doc_list,
        inputs=[],
        outputs=[
            tab_split_document_doc_id_radio,
            tab_delete_document_doc_ids_checkbox_group,
            tab_chat_document_doc_id_checkbox_group
        ]
    )
    tab_chat_document_doc_id_all_checkbox.change(
        lambda x: gr.CheckboxGroup(interactive=False, value=[]) if x else
        gr.CheckboxGroup(interactive=True, value=[]),
        tab_chat_document_doc_id_all_checkbox,
        tab_chat_document_doc_id_checkbox_group
    )
    tab_chat_document_llm_answer_checkbox_group.change(
        set_chat_llm_answer,
        inputs=[
            tab_chat_document_llm_answer_checkbox_group
        ],
        outputs=[
            tab_chat_document_llm_xai_grok_3_accordion,
            tab_chat_document_llm_command_a_accordion,
            tab_chat_document_llm_llama_4_maverick_accordion,
            tab_chat_document_llm_llama_4_scout_accordion,
            tab_chat_document_llm_llama_3_3_70b_accordion,
            tab_chat_document_llm_llama_3_2_90b_vision_accordion,
            tab_chat_document_llm_openai_gpt4o_accordion,
            tab_chat_document_llm_openai_gpt4_accordion,
            tab_chat_document_llm_azure_openai_gpt4o_accordion,
            tab_chat_document_llm_azure_openai_gpt4_accordion
        ]
    ).then(
        set_image_answer_visibility,
        inputs=[
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_use_image_checkbox
        ],
        outputs=[
            tab_chat_document_llm_llama_4_maverick_image_accordion,
            tab_chat_document_llm_llama_4_scout_image_accordion,
            tab_chat_document_llm_llama_3_2_90b_vision_image_accordion,
            tab_chat_document_llm_openai_gpt4o_image_accordion,
            tab_chat_document_llm_azure_openai_gpt4o_image_accordion
        ]
    )
    tab_chat_document_llm_evaluation_checkbox.change(
        lambda x: (
            gr.Textbox(visible=True, interactive=True),
            gr.Textbox(visible=True, interactive=True, value="")
        ) if x else (
            gr.Textbox(visible=False, interactive=False),
            gr.Textbox(visible=False, interactive=False, value="")
        ),
        tab_chat_document_llm_evaluation_checkbox,
        [
            tab_chat_document_system_message_text,
            tab_chat_document_standard_answer_text
        ]
    ).then(
        set_chat_llm_evaluation,
        inputs=[
            tab_chat_document_llm_evaluation_checkbox
        ],
        outputs=[
            tab_chat_document_llm_xai_grok_3_evaluation_accordion,
            tab_chat_document_llm_command_a_evaluation_accordion,
            tab_chat_document_llm_llama_4_maverick_evaluation_accordion,
            tab_chat_document_llm_llama_4_scout_evaluation_accordion,
            tab_chat_document_llm_llama_3_3_70b_evaluation_accordion,
            tab_chat_document_llm_llama_3_2_90b_vision_evaluation_accordion,
            tab_chat_document_llm_openai_gpt4o_evaluation_accordion,
            tab_chat_document_llm_openai_gpt4_evaluation_accordion,
            tab_chat_document_llm_azure_openai_gpt4o_evaluation_accordion,
            tab_chat_document_llm_azure_openai_gpt4_evaluation_accordion
        ]
    )

    tab_chat_document_generate_query_radio.change(
        lambda x: (gr.Textbox(value=""),
                   gr.Textbox(value=""),
                   gr.Textbox(value="")),
        inputs=[
            tab_chat_document_generate_query_radio
        ],
        outputs=[
            tab_chat_document_sub_query1_text,
            tab_chat_document_sub_query2_text,
            tab_chat_document_sub_query3_text
        ]
    )

    # 画像を使って回答チェックボックスの変更イベント
    tab_chat_document_use_image_checkbox.change(
        lambda x: gr.Accordion(visible=True) if x else gr.Accordion(visible=False),  # 画像 Prompt 設定の表示/非表示のみ制御
        inputs=[tab_chat_document_use_image_checkbox],
        outputs=[
            tab_chat_document_image_prompt_accordion
        ]
    ).then(
        set_image_answer_visibility,
        inputs=[
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_use_image_checkbox
        ],
        outputs=[
            tab_chat_document_llm_llama_4_maverick_image_accordion,
            tab_chat_document_llm_llama_4_scout_image_accordion,
            tab_chat_document_llm_llama_3_2_90b_vision_image_accordion,
            tab_chat_document_llm_openai_gpt4o_image_accordion,
            tab_chat_document_llm_azure_openai_gpt4o_image_accordion
        ]
    )

    # tab_chat_document_chat_document_button.click(
    #     check_chat_document_input,
    #     inputs=[
    #         tab_chat_document_llm_answer_checkbox_group,
    #         tab_chat_document_llm_evaluation_checkbox,
    #         tab_chat_document_system_message_text,
    #         tab_chat_document_standard_answer_text,
    #     ],
    #     outputs=[]
    # ).then(
    tab_chat_document_chat_document_button.click(
        lambda: gr.DownloadButton(visible=False),
        outputs=[tab_chat_document_download_output_button]
    ).then(
        reset_all_llm_messages,
        inputs=[],
        outputs=[
            tab_chat_document_xai_grok_3_answer_text,
            tab_chat_document_command_a_answer_text,
            tab_chat_document_llama_4_maverick_answer_text,
            tab_chat_document_llama_4_scout_answer_text,
            tab_chat_document_llama_3_3_70b_answer_text,
            tab_chat_document_llama_3_2_90b_vision_answer_text,
            tab_chat_document_openai_gpt4o_answer_text,
            tab_chat_document_openai_gpt4_answer_text,
            tab_chat_document_azure_openai_gpt4o_answer_text,
            tab_chat_document_azure_openai_gpt4_answer_text,
        ]
    ).then(
        reset_image_answers,
        inputs=[],
        outputs=[
            tab_chat_document_llama_4_maverick_image_answer_text,
            tab_chat_document_llama_4_scout_image_answer_text,
            tab_chat_document_llama_3_2_90b_vision_image_answer_text,
            tab_chat_document_openai_gpt4o_image_answer_text,
            tab_chat_document_azure_openai_gpt4o_image_answer_text
        ]
    ).then(
        reset_llm_evaluations,
        inputs=[],
        outputs=[
            tab_chat_document_xai_grok_3_evaluation_text,
            tab_chat_document_command_a_evaluation_text,
            tab_chat_document_llama_4_maverick_evaluation_text,
            tab_chat_document_llama_4_scout_evaluation_text,
            tab_chat_document_llama_3_3_70b_evaluation_text,
            tab_chat_document_llama_3_2_90b_vision_evaluation_text,
            tab_chat_document_openai_gpt4o_evaluation_text,
            tab_chat_document_openai_gpt4_evaluation_text,
            tab_chat_document_azure_openai_gpt4o_evaluation_text,
            tab_chat_document_azure_openai_gpt4_evaluation_text
        ]
    ).then(
        reset_eval_by_human_result,
        inputs=[],
        outputs=[
            tab_chat_document_command_a_answer_human_eval_feedback_radio,
            tab_chat_document_command_a_answer_human_eval_feedback_text,
            tab_chat_document_llama_4_maverick_answer_human_eval_feedback_radio,
            tab_chat_document_llama_4_maverick_answer_human_eval_feedback_text,
            tab_chat_document_llama_4_scout_answer_human_eval_feedback_radio,
            tab_chat_document_llama_4_scout_answer_human_eval_feedback_text,
            tab_chat_document_llama_3_3_70b_answer_human_eval_feedback_radio,
            tab_chat_document_llama_3_3_70b_answer_human_eval_feedback_text,
            tab_chat_document_llama_3_2_90b_vision_answer_human_eval_feedback_radio,
            tab_chat_document_llama_3_2_90b_vision_answer_human_eval_feedback_text,
            tab_chat_document_openai_gpt4o_answer_human_eval_feedback_radio,
            tab_chat_document_openai_gpt4o_answer_human_eval_feedback_text,
            tab_chat_document_openai_gpt4_answer_human_eval_feedback_radio,
            tab_chat_document_openai_gpt4_answer_human_eval_feedback_text,
            tab_chat_document_azure_openai_gpt4o_answer_human_eval_feedback_radio,
            tab_chat_document_azure_openai_gpt4o_answer_human_eval_feedback_text,
            tab_chat_document_azure_openai_gpt4_answer_human_eval_feedback_radio,
            tab_chat_document_azure_openai_gpt4_answer_human_eval_feedback_text,
        ]
    ).then(
        generate_query,
        inputs=[
            tab_chat_document_query_text,
            tab_chat_document_generate_query_radio
        ],
        outputs=[
            tab_chat_document_sub_query1_text,
            tab_chat_document_sub_query2_text,
            tab_chat_document_sub_query3_text,
        ]
    ).then(
        search_document,
        inputs=[
            tab_chat_document_reranker_model_radio,
            tab_chat_document_reranker_top_k_slider,
            tab_chat_document_reranker_threshold_slider,
            tab_chat_document_threshold_value_slider,
            tab_chat_document_top_k_slider,
            tab_chat_document_doc_id_all_checkbox,
            tab_chat_document_doc_id_checkbox_group,
            tab_chat_document_text_search_checkbox,
            tab_chat_document_text_search_k_slider,
            tab_chat_document_document_metadata_text,
            tab_chat_document_query_text,
            tab_chat_document_sub_query1_text,
            tab_chat_document_sub_query2_text,
            tab_chat_document_sub_query3_text,
            tab_chat_document_partition_by_k_slider,
            tab_chat_document_answer_by_one_checkbox,
            tab_chat_document_extend_first_chunk_size,
            tab_chat_document_extend_around_chunk_size,
            tab_chat_document_use_image_checkbox
        ],
        outputs=[
            tab_chat_document_output_sql_text,
            tab_chat_document_searched_data_summary_text,
            tab_chat_document_searched_result_dataframe
        ]
    ).then(
        chat_document,
        inputs=[
            tab_chat_document_searched_result_dataframe,
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_include_citation_checkbox,
            tab_chat_document_include_current_time_checkbox,
            tab_chat_document_use_image_checkbox,
            tab_chat_document_query_text,
            tab_chat_document_doc_id_all_checkbox,
            tab_chat_document_doc_id_checkbox_group,
            tab_chat_document_rag_prompt_text,
        ],
        outputs=[
            tab_chat_document_xai_grok_3_answer_text,
            tab_chat_document_command_a_answer_text,
            tab_chat_document_llama_4_maverick_answer_text,
            tab_chat_document_llama_4_scout_answer_text,
            tab_chat_document_llama_3_3_70b_answer_text,
            tab_chat_document_llama_3_2_90b_vision_answer_text,
            tab_chat_document_openai_gpt4o_answer_text,
            tab_chat_document_openai_gpt4_answer_text,
            tab_chat_document_azure_openai_gpt4o_answer_text,
            tab_chat_document_azure_openai_gpt4_answer_text,
        ]
    ).then(
        append_citation,
        inputs=[
            tab_chat_document_searched_result_dataframe,
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_include_citation_checkbox,
            tab_chat_document_query_text,
            tab_chat_document_doc_id_all_checkbox,
            tab_chat_document_doc_id_checkbox_group,
            tab_chat_document_xai_grok_3_answer_text,
            tab_chat_document_command_a_answer_text,
            tab_chat_document_llama_4_maverick_answer_text,
            tab_chat_document_llama_4_scout_answer_text,
            tab_chat_document_llama_3_3_70b_answer_text,
            tab_chat_document_llama_3_2_90b_vision_answer_text,
            tab_chat_document_openai_gpt4o_answer_text,
            tab_chat_document_openai_gpt4_answer_text,
            tab_chat_document_azure_openai_gpt4o_answer_text,
            tab_chat_document_azure_openai_gpt4_answer_text,
        ],
        outputs=[
            tab_chat_document_xai_grok_3_answer_text,
            tab_chat_document_command_a_answer_text,
            tab_chat_document_llama_4_maverick_answer_text,
            tab_chat_document_llama_4_scout_answer_text,
            tab_chat_document_llama_3_3_70b_answer_text,
            tab_chat_document_llama_3_2_90b_vision_answer_text,
            tab_chat_document_openai_gpt4o_answer_text,
            tab_chat_document_openai_gpt4_answer_text,
            tab_chat_document_azure_openai_gpt4o_answer_text,
            tab_chat_document_azure_openai_gpt4_answer_text,
        ]
    ).then(
        process_image_answers_streaming,
        inputs=[
            tab_chat_document_searched_result_dataframe,
            tab_chat_document_use_image_checkbox,
            tab_chat_document_single_image_processing_checkbox,
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_query_text,
            tab_chat_document_llama_4_maverick_image_answer_text,
            tab_chat_document_llama_4_scout_image_answer_text,
            tab_chat_document_llama_3_2_90b_vision_image_answer_text,
            tab_chat_document_openai_gpt4o_image_answer_text,
            tab_chat_document_azure_openai_gpt4o_image_answer_text,
            tab_chat_document_image_prompt_text
        ],
        outputs=[
            tab_chat_document_llama_4_maverick_image_answer_text,
            tab_chat_document_llama_4_scout_image_answer_text,
            tab_chat_document_llama_3_2_90b_vision_image_answer_text,
            tab_chat_document_openai_gpt4o_image_answer_text,
            tab_chat_document_azure_openai_gpt4o_image_answer_text
        ]
    ).then(
        eval_by_ragas,
        inputs=[
            tab_chat_document_query_text,
            tab_chat_document_doc_id_all_checkbox,
            tab_chat_document_doc_id_checkbox_group,
            tab_chat_document_searched_result_dataframe,
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_llm_evaluation_checkbox,
            tab_chat_document_system_message_text,
            tab_chat_document_standard_answer_text,
            tab_chat_document_xai_grok_3_answer_text,
            tab_chat_document_command_a_answer_text,
            tab_chat_document_llama_4_maverick_answer_text,
            tab_chat_document_llama_4_scout_answer_text,
            tab_chat_document_llama_3_3_70b_answer_text,
            tab_chat_document_llama_3_2_90b_vision_answer_text,
            tab_chat_document_openai_gpt4o_answer_text,
            tab_chat_document_openai_gpt4_answer_text,
            tab_chat_document_azure_openai_gpt4o_answer_text,
            tab_chat_document_azure_openai_gpt4_answer_text,
        ],
        outputs=[
            tab_chat_document_xai_grok_3_evaluation_text,
            tab_chat_document_command_a_evaluation_text,
            tab_chat_document_llama_4_maverick_evaluation_text,
            tab_chat_document_llama_4_scout_evaluation_text,
            tab_chat_document_llama_3_3_70b_evaluation_text,
            tab_chat_document_llama_3_2_90b_vision_evaluation_text,
            tab_chat_document_openai_gpt4o_evaluation_text,
            tab_chat_document_openai_gpt4_evaluation_text,
            tab_chat_document_azure_openai_gpt4o_evaluation_text,
            tab_chat_document_azure_openai_gpt4_evaluation_text,
        ]
    ).then(
        generate_download_file,
        inputs=[
            tab_chat_document_searched_result_dataframe,
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_include_citation_checkbox,
            tab_chat_document_llm_evaluation_checkbox,
            tab_chat_document_query_text,
            tab_chat_document_doc_id_all_checkbox,
            tab_chat_document_doc_id_checkbox_group,
            tab_chat_document_standard_answer_text,
            tab_chat_document_xai_grok_3_answer_text,
            tab_chat_document_command_a_answer_text,
            tab_chat_document_llama_4_maverick_answer_text,
            tab_chat_document_llama_4_scout_answer_text,
            tab_chat_document_llama_3_3_70b_answer_text,
            tab_chat_document_llama_3_2_90b_vision_answer_text,
            tab_chat_document_openai_gpt4o_answer_text,
            tab_chat_document_openai_gpt4_answer_text,
            tab_chat_document_azure_openai_gpt4o_answer_text,
            tab_chat_document_azure_openai_gpt4_answer_text,
            tab_chat_document_xai_grok_3_evaluation_text,
            tab_chat_document_command_a_evaluation_text,
            tab_chat_document_llama_4_maverick_evaluation_text,
            tab_chat_document_llama_4_scout_evaluation_text,
            tab_chat_document_llama_3_3_70b_evaluation_text,
            tab_chat_document_llama_3_2_90b_vision_evaluation_text,
            tab_chat_document_openai_gpt4o_evaluation_text,
            tab_chat_document_openai_gpt4_evaluation_text,
            tab_chat_document_azure_openai_gpt4o_evaluation_text,
            tab_chat_document_azure_openai_gpt4_evaluation_text,
        ],
        outputs=[
            tab_chat_document_download_output_button
        ]
    ).then(
        set_query_id_state,
        inputs=[],
        outputs=[
            query_id_state,
        ]
    ).then(
        insert_query_result,
        inputs=[
            tab_chat_document_searched_result_dataframe,
            query_id_state,
            tab_chat_document_query_text,
            tab_chat_document_doc_id_all_checkbox,
            tab_chat_document_doc_id_checkbox_group,
            tab_chat_document_output_sql_text,
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_llm_evaluation_checkbox,
            tab_chat_document_standard_answer_text,
            tab_chat_document_xai_grok_3_answer_text,
            tab_chat_document_command_a_answer_text,
            tab_chat_document_llama_4_maverick_answer_text,
            tab_chat_document_llama_4_scout_answer_text,
            tab_chat_document_llama_3_3_70b_answer_text,
            tab_chat_document_llama_3_2_90b_vision_answer_text,
            tab_chat_document_openai_gpt4o_answer_text,
            tab_chat_document_openai_gpt4_answer_text,
            tab_chat_document_azure_openai_gpt4o_answer_text,
            tab_chat_document_azure_openai_gpt4_answer_text,
            tab_chat_document_xai_grok_3_evaluation_text,
            tab_chat_document_command_a_evaluation_text,
            tab_chat_document_llama_4_maverick_evaluation_text,
            tab_chat_document_llama_4_scout_evaluation_text,
            tab_chat_document_llama_3_3_70b_evaluation_text,
            tab_chat_document_llama_3_2_90b_vision_evaluation_text,
            tab_chat_document_openai_gpt4o_evaluation_text,
            tab_chat_document_openai_gpt4_evaluation_text,
            tab_chat_document_azure_openai_gpt4o_evaluation_text,
            tab_chat_document_azure_openai_gpt4_evaluation_text,
        ],
        outputs=[]
    )

    tab_chat_document_xai_grok_3_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="xai/grok-3"),
            tab_chat_document_xai_grok_3_answer_human_eval_feedback_radio,
            tab_chat_document_xai_grok_3_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_xai_grok_3_answer_human_eval_feedback_radio,
            tab_chat_document_xai_grok_3_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_command_a_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="cohere/command-a"),
            tab_chat_document_command_a_answer_human_eval_feedback_radio,
            tab_chat_document_command_a_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_command_a_answer_human_eval_feedback_radio,
            tab_chat_document_command_a_answer_human_eval_feedback_text,
        ]
    )


    tab_chat_document_llama_4_maverick_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="meta/llama-4-maverick-17b-128e-instruct-fp8"),
            tab_chat_document_llama_4_maverick_answer_human_eval_feedback_radio,
            tab_chat_document_llama_4_maverick_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_llama_4_maverick_answer_human_eval_feedback_radio,
            tab_chat_document_llama_4_maverick_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_llama_4_scout_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="meta/llama-4-scout-17b-16e-instruct"),
            tab_chat_document_llama_4_scout_answer_human_eval_feedback_radio,
            tab_chat_document_llama_4_scout_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_llama_4_scout_answer_human_eval_feedback_radio,
            tab_chat_document_llama_4_scout_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_llama_3_3_70b_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="meta/llama-3-3-70b"),
            tab_chat_document_llama_3_3_70b_answer_human_eval_feedback_radio,
            tab_chat_document_llama_3_3_70b_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_llama_3_3_70b_answer_human_eval_feedback_radio,
            tab_chat_document_llama_3_3_70b_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_llama_3_2_90b_vision_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="meta/llama-3-2-90b-vision"),
            tab_chat_document_llama_3_2_90b_vision_answer_human_eval_feedback_radio,
            tab_chat_document_llama_3_2_90b_vision_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_llama_3_2_90b_vision_answer_human_eval_feedback_radio,
            tab_chat_document_llama_3_2_90b_vision_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_openai_gpt4o_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="openai/gpt-4o"),
            tab_chat_document_openai_gpt4o_answer_human_eval_feedback_radio,
            tab_chat_document_openai_gpt4o_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_openai_gpt4o_answer_human_eval_feedback_radio,
            tab_chat_document_openai_gpt4o_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_openai_gpt4_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="openai/gpt-4"),
            tab_chat_document_openai_gpt4_answer_human_eval_feedback_radio,
            tab_chat_document_openai_gpt4_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_openai_gpt4_answer_human_eval_feedback_radio,
            tab_chat_document_openai_gpt4_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_azure_openai_gpt4o_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="azure_openai/gpt-4o"),
            tab_chat_document_azure_openai_gpt4o_answer_human_eval_feedback_radio,
            tab_chat_document_azure_openai_gpt4o_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_azure_openai_gpt4o_answer_human_eval_feedback_radio,
            tab_chat_document_azure_openai_gpt4o_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_azure_openai_gpt4_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="azure_openai/gpt-4"),
            tab_chat_document_azure_openai_gpt4_answer_human_eval_feedback_radio,
            tab_chat_document_azure_openai_gpt4_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_azure_openai_gpt4_answer_human_eval_feedback_radio,
            tab_chat_document_azure_openai_gpt4_answer_human_eval_feedback_text,
        ]
    )

    tab_download_eval_result_generate_button.click(
        generate_eval_result_file,
        inputs=[],
        outputs=[
            tab_download_eval_result_download_button,
        ]
    )


    # RAG Prompt 設定のイベントハンドラー
    def save_rag_prompt(prompt_text):
        """RAG promptを保存する"""
        try:
            update_langgpt_rag_prompt(prompt_text)
            return gr.Info("Promptが保存されました。")
        except Exception as e:
            return gr.Warning(f"Promptの保存に失敗しました: {str(e)}")


    def reset_rag_prompt():
        """RAG promptをデフォルトに戻す"""
        default_prompt = get_langgpt_rag_prompt("{{context}}", "{{query_text}}", False, False)
        update_langgpt_rag_prompt(default_prompt)
        return default_prompt


    tab_chat_document_rag_prompt_save_button.click(
        save_rag_prompt,
        inputs=[tab_chat_document_rag_prompt_text],
        outputs=[]
    )

    tab_chat_document_rag_prompt_reset_button.click(
        reset_rag_prompt,
        inputs=[],
        outputs=[tab_chat_document_rag_prompt_text]
    )


    # 画像 Prompt 設定のイベントハンドラー
    def save_image_prompt(prompt_text):
        """画像 promptを保存する"""
        try:
            update_image_qa_prompt(prompt_text)
            return gr.Info("画像 Promptが保存されました。")
        except Exception as e:
            return gr.Warning(f"画像 Promptの保存に失敗しました: {str(e)}")


    def reset_image_prompt():
        """画像 promptをデフォルトに戻す"""
        default_prompt = get_image_qa_prompt("{{query_text}}")
        update_image_qa_prompt(default_prompt)
        return default_prompt


    # tab_chat_document_image_prompt_save_button.click(
    #     save_image_prompt,
    #     inputs=[tab_chat_document_image_prompt_text],
    #     outputs=[]
    # )

    tab_chat_document_image_prompt_reset_button.click(
        reset_image_prompt,
        inputs=[],
        outputs=[tab_chat_document_image_prompt_text]
    )

app.queue()
if __name__ == "__main__":
    # リソース警告追跡を有効化
    enable_resource_warnings()

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        max_threads=200,
        show_api=False,
        # auth=do_auth,
    )
