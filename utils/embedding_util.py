"""
OCI Generative AI Embedding ユーティリティモジュール

このモジュールは、Oracle Cloud Infrastructure (OCI) の Generative AI サービスを使用して
テキストと画像のembeddingを生成するための関数を提供します。
"""

import os
import time
from typing import List

import gradio as gr
import oci

from utils.common_util import get_region


def generate_embedding_response(inputs: List[str]):
    """
    テキストからembeddingを生成する関数

    Args:
        inputs (List[str]): embedding生成対象のテキストリスト

    Returns:
        List: 生成されたembeddingのリスト（FLOAT32形式）
    """
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
        embed_text_detail.input_type = "SEARCH_DOCUMENT"
        embed_text_detail.inputs = batch
        embed_text_detail.truncate = "NONE"
        embed_text_detail.compartment_id = os.environ["OCI_COMPARTMENT_OCID"]

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)
                print(f"バッチ {i // batch_size + 1} / {(len(inputs) - 1) // batch_size + 1} を処理しました")
                all_embeddings.extend(embed_text_response.data.embeddings)
                break
            except Exception as e:
                print(f"例外が発生しました: {e}")
                retry_count += 1
                print(f"テキストembedding生成エラー: {e}. 再試行中 ({retry_count}/{max_retries})...")
                time.sleep(10 * retry_count)
                if retry_count == max_retries:
                    gr.Warning("保存中にエラーが発生しました。しばらくしてから再度お試しください。")
                    all_embeddings = []
                    return all_embeddings

        time.sleep(1)

    # FLOAT32形式に変換してOracle DBとの互換性を確保
    import array
    converted_embeddings = []
    for embedding in all_embeddings:
        # OCI APIから返されるembeddingをFLOAT32配列に変換
        converted_embeddings.append(array.array("f", embedding))

    return converted_embeddings


def generate_image_embedding_response(image_inputs: List[str], input_type: str = "IMAGE"):
    """
    画像からembeddingを生成する関数

    Args:
        image_inputs (List[str]): base64エンコードされた画像のリスト
        input_type (str): 入力タイプ（デフォルト: "IMAGE"）

    Returns:
        List: 生成されたembeddingのリスト（FLOAT32形式）
    """
    config = oci.config.from_file('/root/.oci/config', "DEFAULT")
    region = get_region()
    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240))

    # 画像の場合はバッチサイズを小さくする（画像データが大きいため）
    batch_size = 1
    all_embeddings = []

    for i in range(0, len(image_inputs), batch_size):
        batch = image_inputs[i:i + batch_size]

        embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
        embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=os.environ["OCI_COHERE_EMBED_MODEL"]
        )
        embed_text_detail.input_type = input_type  # "IMAGE"を指定
        embed_text_detail.inputs = batch  # base64エンコードされた画像
        embed_text_detail.truncate = "NONE"
        embed_text_detail.compartment_id = os.environ["OCI_COMPARTMENT_OCID"]

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                embed_text_response = generative_ai_inference_client.embed_text(embed_text_detail)
                print(
                    f"画像embeddingバッチ {i // batch_size + 1} / {(len(image_inputs) - 1) // batch_size + 1} を処理しました")
                all_embeddings.extend(embed_text_response.data.embeddings)
                break
            except Exception as e:
                print(f"例外が発生しました: {e}")
                retry_count += 1
                print(f"画像embedding生成エラー: {e}. 再試行中 ({retry_count}/{max_retries})...")
                time.sleep(10 * retry_count)
                if retry_count == max_retries:
                    gr.Warning("画像embedding生成中にエラーが発生しました。しばらくしてから再度お試しください。")
                    all_embeddings = []
                    return all_embeddings

        time.sleep(1)

    # FLOAT32形式に変換してOracle DBとの互換性を確保
    import array
    converted_embeddings = []
    for embedding in all_embeddings:
        # OCI APIから返されるembeddingをFLOAT32配列に変換
        converted_embeddings.append(array.array("f", embedding))

    return converted_embeddings
