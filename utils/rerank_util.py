"""
OCI Generative AI および Cohere Rerank ユーティリティモジュール

このモジュールは、Oracle Cloud Infrastructure (OCI) の Generative AI サービスと
Cohere API を使用してドキュメントの再ランキングを行うための関数を提供します。
"""

import os
from typing import List

import cohere
import oci

from utils.common_util import get_region


def rerank_documents_response(input_text, inputs: List[str], rerank_model):
    """
    指定されたモデルを使用してドキュメントを再ランキングする
    
    Args:
        input_text (str): クエリテキスト
        inputs (List[str]): 再ランキング対象のドキュメントリスト
        rerank_model (str): 使用する再ランキングモデル名
        
    Returns:
        List[dict]: 再ランキング結果のリスト。各要素は以下の形式：
            {
                "document": str,  # ドキュメント内容
                "index": int,     # 元のリスト内でのインデックス
                "relevance_score": float  # 関連性スコア
            }
    """
    all_document_ranks = []
    batch_size = 200

    if rerank_model in ["cohere/rerank-multilingual-v3.1", "cohere/rerank-english-v3.1"]:
        # OCI Generative AI を使用した再ランキング
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
        # Cohere API を使用した再ランキング
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
