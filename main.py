import argparse
import logging
import os
import platform
import re
from typing import List, Tuple

import gradio as gr
import oracledb
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from gradio.themes import GoogleFont
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from my_langchain_community.chat_models import ChatOCIGenAI
from utils.auth_util import (
    do_auth,
    create_oci_cred as create_oci_cred_util, create_cohere_cred, create_openai_cred,
    create_azure_openai_cred, create_langfuse_cred, test_oci_cred as test_oci_cred_util
)
from utils.chat_document_util import chat_document as chat_document_util, append_citation as append_citation_util
from utils.chat_util import chat_stream
from utils.cleanup_util import (
    enable_resource_warnings
)
from utils.common_util import get_region
from utils.css_gradio_util import custom_css
from utils.database_util import create_table as create_table_util
from utils.document_conversion_util import (
    convert_pdf_to_markdown, convert_excel_to_text_document, convert_xml_to_text_document, convert_json_to_text_document
)
from utils.document_embed_util import embed_save_document_by_unstructured as embed_save_document_util
from utils.document_loader_util import load_document as load_document_util
from utils.document_management_util import (
    search_document as search_document_util, delete_document as delete_document_util,
    get_doc_list as get_doc_list_util, get_server_path as get_server_path_util
)
from utils.document_split_util import (
    reset_document_chunks_result_dataframe,
    split_document_by_unstructured as split_document_util,
    update_document_chunks_result_detail_with_validation
)
from utils.download_util import generate_download_file
from utils.embedding_util import (
    generate_embedding_response
)
from utils.evaluation_util import eval_by_human as eval_by_human_util, eval_by_ragas, reset_eval_by_human_result
from utils.generator_util import generate_unique_id
from utils.image_processing_util import (
    process_single_image_streaming as process_single_image_streaming_util,
    process_image_answers_streaming as process_image_answers_streaming_util
)
from utils.prompts_util import (
    get_sub_query_prompt, get_rag_fusion_prompt, get_hyde_prompt, get_step_back_prompt,
    get_langgpt_rag_prompt, get_llm_evaluation_system_message, get_chat_system_message,
    get_query_generation_prompt, update_langgpt_rag_prompt,
    get_image_qa_prompt, update_image_qa_prompt
)
from utils.query_util import insert_query_result as insert_query_result_util
from utils.text_util import (
    remove_base64_images_from_text
)

# read local .env file
load_dotenv(find_dotenv())

DEFAULT_COLLECTION_NAME = os.environ["DEFAULT_COLLECTION_NAME"]

# ログ設定
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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


def check_database_pool_health():
    """
    データベース接続プールの健康状態をチェックするためのラッパー関数
    """
    from utils.system_util import check_database_pool_health as check_db_health
    return check_db_health(pool)


def get_doc_list() -> List[Tuple[str, str]]:
    """
    データベースからドキュメントリストを取得するためのラッパー関数
    """
    return get_doc_list_util(pool, DEFAULT_COLLECTION_NAME)


def refresh_doc_list():
    doc_list = get_doc_list()
    return (
        gr.Radio(choices=doc_list, value=None),
        gr.CheckboxGroup(choices=doc_list, value=[]),
        gr.CheckboxGroup(choices=doc_list, value=[])
    )


def get_server_path(doc_id: str) -> str:
    """
    ドキュメントIDからサーバーパスを取得するためのラッパー関数
    """
    return get_server_path_util(pool, DEFAULT_COLLECTION_NAME, doc_id)


def set_chat_llm_answer(llm_answer_checkbox):
    oci_openai_gpt_5_answer_visible = False
    oci_openai_o3_answer_visible = False
    oci_openai_gpt_4_1_answer_visible = False
    oci_xai_grok_4_answer_visible = False
    oci_cohere_command_a_answer_visible = False
    oci_meta_llama_4_scout_answer_visible = False
    openai_gpt_4o_answer_visible = False
    azure_openai_gpt_4o_answer_visible = False
    if "oci_openai/gpt-5" in llm_answer_checkbox:
        oci_openai_gpt_5_answer_visible = True
    if "oci_openai/o3" in llm_answer_checkbox:
        oci_openai_o3_answer_visible = True
    if "oci_openai/gpt-4.1" in llm_answer_checkbox:
        oci_openai_gpt_4_1_answer_visible = True
    if "oci_xai/grok-4" in llm_answer_checkbox:
        oci_xai_grok_4_answer_visible = True
    if "oci_cohere/command-a" in llm_answer_checkbox:
        oci_cohere_command_a_answer_visible = True
    if "oci_meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox:
        oci_meta_llama_4_scout_answer_visible = True
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt_4o_answer_visible = True
    if "azure_openai/gpt-4o" in llm_answer_checkbox:
        azure_openai_gpt_4o_answer_visible = True
    return (
        gr.Accordion(visible=oci_openai_gpt_5_answer_visible),
        gr.Accordion(visible=oci_openai_o3_answer_visible),
        gr.Accordion(visible=oci_openai_gpt_4_1_answer_visible),
        gr.Accordion(visible=oci_xai_grok_4_answer_visible),
        gr.Accordion(visible=oci_cohere_command_a_answer_visible),
        gr.Accordion(visible=oci_meta_llama_4_scout_answer_visible),
        gr.Accordion(visible=openai_gpt_4o_answer_visible),
        gr.Accordion(visible=azure_openai_gpt_4o_answer_visible)
    )


def set_chat_llm_evaluation(llm_evaluation_checkbox):
    oci_openai_gpt_5_evaluation_visible = False
    oci_openai_o3_evaluation_visible = False
    oci_openai_gpt_4_1_evaluation_visible = False
    oci_xai_grok_4_evaluation_visible = False
    oci_cohere_command_a_evaluation_visible = False
    oci_meta_llama_4_scout_evaluation_visible = False
    openai_gpt_4o_evaluation_visible = False
    azure_openai_gpt_4o_evaluation_visible = False
    if llm_evaluation_checkbox:
        oci_openai_gpt_5_evaluation_visible = True
        oci_openai_o3_evaluation_visible = True
        oci_openai_gpt_4_1_evaluation_visible = True
        oci_xai_grok_4_evaluation_visible = True
        oci_cohere_command_a_evaluation_visible = True
        oci_meta_llama_4_scout_evaluation_visible = True
        openai_gpt_4o_evaluation_visible = True
        azure_openai_gpt_4o_evaluation_visible = True
    return (
        gr.Accordion(visible=oci_openai_gpt_5_evaluation_visible),
        gr.Accordion(visible=oci_openai_o3_evaluation_visible),
        gr.Accordion(visible=oci_openai_gpt_4_1_evaluation_visible),
        gr.Accordion(visible=oci_xai_grok_4_evaluation_visible),
        gr.Accordion(visible=oci_cohere_command_a_evaluation_visible),
        gr.Accordion(visible=oci_meta_llama_4_scout_evaluation_visible),
        gr.Accordion(visible=openai_gpt_4o_evaluation_visible),
        gr.Accordion(visible=azure_openai_gpt_4o_evaluation_visible),
    )


def set_image_answer_visibility(llm_answer_checkbox, use_image):
    """
    Vision 回答の可視性を制御する関数
    選択されたLLMモデルと「画像を使って回答」の状態に基づいて、
    対象のモデルのVision 回答Accordionの可視性を決定する
    """
    oci_openai_gpt_5_image_visible = False
    oci_openai_o3_image_visible = False
    oci_openai_gpt_4_1_image_visible = False
    oci_meta_llama_4_scout_image_visible = False
    openai_gpt_4o_image_visible = False
    azure_openai_gpt_4o_image_visible = False

    # 画像を使って回答がオンで、かつ対応するモデルが選択されている場合のみ表示
    if use_image:
        if "oci_openai/gpt-5" in llm_answer_checkbox:
            oci_openai_gpt_5_image_visible = True
        if "oci_openai/o3" in llm_answer_checkbox:
            oci_openai_o3_image_visible = True
        if "oci_openai/gpt-4.1" in llm_answer_checkbox:
            oci_openai_gpt_4_1_image_visible = True
        if "oci_meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox:
            oci_meta_llama_4_scout_image_visible = True
        if "openai/gpt-4o" in llm_answer_checkbox:
            openai_gpt_4o_image_visible = True
        if "azure_openai/gpt-4o" in llm_answer_checkbox:
            azure_openai_gpt_4o_image_visible = True

    return (
        gr.Accordion(visible=oci_openai_gpt_5_image_visible),
        gr.Accordion(visible=oci_openai_o3_image_visible),
        gr.Accordion(visible=oci_openai_gpt_4_1_image_visible),
        gr.Accordion(visible=oci_meta_llama_4_scout_image_visible),
        gr.Accordion(visible=openai_gpt_4o_image_visible),
        gr.Accordion(visible=azure_openai_gpt_4o_image_visible)
    )


def reset_all_llm_messages():
    """
    すべてのLLMメッセージをリセットする
    """
    return (
        gr.Markdown(value=""),  # tab_chat_document_oci_openai_gpt_5_answer_text
        gr.Markdown(value=""),  # tab_chat_document_oci_openai_o3_answer_text
        gr.Markdown(value=""),  # tab_chat_document_oci_openai_gpt_4_1_answer_text
        gr.Markdown(value=""),  # tab_chat_document_oci_xai_grok_4_answer_text
        gr.Markdown(value=""),  # tab_chat_document_oci_cohere_command_a_answer_text
        gr.Markdown(value=""),  # tab_chat_document_oci_meta_llama_4_scout_answer_text
        gr.Markdown(value=""),  # tab_chat_document_openai_gpt_4o_answer_text
        gr.Markdown(value="")  # tab_chat_document_azure_openai_gpt_4o_answer_text
    )


def reset_image_answers():
    """
    Vision 回答をリセットする
    """
    return (
        gr.Markdown(value=""),  # tab_chat_document_oci_openai_gpt_5_image_answer_text
        gr.Markdown(value=""),  # tab_chat_document_oci_openai_o3_image_answer_text
        gr.Markdown(value=""),  # tab_chat_document_oci_openai_gpt_4_1_image_answer_text
        gr.Markdown(value=""),  # tab_chat_document_oci_meta_llama_4_scout_image_answer_text
        gr.Markdown(value=""),  # tab_chat_document_openai_gpt_4o_image_answer_text
        gr.Markdown(value=""),  # tab_chat_document_azure_openai_gpt_4o_image_answer_text
    )


def reset_llm_evaluations():
    """
    LLM評価をリセットする
    """
    return (
        gr.Markdown(value=""),  # tab_chat_document_oci_openai_gpt_5_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_oci_openai_o3_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_oci_openai_gpt_4_1_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_oci_xai_grok_4_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_oci_cohere_command_a_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_oci_meta_llama_4_scout_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_openai_gpt_4o_evaluation_text
        gr.Markdown(value=""),  # tab_chat_document_azure_openai_gpt_4o_evaluation_text
    )


def create_oci_cred(user_ocid, tenancy_ocid, fingerprint, private_key_file, region):
    """
    OCI認証情報を設定するためのラッパー関数
    """
    return create_oci_cred_util(user_ocid, tenancy_ocid, fingerprint, private_key_file, region, pool)


def test_oci_cred(test_query_text):
    """
    OCI認証情報をテストするためのラッパー関数
    """
    return test_oci_cred_util(test_query_text, pool)


def create_table():
    """
    Wrapper function for creating database tables using the utility function
    """
    output_sql_text = create_table_util(pool, DEFAULT_COLLECTION_NAME)
    gr.Info("テーブルの作成が完了しました")
    return gr.Accordion(), gr.Textbox(value=output_sql_text.strip())


def load_document(file_path, server_directory, document_metadata):
    """
    ドキュメントファイルを読み込み、処理してデータベースに保存する

    この関数は utils.document_loader_util モジュールの関数を呼び出すラッパー関数です。
    """
    return load_document_util(
        file_path, server_directory, document_metadata,
        pool, DEFAULT_COLLECTION_NAME, generate_unique_id
    )


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
    """
    unstructured形式のドキュメントを分割し、チャンクをデータベースに保存する

    この関数は utils.document_split_util モジュールの関数を呼び出すラッパー関数です。
    """
    return split_document_util(
        doc_id, chunks_by, chunks_max_size,
        chunks_overlap_size, chunks_split_by, chunks_split_by_custom,
        chunks_language, chunks_normalize, chunks_normalize_options,
        pool, DEFAULT_COLLECTION_NAME, get_server_path, generate_embedding_response
    )


def on_select_split_document_chunks_result(evt: gr.SelectData, df: pd.DataFrame):
    """
    分割ドキュメントチャンク結果の選択イベントハンドラー

    この関数は utils.document_split_util モジュールの関数を呼び出すラッパー関数です。
    """
    print("on_select_split_document_chunks_result() start...")
    selected_index = evt.index[0]  # 選択された行のインデックスを取得
    selected_row = df.iloc[selected_index]  # 選択された行のすべてのデータを取得
    return selected_row['CHUNK_ID'], \
        selected_row['CHUNK_DATA']


def update_document_chunks_result_detail(doc_id, df: pd.DataFrame, chunk_id, chunk_data):
    """
    ドキュメントチャンク結果詳細を更新する

    この関数は utils.document_split_util モジュールの関数を呼び出すラッパー関数です。
    """
    return update_document_chunks_result_detail_with_validation(
        doc_id, df, chunk_id, chunk_data,
        pool, DEFAULT_COLLECTION_NAME, generate_embedding_response
    )


def embed_save_document_by_unstructured(doc_id, chunks_by, chunks_max_size,
                                        chunks_overlap_size,
                                        chunks_split_by, chunks_split_by_custom,
                                        chunks_language, chunks_normalize,
                                        chunks_normalize_options):
    """
    unstructured形式のドキュメントに対して埋め込みベクトルを生成し、データベースに保存する

    この関数は utils.document_embed_util モジュールの関数を呼び出すラッパー関数です。
    """
    return embed_save_document_util(
        doc_id, chunks_by, chunks_max_size,
        chunks_overlap_size, chunks_split_by, chunks_split_by_custom,
        chunks_language, chunks_normalize, chunks_normalize_options,
        pool, DEFAULT_COLLECTION_NAME, get_server_path
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
    # if region == "us-chicago-1":
    #     chat_llm = ChatOCIGenAI(
    #         model_id="xai.grok-4",
    #         provider="xai",
    #         service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
    #         compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
    #         model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 2048},
    #     )
    # else:
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
            try:
                generate_query1 = re.sub(r'^1\. ', '', sub_queries[0])
            except (IndexError, TypeError):
                generate_query1 = ""
            try:
                generate_query2 = re.sub(r'^2\. ', '', sub_queries[1])
            except (IndexError, TypeError):
                generate_query2 = ""
            try:
                generate_query3 = re.sub(r'^3\. ', '', sub_queries[2])
            except (IndexError, TypeError):
                generate_query3 = ""
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
            try:
                generate_query1 = re.sub(r'^1\. ', '', rag_fusion_queries[0])
            except (IndexError, TypeError):
                generate_query1 = ""
            try:
                generate_query2 = re.sub(r'^2\. ', '', rag_fusion_queries[1])
            except (IndexError, TypeError):
                generate_query2 = ""
            try:
                generate_query3 = re.sub(r'^3\. ', '', rag_fusion_queries[2])
            except (IndexError, TypeError):
                generate_query3 = ""
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
            try:
                generate_query1 = re.sub(r'^1\. ', '', hyde_answers[0])
            except (IndexError, TypeError):
                generate_query1 = ""
            try:
                generate_query2 = re.sub(r'^2\. ', '', hyde_answers[1])
            except (IndexError, TypeError):
                generate_query2 = ""
            try:
                generate_query3 = re.sub(r'^3\. ', '', hyde_answers[2])
            except (IndexError, TypeError):
                generate_query3 = ""
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
            try:
                generate_query1 = re.sub(r'^1\. ', '', step_back_queries[0])
            except (IndexError, TypeError):
                generate_query1 = ""
            try:
                generate_query2 = re.sub(r'^2\. ', '', step_back_queries[1])
            except (IndexError, TypeError):
                generate_query2 = ""
            try:
                generate_query3 = re.sub(r'^3\. ', '', step_back_queries[2])
            except (IndexError, TypeError):
                generate_query3 = ""
    elif generate_query_radio == "Customized-Multi-Step-Query":
        region = get_region()
        select_multi_step_query_sql = f"""
                SELECT json_value(dc.cmetadata, '$.file_name') name, de.embed_id embed_id, de.embed_data embed_data, de.doc_id doc_id
                FROM {DEFAULT_COLLECTION_NAME}_embedding de, {DEFAULT_COLLECTION_NAME}_collection dc
                WHERE de.doc_id = dc.id
                ORDER BY vector_distance(de.embed_vector , (
                        SELECT to_vector(et.embed_vector) embed_vector
                        FROM
                            dbms_vector_chain.utl_to_embeddings(:query_text, JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-v4.0"}}')) t,
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
    類似度検索を使用して質問に関連する分割を取得するためのラッパー関数
    """
    return search_document_util(
        pool,
        DEFAULT_COLLECTION_NAME,
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
    )


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
    """
    検索結果を使用してLLMとチャットするためのラッパー関数
    """
    async for result in chat_document_util(
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
        yield result


async def append_citation(
        search_result,
        llm_answer_checkbox,
        include_citation,
        use_image,
        query_text,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        oci_openai_gpt_5_answer_text,
        oci_openai_o3_answer_text,
        oci_openai_gpt_4_1_answer_text,
        oci_xai_grok_4_answer_text,
        oci_cohere_command_a_answer_text,
        oci_meta_llama_4_scout_answer_text,
        openai_gpt_4o_answer_text,
        azure_openai_gpt_4o_answer_text,
):
    """
    LLMの回答に引用情報を追加するためのラッパー関数
    """
    async for result in append_citation_util(
            search_result,
            llm_answer_checkbox,
            include_citation,
            use_image,
            query_text,
            doc_id_all_checkbox_input,
            doc_id_checkbox_group_input,
            oci_openai_gpt_5_answer_text,
            oci_openai_o3_answer_text,
            oci_openai_gpt_4_1_answer_text,
            oci_xai_grok_4_answer_text,
            oci_cohere_command_a_answer_text,
            oci_meta_llama_4_scout_answer_text,
            openai_gpt_4o_answer_text,
            azure_openai_gpt_4o_answer_text,
    ):
        yield result


async def process_single_image_streaming(image_url, query_text, llm_answer_checkbox_group, target_models, image_index,
                                         doc_id, img_id, custom_image_prompt=None):
    """
    単一画像を選択されたLLMモデルで処理し、ストリーミング形式で回答を返すためのラッパー関数
    """
    async for result in process_single_image_streaming_util(
            image_url, query_text, llm_answer_checkbox_group, target_models, image_index,
            doc_id, img_id, custom_image_prompt
    ):
        yield result


async def process_image_answers_streaming(
        search_result,
        use_image,
        single_image_processing,
        llm_answer_checkbox_group,
        query_text,
        oci_openai_gpt_5_image_answer_text,
        oci_openai_o3_image_answer_text,
        oci_openai_gpt_4_1_image_answer_text,
        oci_meta_llama_4_scout_image_answer_text,
        openai_gpt_4o_image_answer_text,
        azure_openai_gpt_4o_image_answer_text,
        image_limit_k=5,
        custom_image_prompt=None,
):
    """
    Vision 回答がオンの場合、検索結果から画像データを取得し、
    選択されたVisionモデルで画像処理を行い、ストリーミング形式で回答を出力するためのラッパー関数
    """
    async for result in process_image_answers_streaming_util(
            pool,
            DEFAULT_COLLECTION_NAME,
            search_result,
            use_image,
            single_image_processing,
            llm_answer_checkbox_group,
            query_text,
            oci_openai_gpt_5_image_answer_text,
            oci_openai_o3_image_answer_text,
            oci_openai_gpt_4_1_image_answer_text,
            oci_meta_llama_4_scout_image_answer_text,
            openai_gpt_4o_image_answer_text,
            azure_openai_gpt_4o_image_answer_text,
            image_limit_k,
            custom_image_prompt
    ):
        yield result


def set_query_id_state():
    print("in set_query_id_state() start...")
    return generate_unique_id("query_")


def eval_by_human(query_id, llm_name, human_evaluation_result, user_comment):
    """
    人間評価のラッパー関数
    """
    return eval_by_human_util(query_id, llm_name, human_evaluation_result, user_comment, pool)


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
                                f.vlm_answer,
                                f.ragas_evaluation_result,
                                f.human_evaluation_result,
                                f.user_comment,
                                TO_CHAR(r.created_date, 'YYYY-MM-DD HH24:MI:SS') AS created_date
                         FROM RAG_QA_RESULT r
                                  LEFT JOIN
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
                'VLM_ANSWER': 'Vision 回答',
                'RAGAS_EVALUATION_RESULT': 'LLM 評価結果',
                'HUMAN_EVALUATION_RESULT': 'Human 評価結果',
                'USER_COMMENT': 'Human コメント',
                'CREATED_DATE': '作成日時'
            }, inplace=True)

            print(f"{result_df=}")

            # 必要に応じてcreated_date列をdatetime型に変換
            result_df['作成日時'] = pd.to_datetime(result_df['作成日時'], format='%Y-%m-%d %H:%M:%S')

            # Vision回答からbase64画像情報を削除
            if 'Vision 回答' in result_df.columns:
                result_df['Vision 回答'] = result_df['Vision 回答'].apply(
                    lambda x: remove_base64_images_from_text(x) if pd.notna(x) else x
                )

            # ファイルパスを定義
            filepath = '/tmp/evaluation_result.xlsx'

            # ExcelWriterを使用して複数のDataFrameを異なるシートに書き込み
            with pd.ExcelWriter(filepath) as writer:
                result_df.to_excel(writer, sheet_name='Sheet1', index=False)

            print(f"Excelファイルが {filepath} に保存されました")
            gr.Info("評価レポートの生成が完了しました")
            return gr.DownloadButton(value=filepath, visible=True)


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
        oci_openai_gpt_5_response,
        oci_openai_o3_response,
        oci_openai_gpt_4_1_response,
        oci_xai_grok_4_response,
        oci_cohere_command_a_response,
        oci_meta_llama_4_scout_response,
        openai_gpt_4o_response,
        azure_openai_gpt_4o_response,
        oci_openai_gpt_5_evaluation,
        oci_openai_o3_evaluation,
        oci_openai_gpt_4_1_evaluation,
        oci_xai_grok_4_evaluation,
        oci_cohere_command_a_evaluation,
        oci_meta_llama_4_scout_evaluation,
        openai_gpt_4o_evaluation,
        azure_openai_gpt_4o_evaluation,
        oci_openai_gpt_5_image_response,
        oci_openai_o3_image_response,
        oci_openai_gpt_4_1_image_response,
        oci_meta_llama_4_scout_image_response,
        openai_gpt_4o_image_response,
        azure_openai_gpt_4o_image_response
):
    """
    クエリ結果をデータベースに挿入するためのラッパー関数
    """
    insert_query_result_util(
        pool,
        search_result,
        query_id,
        query,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        sql,
        llm_answer_checkbox_group,
        llm_evaluation_checkbox,
        standard_answer_text,
        oci_openai_gpt_5_response,
        oci_openai_o3_response,
        oci_openai_gpt_4_1_response,
        oci_xai_grok_4_response,
        oci_cohere_command_a_response,
        oci_meta_llama_4_scout_response,
        openai_gpt_4o_response,
        azure_openai_gpt_4o_response,
        oci_openai_gpt_5_evaluation,
        oci_openai_o3_evaluation,
        oci_openai_gpt_4_1_evaluation,
        oci_xai_grok_4_evaluation,
        oci_cohere_command_a_evaluation,
        oci_meta_llama_4_scout_evaluation,
        openai_gpt_4o_evaluation,
        azure_openai_gpt_4o_evaluation,
        oci_openai_gpt_5_image_response,
        oci_openai_o3_image_response,
        oci_openai_gpt_4_1_image_response,
        oci_meta_llama_4_scout_image_response,
        openai_gpt_4o_image_response,
        azure_openai_gpt_4o_image_response
    )


def delete_document(server_directory, doc_ids):
    """
    指定されたドキュメントを削除するためのラッパー関数
    """
    return delete_document_util(pool, DEFAULT_COLLECTION_NAME, server_directory, doc_ids)


theme = gr.themes.Default(
    spacing_size="sm",
    font=[GoogleFont(name="Noto Sans JP"), GoogleFont(name="Noto Sans SC"), GoogleFont(name="Roboto")]
).set()

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
            with gr.TabItem(label="Azure OpenAIの設定(オプション)", visible=False) as tab_create_azure_openai_cred:
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
                                # "oci_openai/gpt-5",
                                # "oci_openai/o3",
                                # "oci_openai/gpt-4.1",
                                "oci_xai/grok-4",
                                "oci_cohere/command-a",
                                "oci_meta/llama-4-scout-17b-16e-instruct",
                                "openai/gpt-4o",
                                # "azure_openai/gpt-4o",
                            ],
                            label="LLM モデル*",
                            value=[]
                        )
                with gr.Accordion(
                        label="OCI OpenAI GPT-5 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_oci_openai_gpt_5_accordion:
                    tab_chat_with_oci_openai_gpt_5_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="OCI OpenAI o3 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_oci_openai_o3_accordion:
                    tab_chat_with_oci_openai_o3_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="OCI OpenAI GPT-4.1 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_oci_openai_gpt_4_1_accordion:
                    tab_chat_with_oci_openai_gpt_4_1_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="OCI XAI Grok-4 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_oci_xai_grok_4_accordion:
                    tab_chat_with_oci_xai_grok_4_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="OCI Command-A メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_oci_cohere_command_a_accordion:
                    tab_chat_with_oci_cohere_command_a_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="OCI Llama 4 Scout 17b メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_oci_meta_llama_4_scout_accordion:
                    tab_chat_with_oci_meta_llama_4_scout_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="OpenAI gpt-4o メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_openai_gpt_4o_accordion:
                    tab_chat_with_openai_gpt_4o_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=200,
                        min_height=200,
                        max_height=300
                    )
                with gr.Accordion(
                        label="Azure OpenAI gpt-4o メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_with_llm_azure_openai_gpt_4o_accordion:
                    tab_chat_with_azure_openai_gpt_4o_answer_text = gr.Markdown(
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
                # with gr.Accordion(open=False,
                #                   label="画像ファイル(オプション) - OCI OpenAI GPT-5、OCI OpenAI o3、OCI OpenAI GPT-4.1、OCI Llama-4-Scoutモデルを利用する場合に限り、この画像入力が適用されます。"):
                with gr.Accordion(open=False,
                                  label="画像ファイル(オプション) - OCI Llama-4-Scoutモデルを利用する場合に限り、この画像入力が適用されます。"):
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
                with gr.TabItem(label="Pdf2Markdown") as tab_convert_pdf_to_markdown_document:
                    with gr.Row():
                        with gr.Column():
                            tab_convert_document_convert_pdf_to_markdown_file_text = gr.File(
                                label="変換前のファイル*",
                                file_types=[
                                    ".pdf"
                                ],
                                type="filepath",
                                interactive=True,
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_convert_document_convert_pdf_to_markdown_button = gr.Button(
                                value="PdfをMarkdownへ変換",
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
                            tab_convert_excel_to_text_button = gr.Button(
                                value="ExcelをTextへ変換",
                                variant="primary")
                with gr.TabItem(label="Xml2Text", visible=True) as tab_convert_xml_to_text_document:
                    with gr.Row():
                        with gr.Column():
                            tab_convert_document_convert_xml_to_text_file_text = gr.File(
                                label="変換前のファイル*（読み取りエラーが発生した場合は、UTF-8エンコーディングをご使用ください。）",
                                file_types=[
                                    ".xml",
                                ],
                                type="filepath",
                                interactive=True,
                            )
                    with gr.Accordion(label="タグ設定", open=True):
                        with gr.Row():
                            with gr.Column(scale=1):
                                # 前置きタグ
                                tab_convert_document_convert_xml_prefix_tag_text = gr.Textbox(
                                    label="前置きタグ",
                                    lines=1,
                                    interactive=True,
                                    placeholder="prefix1,prefix2,...",
                                    info="コンテンツの前に追加されるタグを指定します。"
                                )
                            with gr.Column(scale=1):
                                # 主タグ
                                tab_convert_document_convert_xml_main_tag_text = gr.Textbox(
                                    label="主タグ*",
                                    lines=1,
                                    interactive=True,
                                    placeholder="tag",
                                    info="メインコンテンツを囲むタグ名を指定します。必須項目です。"
                                )
                            with gr.Column(scale=1):
                                # マージするかどうかのチェックボックス
                                tab_convert_document_convert_xml_merge_checkbox = gr.Checkbox(
                                    label="マージする",
                                    value=True,
                                    interactive=True,
                                    info="複数のタグをマージして処理します。同じタグ名の要素を統合する際に使用します。"
                                )
                        with gr.Row():
                            with gr.Column(scale=1):
                                # グローバルタグ
                                tab_convert_document_convert_xml_global_tag_text = gr.Textbox(
                                    label="グローバルタグ",
                                    lines=1,
                                    interactive=True,
                                    placeholder="tag1,tag2,...",
                                    info="主タグと同じ親要素内から検索するタグ名をカンマ区切りで指定します。主タグの親要素内から指定されたタグを検索し、テキスト内容と属性を抽出します。"
                                )
                            with gr.Column(scale=1):
                                # 固定タグ
                                tab_convert_document_convert_xml_fixed_tag_text = gr.Textbox(
                                    label="固定タグ",
                                    lines=1,
                                    interactive=True,
                                    placeholder="key1=value1,key2=value2,...",
                                    info="変換時に固定的に使用されるタグ名を指定します。"
                                )
                            with gr.Column(scale=1):
                                # 置換タグ
                                tab_convert_document_convert_xml_replace_tag_text = gr.Textbox(
                                    label="置換タグ",
                                    lines=1,
                                    interactive=True,
                                    placeholder="old_tag1=new_tag1,old_tag2=new_tag2,...",
                                    info="既存のタグを別のタグに置換する際の設定を指定します。"
                                )
                        with gr.Row(visible=False):
                            with gr.Column(scale=1):
                                # 後付けタグ
                                tab_convert_document_convert_xml_suffix_tag_text = gr.Textbox(
                                    label="後付けタグ",
                                    lines=1,
                                    interactive=True,
                                    placeholder="suffix1,suffix2,...",
                                    info="コンテンツの後に追加されるタグを指定します。",
                                    visible=False,
                                )
                            with gr.Column(scale=1):
                                gr.Markdown("&nbsp;")
                            with gr.Column(scale=1):
                                gr.Markdown("&nbsp;")
                    with gr.Row():
                        with gr.Column():
                            tab_convert_xml_to_text_button = gr.Button(
                                value="XmlをTextへ変換",
                                variant="primary")
                with gr.TabItem(label="Json2Text", visible=True) as tab_convert_json_to_text_document:
                    with gr.Row():
                        with gr.Column():
                            tab_convert_document_convert_json_to_text_file_text = gr.File(
                                label="変換前のファイル*（読み取りエラーが発生した場合は、UTF-8エンコーディングをご使用ください。）",
                                file_types=[
                                    ".json",
                                ],
                                type="filepath",
                                interactive=True,
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_convert_json_to_text_button = gr.Button(
                                value="JsonをTextへ変換",
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
                            label="コンテンツ（先頭部分）",
                            lines=15,
                            max_lines=15,
                            autoscroll=False,
                            show_copy_button=True,
                            interactive=False
                        )
                with gr.Row():
                    with gr.Column():
                        tab_load_document_file_text = gr.File(
                            label="ファイル*（.txt または .md ファイルで読み取りエラーが発生した場合は、UTF-8エンコーディングをご使用ください。）",
                            file_types=[
                                ".txt", ".csv", ".doc", ".docx", ".epub", ".image",
                                ".md", ".msg", ".odt", ".org", ".pdf", ".ppt",
                                ".pptx",
                                ".rtf", ".rst", ".tsv", ".xls", ".xlsx",
                                ".xml"
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
                            tab_split_document_chunks_max_slider = gr.Slider(
                                label="Max",
                                value=320,
                                minimum=64,
                                maximum=3072,
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
                                # "oci_openai/gpt-5",
                                # "oci_openai/o3",
                                # "oci_openai/gpt-4.1",
                                "oci_xai/grok-4",
                                "oci_cohere/command-a",
                                "oci_meta/llama-4-scout-17b-16e-instruct",
                                "openai/gpt-4o",
                                # "azure_openai/gpt-4o",
                            ],
                            label="LLM モデル",
                            value=[]
                        )
                with gr.Row():
                    with gr.Column(scale=1):
                        tab_chat_document_question_embedding_model_checkbox_group = gr.CheckboxGroup(
                            ["cohere/embed-v4.0"],
                            label="Embedding モデル*",
                            value="cohere/embed-v4.0",
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("&nbsp;")
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
                            info="Default value: 0.70。類似度距離が閾値以下のデータのみを抽出する。",
                            maximum=0.95,
                            step=0.05,
                            value=0.70
                        )
                with gr.Row():
                    with gr.Column(scale=1):
                        tab_chat_document_reranker_model_radio = gr.Radio(
                            [
                                "None",
                                # "cohere/rerank-multilingual-v3.1",
                                # "cohere/rerank-english-v3.1",
                                "cohere/rerank-multilingual-v3.0",
                                "cohere/rerank-english-v3.0",
                            ],
                            label="Rerank モデル*", value="None")
                    with gr.Column(scale=1):
                        gr.Markdown("&nbsp;")
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
                                label="テキスト検索 Limit-K",
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
                        with gr.Column(scale=1):
                            tab_chat_document_use_image_checkbox = gr.Checkbox(
                                label="Vision 回答",
                                value=False,
                                info="検索された画像データを専用のVisionモデルで解析し、より正確な回答を提供。オンの場合、Highest-Ranked-One、Extend-First-K、Extend-Around-K、回答に引用を含める、Promptに現在の時間を含める、LLM 評価の設定は無視される。"
                            )
                        with gr.Column(scale=1):
                            tab_chat_document_image_limit_k_slider = gr.Slider(
                                label="画像 Limit-K",
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=3,
                                visible=False,
                                info="Vision 回答で使用する画像の最大数（1-10）"
                            )
                    with gr.Row():
                        with gr.Column():
                            tab_chat_document_single_image_processing_radio = gr.Radio(
                                label="画像処理方式",
                                choices=["1枚ずつ処理", "全画像まとめて処理", "ファイル単位で処理",
                                         "ファイル単位で処理+最初・最後", "ファイル単位で処理+最初・最後・前後画像"],
                                value="1枚ずつ処理",
                                visible=False,
                                info="Default value: 1枚ずつ処理。1枚ずつ処理: 各画像を個別に分析。全画像まとめて処理: 全ての画像を一度に送信。ファイル単位で処理: 同一ファイルの画像をまとめて処理。ファイル単位で処理+最初・最後: 各ファイルの最初と最後の画像を含めて処理。ファイル単位で処理+最初・最後・前後画像: 各ファイルの最初と最後の画像及び検索された画像の前後を含めて処理。（画像が10枚を超えるか、合計サイズが大きい場合、エラーが発生するため、「画像 Limit-K」を調整するか、「画像処理方式」を変更してください。）"
                            )
                    with gr.Accordion(label="Vision 回答 Prompt 設定", open=False,
                                      visible=False) as tab_chat_document_image_prompt_accordion:
                        with gr.Row():
                            with gr.Column():
                                tab_chat_document_image_prompt_text = gr.Textbox(
                                    label="VisionPromptテンプレート",
                                    lines=15,
                                    max_lines=25,
                                    interactive=True,
                                    show_copy_button=True,
                                    value=get_image_qa_prompt("{{query_text}}"),
                                    info="Vision処理で使用されるPromptテンプレートです。{{query_text}}は実行時に置換されます。"
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
                    with gr.Column(scale=1):
                        tab_chat_document_llm_evaluation_checkbox = gr.Checkbox(
                            label="LLM 評価",
                            show_label=True,
                            interactive=True,
                            value=False,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("&nbsp;")
                with gr.Row(visible=False) as tab_chat_document_system_message_row:
                    tab_chat_document_system_message_text = gr.Textbox(
                        label="システム・メッセージ*",
                        lines=15,
                        max_lines=20,
                        interactive=True,
                        value=get_llm_evaluation_system_message())
                with gr.Row(visible=False) as tab_chat_document_standard_answer_row:
                    tab_chat_document_standard_answer_text = gr.Textbox(
                        label="標準回答*",
                        lines=2,
                        interactive=True
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
                        label="OCI OpenAI GPT-5 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_oci_openai_gpt_5_accordion:
                    tab_chat_document_oci_openai_gpt_5_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Vision 回答",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_oci_openai_gpt_5_image_accordion:
                        tab_chat_document_oci_openai_gpt_5_image_answer_text = gr.Markdown(
                            show_copy_button=True,
                            height=600,
                            min_height=600,
                            max_height=600
                        )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_oci_openai_gpt_5_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_oci_openai_gpt_5_answer_human_eval_feedback_radio = gr.Radio(
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
                                tab_chat_document_oci_openai_gpt_5_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_oci_openai_gpt_5_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_oci_openai_gpt_5_evaluation_accordion:
                        tab_chat_document_oci_openai_gpt_5_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="OCI OpenAI o3 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_oci_openai_o3_accordion:
                    tab_chat_document_oci_openai_o3_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Vision 回答",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_oci_openai_o3_image_accordion:
                        tab_chat_document_oci_openai_o3_image_answer_text = gr.Markdown(
                            show_copy_button=True,
                            height=600,
                            min_height=600,
                            max_height=600
                        )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_oci_openai_o3_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_oci_openai_o3_answer_human_eval_feedback_radio = gr.Radio(
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
                                tab_chat_document_oci_openai_o3_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_oci_openai_o3_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_oci_openai_o3_evaluation_accordion:
                        tab_chat_document_oci_openai_o3_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="OCI OpenAI GPT-4.1 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_oci_openai_gpt_4_1_accordion:
                    tab_chat_document_oci_openai_gpt_4_1_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Vision 回答",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_oci_openai_gpt_4_1_image_accordion:
                        tab_chat_document_oci_openai_gpt_4_1_image_answer_text = gr.Markdown(
                            show_copy_button=True,
                            height=600,
                            min_height=600,
                            max_height=600
                        )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_oci_openai_gpt_4_1_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_oci_openai_gpt_4_1_answer_human_eval_feedback_radio = gr.Radio(
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
                                tab_chat_document_oci_openai_gpt_4_1_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_oci_openai_gpt_4_1_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_oci_openai_gpt_4_1_evaluation_accordion:
                        tab_chat_document_oci_openai_gpt_4_1_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="OCI XAI Grok-4 メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_oci_xai_grok_4_accordion:
                    tab_chat_document_oci_xai_grok_4_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_oci_xai_grok_4_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_oci_xai_grok_4_answer_human_eval_feedback_radio = gr.Radio(
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
                                tab_chat_document_oci_xai_grok_4_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_oci_xai_grok_4_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_oci_xai_grok_4_evaluation_accordion:
                        tab_chat_document_oci_xai_grok_4_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="OCI Command-A メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_oci_cohere_command_a_accordion:
                    tab_chat_document_oci_cohere_command_a_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_oci_cohere_command_a_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_oci_cohere_command_a_answer_human_eval_feedback_radio = gr.Radio(
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
                                tab_chat_document_oci_cohere_command_a_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_oci_cohere_command_a_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_oci_cohere_command_a_evaluation_accordion:
                        tab_chat_document_oci_cohere_command_a_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="OCI Llama 4 Scout 17b メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_oci_meta_llama_4_scout_accordion:
                    tab_chat_document_oci_meta_llama_4_scout_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Vision 回答",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_oci_meta_llama_4_scout_image_accordion:
                        tab_chat_document_oci_meta_llama_4_scout_image_answer_text = gr.Markdown(
                            show_copy_button=True,
                            height=600,
                            min_height=600,
                            max_height=600
                        )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_oci_meta_llama_4_scout_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_oci_meta_llama_4_scout_answer_human_eval_feedback_radio = gr.Radio(
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
                                tab_chat_document_oci_meta_llama_4_scout_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_oci_meta_llama_4_scout_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_oci_meta_llama_4_scout_evaluation_accordion:
                        tab_chat_document_oci_meta_llama_4_scout_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(label="OpenAI gpt-4o メッセージ",
                                  visible=False,
                                  open=True) as tab_chat_document_llm_openai_gpt_4o_accordion:
                    tab_chat_document_openai_gpt_4o_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Vision 回答",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_openai_gpt_4o_image_accordion:
                        tab_chat_document_openai_gpt_4o_image_answer_text = gr.Markdown(
                            show_copy_button=True,
                            height=600,
                            min_height=600,
                            max_height=600
                        )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_openai_gpt_4o_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_openai_gpt_4o_answer_human_eval_feedback_radio = gr.Radio(
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
                                tab_chat_document_openai_gpt_4o_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_openai_gpt_4o_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_openai_gpt_4o_evaluation_accordion:
                        tab_chat_document_openai_gpt_4o_evaluation_text = gr.Markdown(
                            show_copy_button=True,
                            height=200,
                            min_height=200,
                            max_height=300
                        )
                with gr.Accordion(
                        label="Azure OpenAI gpt-4o メッセージ",
                        visible=False,
                        open=True
                ) as tab_chat_document_llm_azure_openai_gpt_4o_accordion:
                    tab_chat_document_azure_openai_gpt_4o_answer_text = gr.Markdown(
                        show_copy_button=True,
                        height=300,
                        min_height=300,
                        max_height=300
                    )
                    with gr.Accordion(
                            label="Vision 回答",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_azure_openai_gpt_4o_image_accordion:
                        tab_chat_document_azure_openai_gpt_4o_image_answer_text = gr.Markdown(
                            show_copy_button=True,
                            height=600,
                            min_height=600,
                            max_height=600
                        )
                    with gr.Accordion(
                            label="Human 評価",
                            visible=True,
                            open=True
                    ) as tab_chat_document_llm_azure_openai_gpt_4o_human_evaluation_accordion:
                        with gr.Row():
                            tab_chat_document_azure_openai_gpt_4o_answer_human_eval_feedback_radio = gr.Radio(
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
                                tab_chat_document_azure_openai_gpt_4o_answer_human_eval_feedback_text = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    lines=2,
                                    interactive=True,
                                    autoscroll=True,
                                    placeholder="具体的な意見や感想を自由に書いてください。",
                                )
                            with gr.Column(scale=1):
                                tab_chat_document_azure_openai_gpt_4o_answer_human_eval_feedback_send_button = gr.Button(
                                    value="送信",
                                    variant="primary",
                                )
                    with gr.Accordion(
                            label="LLM 評価結果",
                            visible=False,
                            open=True
                    ) as tab_chat_document_llm_azure_openai_gpt_4o_evaluation_accordion:
                        tab_chat_document_azure_openai_gpt_4o_evaluation_text = gr.Markdown(
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
        ],
        outputs=[
            tab_create_azure_openai_cred_api_key_text,
            tab_create_azure_openai_cred_endpoint_gpt_4o_text,
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
            tab_chat_with_llm_oci_openai_gpt_5_accordion,
            tab_chat_with_llm_oci_openai_o3_accordion,
            tab_chat_with_llm_oci_openai_gpt_4_1_accordion,
            tab_chat_with_llm_oci_xai_grok_4_accordion,
            tab_chat_with_llm_oci_cohere_command_a_accordion,
            tab_chat_with_llm_oci_meta_llama_4_scout_accordion,
            tab_chat_with_llm_openai_gpt_4o_accordion,
            tab_chat_with_llm_azure_openai_gpt_4o_accordion,
        ]
    )

    tab_chat_with_llm_clear_button.add(
        [
            tab_chat_with_llm_query_image,
            tab_chat_with_llm_query_text,
            tab_chat_with_llm_answer_checkbox_group,
            tab_chat_with_oci_openai_gpt_5_answer_text,
            tab_chat_with_oci_openai_o3_answer_text,
            tab_chat_with_oci_openai_gpt_4_1_answer_text,
            tab_chat_with_oci_xai_grok_4_answer_text,
            tab_chat_with_oci_cohere_command_a_answer_text,
            tab_chat_with_oci_meta_llama_4_scout_answer_text,
            tab_chat_with_openai_gpt_4o_answer_text,
            tab_chat_with_azure_openai_gpt_4o_answer_text
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
            tab_chat_with_oci_openai_gpt_5_answer_text,
            tab_chat_with_oci_openai_o3_answer_text,
            tab_chat_with_oci_openai_gpt_4_1_answer_text,
            tab_chat_with_oci_xai_grok_4_answer_text,
            tab_chat_with_oci_cohere_command_a_answer_text,
            tab_chat_with_oci_meta_llama_4_scout_answer_text,
            tab_chat_with_openai_gpt_4o_answer_text,
            tab_chat_with_azure_openai_gpt_4o_answer_text
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

    tab_convert_document_convert_pdf_to_markdown_button.click(
        fn=convert_pdf_to_markdown,
        inputs=[tab_convert_document_convert_pdf_to_markdown_file_text],
        outputs=[
            tab_convert_document_convert_pdf_to_markdown_file_text,
            tab_load_document_file_text
        ]
    )

    tab_convert_excel_to_text_button.click(
        convert_excel_to_text_document,
        inputs=[
            tab_convert_document_convert_excel_to_text_file_text,
        ],
        outputs=[
            tab_convert_document_convert_excel_to_text_file_text,
            tab_load_document_file_text,
        ],
    )

    tab_convert_xml_to_text_button.click(
        convert_xml_to_text_document,
        inputs=[
            tab_convert_document_convert_xml_to_text_file_text,
            tab_convert_document_convert_xml_global_tag_text,
            tab_convert_document_convert_xml_fixed_tag_text,
            tab_convert_document_convert_xml_replace_tag_text,
            tab_convert_document_convert_xml_prefix_tag_text,
            tab_convert_document_convert_xml_main_tag_text,
            tab_convert_document_convert_xml_suffix_tag_text,
            tab_convert_document_convert_xml_merge_checkbox,
        ],
        outputs=[
            tab_convert_document_convert_xml_to_text_file_text,
            tab_convert_document_convert_xml_global_tag_text,
            tab_convert_document_convert_xml_fixed_tag_text,
            tab_convert_document_convert_xml_replace_tag_text,
            tab_convert_document_convert_xml_prefix_tag_text,
            tab_convert_document_convert_xml_main_tag_text,
            tab_convert_document_convert_xml_suffix_tag_text,
            tab_convert_document_convert_xml_merge_checkbox,
            tab_load_document_file_text,
        ],
    )

    tab_convert_json_to_text_button.click(
        convert_json_to_text_document,
        inputs=[
            tab_convert_document_convert_json_to_text_file_text,
        ],
        outputs=[
            tab_convert_document_convert_json_to_text_file_text,
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
            tab_delete_document_doc_ids_checkbox_group,
            tab_chat_document_doc_id_checkbox_group,
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
            tab_chat_document_llm_oci_openai_gpt_5_accordion,
            tab_chat_document_llm_oci_openai_o3_accordion,
            tab_chat_document_llm_oci_openai_gpt_4_1_accordion,
            tab_chat_document_llm_oci_xai_grok_4_accordion,
            tab_chat_document_llm_oci_cohere_command_a_accordion,
            tab_chat_document_llm_oci_meta_llama_4_scout_accordion,
            tab_chat_document_llm_openai_gpt_4o_accordion,
            tab_chat_document_llm_azure_openai_gpt_4o_accordion
        ]
    ).then(
        set_image_answer_visibility,
        inputs=[
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_use_image_checkbox
        ],
        outputs=[
            tab_chat_document_llm_oci_openai_gpt_5_image_accordion,
            tab_chat_document_llm_oci_openai_o3_image_accordion,
            tab_chat_document_llm_oci_openai_gpt_4_1_image_accordion,
            tab_chat_document_llm_oci_meta_llama_4_scout_image_accordion,
            tab_chat_document_llm_openai_gpt_4o_image_accordion,
            tab_chat_document_llm_azure_openai_gpt_4o_image_accordion
        ]
    )

    tab_chat_document_llm_evaluation_checkbox.change(
        lambda x: (
            gr.Row(visible=True),
            gr.Row(visible=True)
        ) if x else (
            gr.Row(visible=False),
            gr.Row(visible=False)
        ),
        tab_chat_document_llm_evaluation_checkbox,
        [
            tab_chat_document_system_message_row,
            tab_chat_document_standard_answer_row
        ]
    ).then(
        set_chat_llm_evaluation,
        inputs=[
            tab_chat_document_llm_evaluation_checkbox
        ],
        outputs=[
            tab_chat_document_llm_oci_openai_gpt_5_evaluation_accordion,
            tab_chat_document_llm_oci_openai_o3_evaluation_accordion,
            tab_chat_document_llm_oci_openai_gpt_4_1_evaluation_accordion,
            tab_chat_document_llm_oci_xai_grok_4_evaluation_accordion,
            tab_chat_document_llm_oci_cohere_command_a_evaluation_accordion,
            tab_chat_document_llm_oci_meta_llama_4_scout_evaluation_accordion,
            tab_chat_document_llm_openai_gpt_4o_evaluation_accordion,
            tab_chat_document_llm_azure_openai_gpt_4o_evaluation_accordion,
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
        lambda x: (gr.Accordion(visible=x), gr.Slider(visible=x), gr.Radio(visible=x)) if x else (
            gr.Accordion(visible=False), gr.Slider(visible=False), gr.Radio(visible=False)),
        inputs=[tab_chat_document_use_image_checkbox],
        outputs=[
            tab_chat_document_image_prompt_accordion,
            tab_chat_document_image_limit_k_slider,
            tab_chat_document_single_image_processing_radio,
        ]
    ).then(
        set_image_answer_visibility,
        inputs=[
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_use_image_checkbox
        ],
        outputs=[
            tab_chat_document_llm_oci_openai_gpt_5_image_accordion,
            tab_chat_document_llm_oci_openai_o3_image_accordion,
            tab_chat_document_llm_oci_openai_gpt_4_1_image_accordion,
            tab_chat_document_llm_oci_meta_llama_4_scout_image_accordion,
            tab_chat_document_llm_openai_gpt_4o_image_accordion,
            tab_chat_document_llm_azure_openai_gpt_4o_image_accordion
        ]
    )

    tab_chat_document_chat_document_button.click(
        lambda: gr.DownloadButton(visible=False),
        outputs=[tab_chat_document_download_output_button]
    ).then(
        reset_all_llm_messages,
        inputs=[],
        outputs=[
            tab_chat_document_oci_openai_gpt_5_answer_text,
            tab_chat_document_oci_openai_o3_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_answer_text,
            tab_chat_document_oci_xai_grok_4_answer_text,
            tab_chat_document_oci_cohere_command_a_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_answer_text,
            tab_chat_document_openai_gpt_4o_answer_text,
            tab_chat_document_azure_openai_gpt_4o_answer_text
        ]
    ).then(
        reset_image_answers,
        inputs=[],
        outputs=[
            tab_chat_document_oci_openai_gpt_5_image_answer_text,
            tab_chat_document_oci_openai_o3_image_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_image_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_image_answer_text,
            tab_chat_document_openai_gpt_4o_image_answer_text,
            tab_chat_document_azure_openai_gpt_4o_image_answer_text
        ]
    ).then(
        reset_llm_evaluations,
        inputs=[],
        outputs=[
            tab_chat_document_oci_openai_gpt_5_evaluation_text,
            tab_chat_document_oci_openai_o3_evaluation_text,
            tab_chat_document_oci_openai_gpt_4_1_evaluation_text,
            tab_chat_document_oci_xai_grok_4_evaluation_text,
            tab_chat_document_oci_cohere_command_a_evaluation_text,
            tab_chat_document_oci_meta_llama_4_scout_evaluation_text,
            tab_chat_document_openai_gpt_4o_evaluation_text,
            tab_chat_document_azure_openai_gpt_4o_evaluation_text
        ]
    ).then(
        reset_eval_by_human_result,
        inputs=[],
        outputs=[
            tab_chat_document_oci_openai_gpt_5_answer_human_eval_feedback_radio,
            tab_chat_document_oci_openai_gpt_5_answer_human_eval_feedback_text,
            tab_chat_document_oci_openai_o3_answer_human_eval_feedback_radio,
            tab_chat_document_oci_openai_o3_answer_human_eval_feedback_text,
            tab_chat_document_oci_openai_gpt_4_1_answer_human_eval_feedback_radio,
            tab_chat_document_oci_openai_gpt_4_1_answer_human_eval_feedback_text,
            tab_chat_document_oci_xai_grok_4_answer_human_eval_feedback_radio,
            tab_chat_document_oci_xai_grok_4_answer_human_eval_feedback_text,
            tab_chat_document_oci_cohere_command_a_answer_human_eval_feedback_radio,
            tab_chat_document_oci_cohere_command_a_answer_human_eval_feedback_text,
            tab_chat_document_oci_meta_llama_4_scout_answer_human_eval_feedback_radio,
            tab_chat_document_oci_meta_llama_4_scout_answer_human_eval_feedback_text,
            tab_chat_document_openai_gpt_4o_answer_human_eval_feedback_radio,
            tab_chat_document_openai_gpt_4o_answer_human_eval_feedback_text,
            tab_chat_document_azure_openai_gpt_4o_answer_human_eval_feedback_radio,
            tab_chat_document_azure_openai_gpt_4o_answer_human_eval_feedback_text,
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
            tab_chat_document_oci_openai_gpt_5_answer_text,
            tab_chat_document_oci_openai_o3_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_answer_text,
            tab_chat_document_oci_xai_grok_4_answer_text,
            tab_chat_document_oci_cohere_command_a_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_answer_text,
            tab_chat_document_openai_gpt_4o_answer_text,
            tab_chat_document_azure_openai_gpt_4o_answer_text,
        ]
    ).then(
        append_citation,
        inputs=[
            tab_chat_document_searched_result_dataframe,
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_include_citation_checkbox,
            tab_chat_document_use_image_checkbox,
            tab_chat_document_query_text,
            tab_chat_document_doc_id_all_checkbox,
            tab_chat_document_doc_id_checkbox_group,
            tab_chat_document_oci_openai_gpt_5_answer_text,
            tab_chat_document_oci_openai_o3_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_answer_text,
            tab_chat_document_oci_xai_grok_4_answer_text,
            tab_chat_document_oci_cohere_command_a_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_answer_text,
            tab_chat_document_openai_gpt_4o_answer_text,
            tab_chat_document_azure_openai_gpt_4o_answer_text,
        ],
        outputs=[
            tab_chat_document_oci_openai_gpt_5_answer_text,
            tab_chat_document_oci_openai_o3_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_answer_text,
            tab_chat_document_oci_xai_grok_4_answer_text,
            tab_chat_document_oci_cohere_command_a_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_answer_text,
            tab_chat_document_openai_gpt_4o_answer_text,
            tab_chat_document_azure_openai_gpt_4o_answer_text,
        ]
    ).then(
        process_image_answers_streaming,
        inputs=[
            tab_chat_document_searched_result_dataframe,
            tab_chat_document_use_image_checkbox,
            tab_chat_document_single_image_processing_radio,
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_query_text,
            tab_chat_document_oci_openai_gpt_5_image_answer_text,
            tab_chat_document_oci_openai_o3_image_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_image_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_image_answer_text,
            tab_chat_document_openai_gpt_4o_image_answer_text,
            tab_chat_document_azure_openai_gpt_4o_image_answer_text,
            tab_chat_document_image_limit_k_slider,
            tab_chat_document_image_prompt_text,
        ],
        outputs=[
            tab_chat_document_oci_openai_gpt_5_image_answer_text,
            tab_chat_document_oci_openai_o3_image_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_image_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_image_answer_text,
            tab_chat_document_openai_gpt_4o_image_answer_text,
            tab_chat_document_azure_openai_gpt_4o_image_answer_text
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
            tab_chat_document_use_image_checkbox,
            tab_chat_document_system_message_text,
            tab_chat_document_standard_answer_text,
            tab_chat_document_oci_openai_gpt_5_answer_text,
            tab_chat_document_oci_openai_o3_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_answer_text,
            tab_chat_document_oci_xai_grok_4_answer_text,
            tab_chat_document_oci_cohere_command_a_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_answer_text,
            tab_chat_document_openai_gpt_4o_answer_text,
            tab_chat_document_azure_openai_gpt_4o_answer_text,
        ],
        outputs=[
            tab_chat_document_oci_openai_gpt_5_evaluation_text,
            tab_chat_document_oci_openai_o3_evaluation_text,
            tab_chat_document_oci_openai_gpt_4_1_evaluation_text,
            tab_chat_document_oci_xai_grok_4_evaluation_text,
            tab_chat_document_oci_cohere_command_a_evaluation_text,
            tab_chat_document_oci_meta_llama_4_scout_evaluation_text,
            tab_chat_document_openai_gpt_4o_evaluation_text,
            tab_chat_document_azure_openai_gpt_4o_evaluation_text,
        ]
    ).then(
        generate_download_file,
        inputs=[
            tab_chat_document_searched_result_dataframe,
            tab_chat_document_llm_answer_checkbox_group,
            tab_chat_document_include_citation_checkbox,
            tab_chat_document_use_image_checkbox,
            tab_chat_document_llm_evaluation_checkbox,
            tab_chat_document_query_text,
            tab_chat_document_doc_id_all_checkbox,
            tab_chat_document_doc_id_checkbox_group,
            tab_chat_document_standard_answer_text,
            tab_chat_document_oci_openai_gpt_5_answer_text,
            tab_chat_document_oci_openai_o3_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_answer_text,
            tab_chat_document_oci_xai_grok_4_answer_text,
            tab_chat_document_oci_cohere_command_a_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_answer_text,
            tab_chat_document_openai_gpt_4o_answer_text,
            tab_chat_document_azure_openai_gpt_4o_answer_text,
            tab_chat_document_oci_openai_gpt_5_evaluation_text,
            tab_chat_document_oci_openai_o3_evaluation_text,
            tab_chat_document_oci_openai_gpt_4_1_evaluation_text,
            tab_chat_document_oci_xai_grok_4_evaluation_text,
            tab_chat_document_oci_cohere_command_a_evaluation_text,
            tab_chat_document_oci_meta_llama_4_scout_evaluation_text,
            tab_chat_document_openai_gpt_4o_evaluation_text,
            tab_chat_document_azure_openai_gpt_4o_evaluation_text,
            tab_chat_document_oci_openai_gpt_5_image_answer_text,
            tab_chat_document_oci_openai_o3_image_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_image_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_image_answer_text,
            tab_chat_document_openai_gpt_4o_image_answer_text,
            tab_chat_document_azure_openai_gpt_4o_image_answer_text,
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
            tab_chat_document_oci_openai_gpt_5_answer_text,
            tab_chat_document_oci_openai_o3_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_answer_text,
            tab_chat_document_oci_xai_grok_4_answer_text,
            tab_chat_document_oci_cohere_command_a_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_answer_text,
            tab_chat_document_openai_gpt_4o_answer_text,
            tab_chat_document_azure_openai_gpt_4o_answer_text,
            tab_chat_document_oci_openai_gpt_5_evaluation_text,
            tab_chat_document_oci_openai_o3_evaluation_text,
            tab_chat_document_oci_openai_gpt_4_1_evaluation_text,
            tab_chat_document_oci_xai_grok_4_evaluation_text,
            tab_chat_document_oci_cohere_command_a_evaluation_text,
            tab_chat_document_oci_meta_llama_4_scout_evaluation_text,
            tab_chat_document_openai_gpt_4o_evaluation_text,
            tab_chat_document_azure_openai_gpt_4o_evaluation_text,
            tab_chat_document_oci_openai_gpt_5_image_answer_text,
            tab_chat_document_oci_openai_o3_image_answer_text,
            tab_chat_document_oci_openai_gpt_4_1_image_answer_text,
            tab_chat_document_oci_meta_llama_4_scout_image_answer_text,
            tab_chat_document_openai_gpt_4o_image_answer_text,
            tab_chat_document_azure_openai_gpt_4o_image_answer_text,
        ],
        outputs=[]
    )

    tab_chat_document_oci_openai_gpt_5_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="oci_openai/gpt-5"),
            tab_chat_document_oci_openai_gpt_5_answer_human_eval_feedback_radio,
            tab_chat_document_oci_openai_gpt_5_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_oci_openai_gpt_5_answer_human_eval_feedback_radio,
            tab_chat_document_oci_openai_gpt_5_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_oci_openai_o3_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="oci_openai/o3"),
            tab_chat_document_oci_openai_o3_answer_human_eval_feedback_radio,
            tab_chat_document_oci_openai_o3_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_oci_openai_o3_answer_human_eval_feedback_radio,
            tab_chat_document_oci_openai_o3_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_oci_openai_gpt_4_1_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="oci_openai/gpt-4.1"),
            tab_chat_document_oci_openai_gpt_4_1_answer_human_eval_feedback_radio,
            tab_chat_document_oci_openai_gpt_4_1_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_oci_openai_gpt_4_1_answer_human_eval_feedback_radio,
            tab_chat_document_oci_openai_gpt_4_1_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_oci_xai_grok_4_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="oci_xai/grok-4"),
            tab_chat_document_oci_xai_grok_4_answer_human_eval_feedback_radio,
            tab_chat_document_oci_xai_grok_4_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_oci_xai_grok_4_answer_human_eval_feedback_radio,
            tab_chat_document_oci_xai_grok_4_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_oci_cohere_command_a_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="oci_cohere/command-a"),
            tab_chat_document_oci_cohere_command_a_answer_human_eval_feedback_radio,
            tab_chat_document_oci_cohere_command_a_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_oci_cohere_command_a_answer_human_eval_feedback_radio,
            tab_chat_document_oci_cohere_command_a_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_oci_meta_llama_4_scout_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="oci_meta/llama-4-scout-17b-16e-instruct"),
            tab_chat_document_oci_meta_llama_4_scout_answer_human_eval_feedback_radio,
            tab_chat_document_oci_meta_llama_4_scout_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_oci_meta_llama_4_scout_answer_human_eval_feedback_radio,
            tab_chat_document_oci_meta_llama_4_scout_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_openai_gpt_4o_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="openai/gpt-4o"),
            tab_chat_document_openai_gpt_4o_answer_human_eval_feedback_radio,
            tab_chat_document_openai_gpt_4o_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_openai_gpt_4o_answer_human_eval_feedback_radio,
            tab_chat_document_openai_gpt_4o_answer_human_eval_feedback_text,
        ]
    )

    tab_chat_document_azure_openai_gpt_4o_answer_human_eval_feedback_send_button.click(
        eval_by_human,
        inputs=[
            query_id_state,
            gr.State(value="azure_openai/gpt-4o"),
            tab_chat_document_azure_openai_gpt_4o_answer_human_eval_feedback_radio,
            tab_chat_document_azure_openai_gpt_4o_answer_human_eval_feedback_text,
        ],
        outputs=[
            tab_chat_document_azure_openai_gpt_4o_answer_human_eval_feedback_radio,
            tab_chat_document_azure_openai_gpt_4o_answer_human_eval_feedback_text,
        ]
    )

    tab_download_eval_result_generate_button.click(
        generate_eval_result_file,
        inputs=[],
        outputs=[
            tab_download_eval_result_download_button,
        ]
    )


    # RAGPrompt設定のイベントハンドラー
    def save_rag_prompt(prompt_text):
        """RAGPromptを保存する"""
        try:
            update_langgpt_rag_prompt(prompt_text)
            return gr.Info("Promptが保存されました。")
        except Exception as e:
            return gr.Warning(f"Promptの保存に失敗しました: {str(e)}")


    def reset_rag_prompt():
        """RAGPromptをデフォルトに戻す"""
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


    # VisionPrompt設定のイベントハンドラー
    def save_image_prompt(prompt_text):
        """VisionPromptを保存する"""
        try:
            update_image_qa_prompt(prompt_text)
            return gr.Info("VisionPromptが保存されました。")
        except Exception as e:
            return gr.Warning(f"VisionPromptの保存に失敗しました: {str(e)}")


    def reset_image_prompt():
        """VisionPromptをデフォルトに戻す"""
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
        auth=do_auth,
    )
