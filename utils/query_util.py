"""
クエリ操作関連の関数を提供するモジュール

このモジュールには以下の機能が含まれています：
- クエリ結果挿入 (insert_query_result)
"""

import oracledb

from utils.openai_compatible_util import OPENAI_COMPATIBLE_MODEL_KEY, get_openai_compatible_llm_name
from utils.text_util import remove_base64_images_from_text


def insert_query_result(
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
        oci_xai_grok_4_response,
        oci_cohere_command_a_response,
        oci_meta_llama_4_scout_response,
        openai_compatible_response,
        oci_xai_grok_4_evaluation,
        oci_cohere_command_a_evaluation,
        oci_meta_llama_4_scout_evaluation,
        openai_compatible_evaluation,
        oci_xai_grok_4_image_response,
        oci_meta_llama_4_scout_image_response,
        openai_compatible_image_response,
):
    """
    クエリ結果をデータベースに挿入する
    """
    print("in insert_query_result() start...")
    if not query:
        return
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        return
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        return

    def selected(model_key):
        return model_key in llm_answer_checkbox_group

    def evaluation_value(value):
        return value if llm_evaluation_checkbox else ""

    def insert_feedback(cursor, llm_name, llm_answer, vlm_answer, ragas_evaluation_result):
        insert_sql = """
                     INSERT INTO RAG_QA_FEEDBACK (query_id,
                                                  llm_name,
                                                  llm_answer,
                                                  vlm_answer,
                                                  ragas_evaluation_result)
                     VALUES (:1,
                             :2,
                             :3,
                             :4,
                             :5) \
                     """
        cursor.setinputsizes(None, None, oracledb.CLOB, oracledb.CLOB, oracledb.CLOB)
        cursor.execute(
            insert_sql,
            [
                query_id,
                llm_name,
                llm_answer,
                vlm_answer,
                ragas_evaluation_result
            ]
        )

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
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

            if selected("oci_xai/grok-4.3"):
                insert_feedback(
                    cursor,
                    "oci_xai/grok-4.3",
                    oci_xai_grok_4_response,
                    remove_base64_images_from_text(oci_xai_grok_4_image_response),
                    evaluation_value(oci_xai_grok_4_evaluation)
                )

            if selected("oci_cohere/command-a"):
                insert_feedback(
                    cursor,
                    "oci_cohere/command-a",
                    oci_cohere_command_a_response,
                    "",
                    evaluation_value(oci_cohere_command_a_evaluation)
                )

            if selected("oci_meta/llama-4-scout-17b-16e-instruct"):
                insert_feedback(
                    cursor,
                    "oci_meta/llama-4-scout-17b-16e-instruct",
                    oci_meta_llama_4_scout_response,
                    remove_base64_images_from_text(oci_meta_llama_4_scout_image_response),
                    evaluation_value(oci_meta_llama_4_scout_evaluation)
                )

            if selected(OPENAI_COMPATIBLE_MODEL_KEY):
                insert_feedback(
                    cursor,
                    get_openai_compatible_llm_name(),
                    openai_compatible_response,
                    remove_base64_images_from_text(openai_compatible_image_response),
                    evaluation_value(openai_compatible_evaluation)
                )

        conn.commit()
