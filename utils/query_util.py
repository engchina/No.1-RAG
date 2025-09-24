"""
クエリ操作関連の関数を提供するモジュール

このモジュールには以下の機能が含まれています：
- クエリ結果挿入 (insert_query_result)
"""

import oracledb

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
    クエリ結果をデータベースに挿入する
    
    Args:
        pool: データベース接続プール
        search_result: 検索結果のDataFrame
        query_id: クエリID
        query: クエリテキスト
        doc_id_all_checkbox_input: 全ドキュメント選択フラグ
        doc_id_checkbox_group_input: 選択されたドキュメントIDのリスト
        sql: 実行されたSQL
        llm_answer_checkbox_group: 選択されたLLMモデルのリスト
        llm_evaluation_checkbox: LLM評価フラグ
        standard_answer_text: 標準回答テキスト
        各LLMの回答と評価結果
        各LLMの画像回答
    """
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

            if "oci_openai/gpt-5" in llm_answer_checkbox_group:
                oci_openai_gpt_5_response = oci_openai_gpt_5_response
                if llm_evaluation_checkbox:
                    oci_openai_gpt_5_evaluation = oci_openai_gpt_5_evaluation
                else:
                    oci_openai_gpt_5_evaluation = ""
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
                            "oci_openai/gpt-5",
                            oci_openai_gpt_5_response,
                            remove_base64_images_from_text(oci_openai_gpt_5_image_response),
                            oci_openai_gpt_5_evaluation
                        ]
                    )

            if "oci_openai/o3" in llm_answer_checkbox_group:
                oci_openai_o3_response = oci_openai_o3_response
                if llm_evaluation_checkbox:
                    oci_openai_o3_evaluation = oci_openai_o3_evaluation
                else:
                    oci_openai_o3_evaluation = ""

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
                        "oci_openai/o3",
                        oci_openai_o3_response,
                        remove_base64_images_from_text(oci_openai_o3_image_response),
                        oci_openai_o3_evaluation
                    ]
                )

            if "oci_openai/gpt-4.1" in llm_answer_checkbox_group:
                oci_openai_gpt_4_1_response = oci_openai_gpt_4_1_response
                if llm_evaluation_checkbox:
                    oci_openai_gpt_4_1_evaluation = oci_openai_gpt_4_1_evaluation
                else:
                    oci_openai_gpt_4_1_evaluation = ""
                
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
                        "oci_openai/gpt-4.1",
                        oci_openai_gpt_4_1_response,
                        remove_base64_images_from_text(oci_openai_gpt_4_1_image_response),
                        oci_openai_gpt_4_1_evaluation
                    ]
                )

            if "oci_xai/grok-4" in llm_answer_checkbox_group:
                oci_xai_grok_4_response = oci_xai_grok_4_response
                if llm_evaluation_checkbox:
                    oci_xai_grok_4_evaluation = oci_xai_grok_4_evaluation
                else:
                    oci_xai_grok_4_evaluation = ""

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
                        "oci_xai/grok-4",
                        oci_xai_grok_4_response,
                        "",  # Vision機能なし
                        oci_xai_grok_4_evaluation
                    ]
                )

            if "oci_cohere/command-a" in llm_answer_checkbox_group:
                oci_cohere_command_a_response = oci_cohere_command_a_response
                if llm_evaluation_checkbox:
                    oci_cohere_command_a_evaluation = oci_cohere_command_a_evaluation
                else:
                    oci_cohere_command_a_evaluation = ""

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
                        "oci_cohere/command-a",
                        oci_cohere_command_a_response,
                        "",  # Vision機能なし
                        oci_cohere_command_a_evaluation
                    ]
                )

            if "oci_meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox_group:
                oci_meta_llama_4_scout_response = oci_meta_llama_4_scout_response
                if llm_evaluation_checkbox:
                    oci_meta_llama_4_scout_evaluation = oci_meta_llama_4_scout_evaluation
                else:
                    oci_meta_llama_4_scout_evaluation = ""

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
                        "oci_meta/llama-4-scout-17b-16e-instruct",
                        oci_meta_llama_4_scout_response,
                        remove_base64_images_from_text(oci_meta_llama_4_scout_image_response),
                        oci_meta_llama_4_scout_evaluation
                    ]
                )

            if "openai/gpt-4o" in llm_answer_checkbox_group:
                openai_gpt_4o_response = openai_gpt_4o_response
                if llm_evaluation_checkbox:
                    openai_gpt_4o_evaluation = openai_gpt_4o_evaluation
                else:
                    openai_gpt_4o_evaluation = ""

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
                        "openai/gpt-4o",
                        openai_gpt_4o_response,
                        remove_base64_images_from_text(openai_gpt_4o_image_response),
                        openai_gpt_4o_evaluation
                    ]
                )

            if "azure_openai/gpt-4o" in llm_answer_checkbox_group:
                azure_openai_gpt_4o_response = azure_openai_gpt_4o_response
                if llm_evaluation_checkbox:
                    azure_openai_gpt_4o_evaluation = azure_openai_gpt_4o_evaluation
                else:
                    azure_openai_gpt_4o_evaluation = ""

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
                        "azure_openai/gpt-4o",
                        azure_openai_gpt_4o_response,
                        remove_base64_images_from_text(azure_openai_gpt_4o_image_response),
                        azure_openai_gpt_4o_evaluation
                    ]
                )

        conn.commit()
