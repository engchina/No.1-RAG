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
        xai_grok_4_response,
        command_a_response,
        llama_4_scout_response,
        openai_gpt4o_response,
        azure_openai_gpt4o_response,
        xai_grok_4_evaluation,
        command_a_evaluation,
        llama_4_scout_evaluation,
        openai_gpt4o_evaluation,
        azure_openai_gpt4o_evaluation,
        xai_grok_4_image_response,
        llama_4_scout_image_response,
        openai_gpt4o_image_response,
        azure_openai_gpt4o_image_response
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
    print("in insert_query_result() 開始...")  # クエリ結果挿入処理開始
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

            if "xai/grok-4" in llm_answer_checkbox_group:
                xai_grok_4_response = xai_grok_4_response
                if llm_evaluation_checkbox:
                    xai_grok_4_evaluation = xai_grok_4_evaluation
                else:
                    xai_grok_4_evaluation = ""

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
                        "xai/grok-4",
                        xai_grok_4_response,
                        remove_base64_images_from_text(xai_grok_4_image_response),  # Vision機能対応
                        xai_grok_4_evaluation
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
                        "xai/grok-3",
                        xai_grok_3_response,
                        "",  # Vision機能なし
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
                        "cohere/command-a",
                        command_a_response,
                        "",  # Vision機能なし
                        command_a_evaluation
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
                        "meta/llama-4-scout-17b-16e-instruct",
                        llama_4_scout_response,
                        remove_base64_images_from_text(llama_4_scout_image_response),
                        llama_4_scout_evaluation
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
                        openai_gpt4o_response,
                        remove_base64_images_from_text(openai_gpt4o_image_response),
                        openai_gpt4o_evaluation
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
                        azure_openai_gpt4o_response,
                        remove_base64_images_from_text(azure_openai_gpt4o_image_response),
                        azure_openai_gpt4o_evaluation
                    ]
                )



        conn.commit()
