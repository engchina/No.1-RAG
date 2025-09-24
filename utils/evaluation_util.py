"""
評価処理ハンドラーモジュール

このモジュールは、LLMの回答に対する人間評価とRAGAS評価の機能を提供します。
"""

import gradio as gr

from utils.chat_util import chat


def eval_by_human(
        query_id,
        llm_name,
        human_evaluation_result,
        user_comment,
        pool
):
    """
    人間による評価結果をデータベースに保存する関数
    
    Args:
        query_id: クエリID
        llm_name: LLMモデル名
        human_evaluation_result: 人間による評価結果
        user_comment: ユーザーコメント
        pool: データベース接続プール
        
    Returns:
        tuple: Gradio コンポーネントのタプル
    """
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


def reset_eval_by_human_result():
    """
    人間評価結果をリセットする関数
    
    Returns:
        tuple: リセットされたGradio コンポーネントのタプル
    """
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
    )


async def eval_by_ragas(
        query_text,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        search_result,
        llm_answer_checkbox_group,
        llm_evaluation_checkbox,
        use_image,
        system_text,
        standard_answer_text,
        oci_openai_gpt_5_response,
        oci_openai_o3_response,
        oci_openai_gpt_4_1_response,
        oci_xai_grok_4_response,
        oci_cohere_command_a_response,
        oci_meta_llama_4_scout_response,
        openai_gpt_4o_response,
        azure_openai_gpt_4o_response,
):
    """
    RAGAS評価を実行する関数
    
    Args:
        query_text: クエリテキスト
        doc_id_all_checkbox_input: 全ドキュメント選択フラグ
        doc_id_checkbox_group_input: 選択されたドキュメントIDリスト
        search_result: 検索結果
        llm_answer_checkbox_group: 選択されたLLMモデルリスト
        llm_evaluation_checkbox: LLM評価有効フラグ
        use_image: 画像使用フラグ
        system_text: システムメッセージ
        standard_answer_text: 標準回答テキスト
        *_response: 各LLMモデルの回答
        
    Yields:
        tuple: 各モデルの評価結果をGradio Markdownコンポーネントとして返すタプル
    """
    # Vision回答がONの場合、LLM評価をスキップ
    if use_image:
        print("Vision回答がオンのため、LLM評価をスキップします")
        yield (
            gr.Markdown(value=""),  # oci_openai_gpt_5_evaluation
            gr.Markdown(value=""),  # oci_openai_o3_evaluation
            gr.Markdown(value=""),  # oci_openai_gpt_4_1_response
            gr.Markdown(value=""),  # oci_xai_grok_4_evaluation
            gr.Markdown(value=""),  # oci_cohere_command_a_evaluation
            gr.Markdown(value=""),  # oci_meta_llama_4_scout_evaluation
            gr.Markdown(value=""),  # openai_gpt_4o_evaluation
            gr.Markdown(value=""),  # azure_openai_gpt_4o_evaluation
        )
        return

    # 入力検証
    has_error = False
    if not query_text:
        has_error = True
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        has_error = True
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        has_error = True
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
            gr.Markdown(value=""), # oci_openai_gpt_5_evaluation
            gr.Markdown(value=""), # oci_openai_o3_evaluation
            gr.Markdown(value=""), # oci_openai_gpt_4_1_response
            gr.Markdown(value=""), # oci_xai_grok_4_evaluation
            gr.Markdown(value=""), # oci_cohere_command_a_evaluation
            gr.Markdown(value=""), # oci_meta_llama_4_scout_evaluation
            gr.Markdown(value=""), # openai_gpt_4o_evaluation
            gr.Markdown(value=""), # azure_openai_gpt_4o_evaluation
        )
        return

    def remove_last_line(text):
        """推論時間の行を削除するヘルパー関数"""
        if text:
            lines = text.splitlines()
            if lines[-1].startswith("推論時間"):
                lines.pop()
            return '\n'.join(lines)
        else:
            return text

    # 標準回答の準備
    if standard_answer_text:
        standard_answer_text = standard_answer_text.strip()
    else:
        standard_answer_text = "入力されていません。"

    print(f"{llm_evaluation_checkbox=}")
    if not llm_evaluation_checkbox:
        yield (
            gr.Markdown(value=""), # oci_openai_gpt_5_evaluation
            gr.Markdown(value=""), # oci_openai_o3_evaluation
            gr.Markdown(value=""), # oci_openai_gpt_4_1_response
            gr.Markdown(value=""), # oci_xai_grok_4_evaluation
            gr.Markdown(value=""), # oci_cohere_command_a_evaluation
            gr.Markdown(value=""), # oci_meta_llama_4_scout_evaluation
            gr.Markdown(value=""), # openai_gpt_4o_evaluation
            gr.Markdown(value=""), # azure_openai_gpt_4o_evaluation
        )
        return

    # 各モデルのチェックボックス状態を初期化
    oci_openai_gpt_5_checkbox = False
    oci_openai_o3_checkbox = False
    oci_openai_gpt_4_1_checkbox = False
    oci_xai_grok_4_checkbox = False
    oci_cohere_command_a_checkbox = False
    oci_meta_llama_4_scout_checkbox = False
    openai_gpt_4o_checkbox = False
    azure_openai_gpt_4o_checkbox = False

    # 選択されたモデルに基づいてチェックボックス状態を設定
    if "oci_openai/gpt-5" in llm_answer_checkbox_group:
        oci_openai_gpt_5_checkbox = True
    if "oci_openai/o3" in llm_answer_checkbox_group:
        oci_openai_o3_checkbox = True
    if "oci_openai/gpt-4.1" in llm_answer_checkbox_group:
        oci_openai_gpt_4_1_checkbox = True
    if "oci_xai/grok-4" in llm_answer_checkbox_group:
        oci_xai_grok_4_checkbox = True
    if "oci_cohere/command-a" in llm_answer_checkbox_group:
        oci_cohere_command_a_checkbox = True
    if "oci_meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox_group:
        oci_meta_llama_4_scout_checkbox = True
    if "openai/gpt-4o" in llm_answer_checkbox_group:
        openai_gpt_4o_checkbox = True
    if "azure_openai/gpt-4o" in llm_answer_checkbox_group:
        azure_openai_gpt_4o_checkbox = True

    # 各回答から推論時間の行を削除
    oci_openai_gpt_5_response = remove_last_line(oci_openai_gpt_5_response)
    oci_openai_o3_response = remove_last_line(oci_openai_o3_response)
    oci_openai_gpt_4_1_response = remove_last_line(oci_openai_gpt_4_1_response)
    oci_xai_grok_4_response = remove_last_line(oci_xai_grok_4_response)
    oci_cohere_command_a_response = remove_last_line(oci_cohere_command_a_response)
    oci_meta_llama_4_scout_response = remove_last_line(oci_meta_llama_4_scout_response)
    openai_gpt_4o_response = remove_last_line(openai_gpt_4o_response)
    azure_openai_gpt_4o_response = remove_last_line(azure_openai_gpt_4o_response)

    # 各モデル用の評価プロンプトを構築
    oci_openai_gpt_5_user_text = f"""
-標準回答-
 {standard_answer_text}

-与えられた回答-
 {oci_openai_gpt_5_response}

-出力-\n　"""

    oci_openai_o3_user_text = f"""
-標準回答-
 {standard_answer_text}

-与えられた回答-
 {oci_openai_o3_response}

-出力-\n　"""

    oci_openai_gpt_4_1_user_text = f"""
-標準回答-
 {standard_answer_text}

-与えられた回答-
 {oci_openai_gpt_4_1_response}

-出力-\n　"""

    oci_xai_grok_4_user_text = f"""
-標準回答-
 {standard_answer_text}

-与えられた回答-
 {oci_xai_grok_4_response}

-出力-\n　"""

    oci_cohere_command_a_user_text = f"""
-標準回答-
 {standard_answer_text}

-与えられた回答-
 {oci_cohere_command_a_response}

-出力-\n　"""

    oci_meta_llama_4_scout_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{oci_meta_llama_4_scout_response}

-出力-\n　"""

    openai_gpt_4o_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{openai_gpt_4o_response}

-出力-\n　"""

    azure_openai_gpt_4o_user_text = f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{azure_openai_gpt_4o_response}

-出力-\n　"""

    # 評価応答を初期化
    eval_oci_openai_gpt_5_response = ""
    eval_oci_openai_o3_response = ""
    eval_oci_openai_gpt_4_1_response = ""
    eval_oci_xai_grok_4_response = ""
    eval_oci_cohere_command_a_response = ""
    eval_oci_meta_llama_4_scout_response = ""
    eval_openai_gpt_4o_response = ""
    eval_azure_openai_gpt_4o_response = ""

    # chat関数を呼び出して評価を実行
    async for oci_openai_gpt_5, oci_openai_o3, oci_openai_gpt_4_1, oci_xai_grok_4, oci_cohere_command_a, oci_meta_llama_4_scout, gpt_4o, azure_gpt_4o in chat(
            system_text,
            None,
            oci_openai_gpt_5_user_text,
            None,
            oci_openai_o3_user_text,
            None,
            oci_openai_gpt_4_1_user_text,
            oci_xai_grok_4_user_text,
            oci_cohere_command_a_user_text,
            None,
            oci_meta_llama_4_scout_user_text,
            openai_gpt_4o_user_text,
            azure_openai_gpt_4o_user_text,
            oci_openai_gpt_5_checkbox,
            oci_openai_o3_checkbox,
            oci_openai_gpt_4_1_checkbox,
            oci_xai_grok_4_checkbox,
            oci_cohere_command_a_checkbox,
            oci_meta_llama_4_scout_checkbox,
            openai_gpt_4o_checkbox,
            azure_openai_gpt_4o_checkbox,
    ):
        # 評価結果を累積
        eval_oci_openai_gpt_5_response += oci_openai_gpt_5
        eval_oci_openai_o3_response += oci_openai_o3
        eval_oci_openai_gpt_4_1_response += oci_openai_gpt_4_1
        eval_oci_xai_grok_4_response += oci_xai_grok_4
        eval_oci_cohere_command_a_response += oci_cohere_command_a
        eval_oci_meta_llama_4_scout_response += oci_meta_llama_4_scout
        eval_openai_gpt_4o_response += gpt_4o
        eval_azure_openai_gpt_4o_response += azure_gpt_4o

        # Gradio Markdownコンポーネントとして結果を返す
        yield (
            gr.Markdown(value=eval_oci_openai_gpt_5_response),
            gr.Markdown(value=eval_oci_openai_o3_response),
            gr.Markdown(value=eval_oci_openai_gpt_4_1_response),
            gr.Markdown(value=eval_oci_xai_grok_4_response),
            gr.Markdown(value=eval_oci_cohere_command_a_response),
            gr.Markdown(value=eval_oci_meta_llama_4_scout_response),
            gr.Markdown(value=eval_openai_gpt_4o_response),
            gr.Markdown(value=eval_azure_openai_gpt_4o_response),
        )
