"""
評価処理ハンドラーモジュール

このモジュールは、LLMの回答に対する人間評価とRAGAS評価の機能を提供します。
"""

import gradio as gr

from utils.chat_util import chat
from utils.openai_compatible_util import OPENAI_COMPATIBLE_MODEL_KEY


def eval_by_human(
        query_id,
        llm_name,
        human_evaluation_result,
        user_comment,
        pool
):
    """
    人間による評価結果をデータベースに保存する関数
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
        oci_xai_grok_4_response,
        oci_cohere_command_a_response,
        oci_meta_llama_4_scout_response,
        openai_compatible_response,
):
    """
    RAGAS評価を実行する関数
    """
    empty_result = (
        gr.Markdown(value=""),
        gr.Markdown(value=""),
        gr.Markdown(value=""),
        gr.Markdown(value=""),
    )

    if use_image:
        print("Vision回答がオンのため、LLM評価をスキップします")
        yield empty_result
        return

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
        yield empty_result
        return

    def remove_last_line(text):
        """推論時間の行を削除するヘルパー関数"""
        if text:
            lines = text.splitlines()
            if lines and lines[-1].startswith("推論時間"):
                lines.pop()
            return '\n'.join(lines)
        return text

    standard_answer_text = standard_answer_text.strip() if standard_answer_text else "入力されていません。"

    print(f"{llm_evaluation_checkbox=}")
    if not llm_evaluation_checkbox:
        yield empty_result
        return

    oci_xai_grok_4_checkbox = "oci_xai/grok-4.3" in llm_answer_checkbox_group
    oci_cohere_command_a_checkbox = "oci_cohere/command-a" in llm_answer_checkbox_group
    oci_meta_llama_4_scout_checkbox = "oci_meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox_group
    openai_compatible_checkbox = OPENAI_COMPATIBLE_MODEL_KEY in llm_answer_checkbox_group

    oci_xai_grok_4_response = remove_last_line(oci_xai_grok_4_response)
    oci_cohere_command_a_response = remove_last_line(oci_cohere_command_a_response)
    oci_meta_llama_4_scout_response = remove_last_line(oci_meta_llama_4_scout_response)
    openai_compatible_response = remove_last_line(openai_compatible_response)

    def build_eval_prompt(answer):
        return f"""
-標準回答-
{standard_answer_text}

-与えられた回答-
{answer}

-出力-\n　"""

    oci_xai_grok_4_user_text = build_eval_prompt(oci_xai_grok_4_response)
    oci_cohere_command_a_user_text = build_eval_prompt(oci_cohere_command_a_response)
    oci_meta_llama_4_scout_user_text = build_eval_prompt(oci_meta_llama_4_scout_response)
    openai_compatible_user_text = build_eval_prompt(openai_compatible_response)

    eval_oci_xai_grok_4_response = ""
    eval_oci_cohere_command_a_response = ""
    eval_oci_meta_llama_4_scout_response = ""
    eval_openai_compatible_response = ""

    async for oci_xai_grok_4, oci_cohere_command_a, oci_meta_llama_4_scout, openai_compatible in chat(
            system_text,
            None,
            oci_xai_grok_4_user_text,
            oci_cohere_command_a_user_text,
            None,
            oci_meta_llama_4_scout_user_text,
            None,
            openai_compatible_user_text,
            oci_xai_grok_4_checkbox,
            oci_cohere_command_a_checkbox,
            oci_meta_llama_4_scout_checkbox,
            openai_compatible_checkbox,
    ):
        eval_oci_xai_grok_4_response += oci_xai_grok_4
        eval_oci_cohere_command_a_response += oci_cohere_command_a
        eval_oci_meta_llama_4_scout_response += oci_meta_llama_4_scout
        eval_openai_compatible_response += openai_compatible

        yield (
            gr.Markdown(value=eval_oci_xai_grok_4_response),
            gr.Markdown(value=eval_oci_cohere_command_a_response),
            gr.Markdown(value=eval_oci_meta_llama_4_scout_response),
            gr.Markdown(value=eval_openai_compatible_response),
        )
