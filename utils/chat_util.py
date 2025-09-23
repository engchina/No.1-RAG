"""
チャット処理ハンドラーモジュール

このモジュールは、複数のLLMモデルを並行して実行し、
ストリーミング形式で結果を返すチャット処理機能を提供します。
"""

import gradio as gr

from utils.llm_tasks_util import (
    oci_openai_gpt_5_task, oci_openai_o3_task, 
    oci_openai_gpt_4_1_task, oci_xai_grok_4_task,
    oci_cohere_command_a_task, oci_meta_llama_4_scout_task,
    openai_gpt_4o_task, azure_openai_gpt_4o_task
)


async def chat(
        system_text,
        oci_openai_gpt_5_user_image,
        oci_openai_gpt_5_user_text,
        oci_openai_o3_user_image,
        oci_openai_o3_user_text,
        oci_openai_gpt_4_1_user_image,
        oci_openai_gpt_4_1_user_text,
        oci_xai_grok_4_user_text,
        oci_cohere_command_a_user_text,
        oci_meta_llama_4_scout_user_image,
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
        azure_openai_gpt_4o_checkbox
):
    """
    複数のLLMモデルを並行実行し、ストリーミング形式で結果を返すチャット処理関数
    
    Args:
        system_text: システムメッセージ
        *_user_text: 各モデル用のユーザーテキスト
        *_user_image: 各モデル用のユーザー画像
        *_checkbox: 各モデルの有効/無効フラグ
        
    Yields:
        tuple: 各モデルからの応答のタプル
    """
    # 各モデルのジェネレーターを初期化
    oci_openai_gpt_5_gen = oci_openai_gpt_5_task(system_text, oci_openai_gpt_5_user_image, oci_openai_gpt_5_user_text,
                                                 oci_openai_gpt_5_checkbox) if oci_openai_gpt_5_checkbox else None
    oci_openai_o3_gen = oci_openai_o3_task(system_text, oci_openai_o3_user_image, oci_openai_o3_user_text,
                                           oci_openai_o3_checkbox) if oci_openai_o3_checkbox else None
    oci_openai_gpt_4_1_gen = oci_openai_gpt_4_1_task(system_text, oci_openai_gpt_4_1_user_image, oci_openai_gpt_4_1_user_text,
                                                     oci_openai_gpt_4_1_checkbox) if oci_openai_gpt_4_1_checkbox else None
    oci_xai_grok_4_gen = oci_xai_grok_4_task(system_text, oci_xai_grok_4_user_text,
                                             oci_xai_grok_4_checkbox) if oci_xai_grok_4_checkbox else None
    oci_cohere_command_a_gen = oci_cohere_command_a_task(system_text, oci_cohere_command_a_user_text,
                                                         oci_cohere_command_a_checkbox) if oci_cohere_command_a_checkbox else None
    oci_meta_llama_4_scout_gen = oci_meta_llama_4_scout_task(system_text, oci_meta_llama_4_scout_user_image,
                                                             oci_meta_llama_4_scout_user_text,
                                                             oci_meta_llama_4_scout_checkbox) if oci_meta_llama_4_scout_checkbox else None
    openai_gpt_4o_gen = openai_gpt_4o_task(system_text, openai_gpt_4o_user_text,
                                         openai_gpt_4o_checkbox) if openai_gpt_4o_checkbox else None
    azure_openai_gpt_4o_gen = azure_openai_gpt_4o_task(system_text, azure_openai_gpt_4o_user_text,
                                                     azure_openai_gpt_4o_checkbox) if azure_openai_gpt_4o_checkbox else None

    # 応答状態とジェネレーター名の初期化
    responses_status = ["", "", "", "", "", "", "", ""]
    generator_names = ["OCI OpenAI GPT-5", "OCI OpenAI o3", "OCI OpenAI GPT-4.1", "OCI XAI Grok-4", "OCI Cohere Command-A",
                       "OCI Meta Llama-4-Scout", "OpenAI GPT-4o", "Azure OpenAI GPT-4o"]
    iteration_count = 0

    while True:
        iteration_count += 1
        responses = ["", "", "", "", "", "", "", ""]
        generators = [oci_openai_gpt_5_gen, oci_openai_o3_gen, oci_openai_gpt_4_1_gen, oci_xai_grok_4_gen, oci_cohere_command_a_gen,
                      oci_meta_llama_4_scout_gen, openai_gpt_4o_gen, azure_openai_gpt_4o_gen]

        active_generators = 0
        for i, gen in enumerate(generators):
            if responses_status[i] != "TASK_DONE":
                active_generators += 1
                try:
                    response = await anext(gen)
                    if response:
                        if response == "TASK_DONE":
                            responses_status[i] = response
                        else:
                            responses[i] = response
                            # 空の応答のデバッグログ
                            if not response.strip():
                                print(
                                    f"DEBUG: {generator_names[i]} yielded empty response (iteration {iteration_count})")
                except StopAsyncIteration:
                    responses_status[i] = "TASK_DONE"
                except Exception as e:
                    responses_status[i] = "TASK_DONE"

        yield tuple(responses)

        # すべてのタスクが完了したかチェック
        if all(response_status == "TASK_DONE" for response_status in responses_status):
            break


async def chat_stream(
        system_text,
        query_image,
        query_text,
        llm_answer_checkbox
):
    """
    チャットストリーミング処理関数

    ユーザーからの入力を検証し、適切なパラメータに変換してchat関数を呼び出し、
    Gradio UI用の形式でストリーミング結果を返します。

    Args:
        system_text: システムメッセージ
        query_image: クエリ画像
        query_text: クエリテキスト
        llm_answer_checkbox: 選択されたLLMモデルのリスト

    Yields:
        tuple: 各モデルからの応答をGradio Markdownコンポーネントとして返すタプル
    """
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
            ""
        )
        return

    # 各モデル用のパラメータを設定
    oci_openai_gpt_5_user_image = query_image
    oci_openai_gpt_5_user_text = query_text
    oci_openai_o3_user_image = query_image
    oci_openai_o3_user_text = query_text
    oci_openai_gpt_4_1_user_image = query_image
    oci_openai_gpt_4_1_user_text = query_text
    oci_xai_grok_4_user_text = query_text
    oci_cohere_command_a_user_text = query_text
    oci_meta_llama_4_scout_user_image = query_image
    oci_meta_llama_4_scout_user_text = query_text
    openai_gpt_4o_user_text = query_text
    azure_openai_gpt_4o_user_text = query_text

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
    if "oci_openai/gpt-5" in llm_answer_checkbox:
        oci_openai_gpt_5_checkbox = True
    if "oci_openai/o3" in llm_answer_checkbox:
        oci_openai_o3_checkbox = True
    if "oci_openai/gpt-4.1" in llm_answer_checkbox:
        oci_openai_gpt_4_1_checkbox = True
    if "oci_xai/grok-4" in llm_answer_checkbox:
        oci_xai_grok_4_checkbox = True
    if "oci_cohere/command-a" in llm_answer_checkbox:
        oci_cohere_command_a_checkbox = True
    if "oci_meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox:
        oci_meta_llama_4_scout_checkbox = True
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt_4o_checkbox = True
    if "azure_openai/gpt-4o" in llm_answer_checkbox:
        azure_openai_gpt_4o_checkbox = True

    # 各モデルの応答を初期化
    oci_openai_gpt_5_response = ""
    oci_openai_o3_response = ""
    oci_openai_gpt_4_1_response = ""
    oci_xai_grok_4_response = ""
    oci_cohere_command_a_response = ""
    oci_meta_llama_4_scout_response = ""
    openai_gpt_4o_response = ""
    azure_openai_gpt_4o_response = ""

    # chat関数を呼び出してストリーミング処理
    async for oci_openai_gpt_5, oci_openai_o3, oci_openai_gpt_4_1, oci_xai_grok_4, oci_cohere_command_a, oci_meta_llama_4_scout, gpt_4o, azure_gpt_4o in chat(
            system_text,
            oci_openai_gpt_5_user_image,
            oci_openai_gpt_5_user_text,
            oci_openai_o3_user_image,
            oci_openai_o3_user_text,
            oci_openai_gpt_4_1_user_image,
            oci_openai_gpt_4_1_user_text,
            oci_xai_grok_4_user_text,
            oci_cohere_command_a_user_text,
            oci_meta_llama_4_scout_user_image,
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
        # 各モデルからの応答を累積
        oci_openai_gpt_5_response += oci_openai_gpt_5
        oci_openai_o3_response += oci_openai_o3
        oci_openai_gpt_4_1_response += oci_openai_gpt_4_1
        oci_xai_grok_4_response += oci_xai_grok_4
        oci_cohere_command_a_response += oci_cohere_command_a
        oci_meta_llama_4_scout_response += oci_meta_llama_4_scout
        openai_gpt_4o_response += gpt_4o
        azure_openai_gpt_4o_response += azure_gpt_4o

        # Gradio Markdownコンポーネントとして結果を返す
        yield (
            gr.Markdown(value=oci_openai_gpt_5_response),
            gr.Markdown(value=oci_openai_o3_response),
            gr.Markdown(value=oci_openai_gpt_4_1_response),
            gr.Markdown(value=oci_xai_grok_4_response),
            gr.Markdown(value=oci_cohere_command_a_response),
            gr.Markdown(value=oci_meta_llama_4_scout_response),
            gr.Markdown(value=openai_gpt_4o_response),
            gr.Markdown(value=azure_openai_gpt_4o_response)
        )
