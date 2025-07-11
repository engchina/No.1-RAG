"""
チャット処理ハンドラーモジュール

このモジュールは、複数のLLMモデルを並行して実行し、
ストリーミング形式で結果を返すチャット処理機能を提供します。
"""

import gradio as gr

from utils.llm_tasks_util import (
    xai_grok_3_task, command_a_task, llama_3_3_70b_task, llama_3_2_90b_vision_task,
    llama_4_maverick_task, llama_4_scout_task, openai_gpt4o_task, openai_gpt4_task,
    azure_openai_gpt4o_task, azure_openai_gpt4_task
)


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
    # 各LLMタスクのジェネレーターを初期化
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

    # 応答状態とジェネレーター名の初期化
    responses_status = ["", "", "", "", "", "", "", "", "", ""]
    generator_names = ["XAI Grok-3", "Command-A", "Llama-4-Maverick", "Llama-4-Scout",
                       "Llama-3.3-70B", "Llama-3.2-90B-Vision", "OpenAI GPT-4o", "OpenAI GPT-4",
                       "Azure OpenAI GPT-4o", "Azure OpenAI GPT-4"]
    iteration_count = 0

    while True:
        iteration_count += 1
        responses = ["", "", "", "", "", "", "", "", "", ""]
        generators = [xai_grok_3_gen, command_a_gen,
                      llama_4_maverick_gen, llama_4_scout_gen,
                      llama_3_3_70b_gen, llama_3_2_90b_vision_gen,
                      openai_gpt4o_gen, openai_gpt4_gen,
                      azure_openai_gpt4o_gen, azure_openai_gpt4_gen]

        active_generators = 0
        for i, gen in enumerate(generators):
            if responses_status[i] != "TASK_DONE":
                active_generators += 1
                try:
                    response = await anext(gen)
                    if response:
                        if response == "TASK_DONE":
                            responses_status[i] = response
                            print(f"DEBUG: {generator_names[i]} completed (iteration {iteration_count})")
                        else:
                            responses[i] = response
                            # 空の応答のデバッグログ
                            if not response.strip():
                                print(
                                    f"DEBUG: {generator_names[i]} yielded empty response (iteration {iteration_count})")
                except StopAsyncIteration:
                    responses_status[i] = "TASK_DONE"
                    print(f"DEBUG: {generator_names[i]} stopped iteration (iteration {iteration_count})")
                except Exception as e:
                    print(f"ERROR: {generator_names[i]} generator failed: {e}")
                    responses_status[i] = "TASK_DONE"

        print(f"DEBUG: Iteration {iteration_count}, active generators: {active_generators}")
        yield tuple(responses)

        # すべてのタスクが完了したかチェック
        if all(response_status == "TASK_DONE" for response_status in responses_status):
            print(f"DEBUG: All tasks completed after {iteration_count} iterations")
            break


async def chat_stream(system_text, query_image, query_text, llm_answer_checkbox):
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
            "",
            "",
            ""
        )
        return

    # 各モデル用のパラメータを設定
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

    # 各モデルのチェックボックス状態を初期化
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

    # 選択されたモデルに基づいてチェックボックス状態を設定
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

    # 各モデルの応答を初期化
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

    # chat関数を呼び出してストリーミング処理
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
        # 各モデルからの応答を累積
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

        # Gradio Markdownコンポーネントとして結果を返す
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
