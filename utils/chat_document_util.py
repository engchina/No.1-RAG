"""
チャット操作関連の関数を提供するモジュール

このモジュールには以下の機能が含まれています：
- ドキュメントチャット (chat_document)
- 引用追加 (append_citation)
- クエリ結果挿入 (insert_query_result)
"""

import gradio as gr

from utils.chat_util import chat
from utils.prompts_util import get_langgpt_rag_prompt
from utils.text_util import extract_and_format


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
    検索結果を使用してLLMとチャットする

    Args:
        search_result: 検索結果のDataFrame
        llm_answer_checkbox: 選択されたLLMモデルのリスト
        include_citation: 引用を含めるかどうか
        include_current_time: 現在時刻を含めるかどうか
        use_image: 画像を使用するかどうか
        query_text: クエリテキスト
        doc_id_all_checkbox_input: 全ドキュメント選択フラグ
        doc_id_checkbox_group_input: 選択されたドキュメントIDのリスト
        rag_prompt_template: RAGプロンプトテンプレート

    Yields:
        tuple: 各LLMの回答を含むGradio Markdownコンポーネントのタプル
    """
    # Vision 回答がオンの場合、引用と時間の設定を固定でFalseにする
    if use_image:
        include_citation = False
        include_current_time = False
        print("Vision 回答がオンのため、include_citation=False, include_current_time=Falseに設定されました")  # Vision回答設定ログ

    has_error = False
    if not query_text:
        has_error = True
        # gr.Warning("クエリを入力してください")
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        has_error = True
        # gr.Warning("ドキュメントを選択してください")
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        has_error = True
        gr.Warning("検索結果が見つかりませんでした。設定もしくはクエリを変更して再度ご確認ください。")
    if has_error:
        yield (
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value="")
        )
        return

    query_text = query_text.strip()

    xai_grok_4_response = ""
    command_a_response = ""
    llama_4_scout_response = ""
    openai_gpt4o_response = ""
    azure_openai_gpt4o_response = ""

    xai_grok_4_checkbox = False
    command_a_checkbox = False
    llama_4_scout_checkbox = False
    openai_gpt4o_checkbox = False
    azure_openai_gpt4o_checkbox = False
    if "xai/grok-4" in llm_answer_checkbox:
        xai_grok_4_checkbox = True
    if "cohere/command-a" in llm_answer_checkbox:
        command_a_checkbox = True
    if "meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox:
        llama_4_scout_checkbox = True
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt4o_checkbox = True
    if "azure_openai/gpt-4o" in llm_answer_checkbox:
        azure_openai_gpt4o_checkbox = True

    # context = '\n'.join(search_result['CONTENT'].astype(str).values)
    context = search_result[['EMBED_ID', 'SOURCE', 'CONTENT']].to_dict('records')

    system_text = ""

    # Vision 回答がオンの場合、固定メッセージを設定
    if use_image:
        fixed_image_message = """Vision 回答モードが有効です。

画像データをVisionモデルで解析して回答します。

テキストベースの回答をご希望の場合は、Vision 回答をオフにしてください。"""

        user_text = fixed_image_message
    else:
        user_text = get_langgpt_rag_prompt(context, query_text, include_citation, include_current_time,
                                           rag_prompt_template)

    xai_grok_4_user_text = user_text
    command_a_user_text = user_text
    llama_4_scout_user_text = user_text
    openai_gpt4o_user_text = user_text
    azure_openai_gpt4o_user_text = user_text

    # Vision 回答がオンの場合、固定メッセージを即座に返す
    if use_image:
        # 選択されたLLMに対してのみ固定メッセージを設定
        if xai_grok_4_checkbox:
            xai_grok_4_response = fixed_image_message
        if command_a_checkbox:
            command_a_response = fixed_image_message
        if llama_4_scout_checkbox:
            llama_4_scout_response = fixed_image_message
        if openai_gpt4o_checkbox:
            openai_gpt4o_response = fixed_image_message
        if azure_openai_gpt4o_checkbox:
            azure_openai_gpt4o_response = fixed_image_message

        # 固定メッセージを一度だけ返す
        yield (
            gr.Markdown(value=xai_grok_4_response),
            gr.Markdown(value=command_a_response),
            gr.Markdown(value=llama_4_scout_response),
            gr.Markdown(value=openai_gpt4o_response),
            gr.Markdown(value=azure_openai_gpt4o_response)
        )
    else:
        # 通常のLLM処理
        async for xai_grok_4, command_a, llama_4_scout, gpt4o, azure_gpt4o in chat(
                system_text,
                None,  # xai_grok_4_user_image - 通常のRAGでは画像なし
                xai_grok_4_user_text,
                command_a_user_text,
                None,  # llama_4_scout_user_image - 通常のRAGでは画像なし
                llama_4_scout_user_text,
                openai_gpt4o_user_text,
                azure_openai_gpt4o_user_text,
                xai_grok_4_checkbox,
                command_a_checkbox,
                llama_4_scout_checkbox,
                openai_gpt4o_checkbox,
                azure_openai_gpt4o_checkbox
        ):
            xai_grok_4_response += xai_grok_4
            command_a_response += command_a
            llama_4_scout_response += llama_4_scout
            openai_gpt4o_response += gpt4o
            azure_openai_gpt4o_response += azure_gpt4o
            yield (
                gr.Markdown(value=xai_grok_4_response),
                gr.Markdown(value=command_a_response),
                gr.Markdown(value=llama_4_scout_response),
                gr.Markdown(value=openai_gpt4o_response),
                gr.Markdown(value=azure_openai_gpt4o_response),
            )


async def append_citation(
        search_result,
        llm_answer_checkbox,
        include_citation,
        use_image,
        query_text,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        xai_grok_4_answer_text,
        command_a_answer_text,
        llama_4_scout_answer_text,
        openai_gpt4o_answer_text,
        azure_openai_gpt4o_answer_text,
):
    """
    LLMの回答に引用情報を追加する

    Args:
        search_result: 検索結果のDataFrame
        llm_answer_checkbox: 選択されたLLMモデルのリスト
        include_citation: 引用を含めるかどうか
        use_image: 画像を使用するかどうか
        query_text: クエリテキスト
        doc_id_all_checkbox_input: 全ドキュメント選択フラグ
        doc_id_checkbox_group_input: 選択されたドキュメントIDのリスト
        各LLMの回答テキスト

    Yields:
        tuple: 引用情報が追加された各LLMの回答を含むGradio Markdownコンポーネントのタプル
    """
    # Vision 回答がオンの場合、引用設定を固定でFalseにする
    if use_image:
        include_citation = False
        print("Vision 回答がオンのため、append_citation内でinclude_citation=Falseに設定されました")

    has_error = False
    if not query_text:
        has_error = True
        # gr.Warning("クエリを入力してください")
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        has_error = True
        # gr.Warning("ドキュメントを選択してください")
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        has_error = True
        # gr.Warning("検索結果が見つかりませんでした。設定もしくはクエリを変更して再度ご確認ください。")
    if has_error:
        yield (
            gr.Markdown(value=xai_grok_4_answer_text),
            gr.Markdown(value=command_a_answer_text),
            gr.Markdown(value=llama_4_scout_answer_text),
            gr.Markdown(value=openai_gpt4o_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_answer_text),
        )
        return

    if not include_citation:
        yield (
            gr.Markdown(value=xai_grok_4_answer_text),
            gr.Markdown(value=command_a_answer_text),
            gr.Markdown(value=llama_4_scout_answer_text),
            gr.Markdown(value=openai_gpt4o_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_answer_text)
        )
        return

    if "xai/grok-4" in llm_answer_checkbox:
        xai_grok_4_answer_text = extract_and_format(xai_grok_4_answer_text, search_result)
    if "cohere/command-a" in llm_answer_checkbox:
        command_a_answer_text = extract_and_format(command_a_answer_text, search_result)
    if "meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox:
        llama_4_scout_answer_text = extract_and_format(llama_4_scout_answer_text, search_result)
    if "openai/gpt-4o" in llm_answer_checkbox:
        openai_gpt4o_answer_text = extract_and_format(openai_gpt4o_answer_text, search_result)
    if "azure_openai/gpt-4o" in llm_answer_checkbox:
        azure_openai_gpt4o_answer_text = extract_and_format(azure_openai_gpt4o_answer_text, search_result)
    yield (
        gr.Markdown(value=xai_grok_4_answer_text),
        gr.Markdown(value=command_a_answer_text),
        gr.Markdown(value=llama_4_scout_answer_text),
        gr.Markdown(value=openai_gpt4o_answer_text),
        gr.Markdown(value=azure_openai_gpt4o_answer_text)
    )
    return
