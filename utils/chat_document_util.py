"""
チャット操作関連の関数を提供するモジュール

このモジュールには以下の機能が含まれています：
- ドキュメントチャット (chat_document)
- 引用追加 (append_citation)
"""

import gradio as gr

from utils.chat_util import chat
from utils.openai_compatible_util import OPENAI_COMPATIBLE_MODEL_KEY
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
    """
    if use_image:
        include_citation = False
        include_current_time = False
        print("Vision 回答がオンのため、include_citation=False, include_current_time=Falseに設定されました")

    has_error = False
    if not query_text:
        has_error = True
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        has_error = True
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        has_error = True
        gr.Warning("検索結果が見つかりませんでした。設定もしくはクエリを変更して再度ご確認ください。")
    if has_error:
        yield (
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
            gr.Markdown(value=""),
        )
        return

    query_text = query_text.strip()

    oci_xai_grok_4_response = ""
    oci_cohere_command_a_response = ""
    oci_meta_llama_4_scout_response = ""
    openai_compatible_response = ""

    oci_xai_grok_4_checkbox = "oci_xai/grok-4.3" in llm_answer_checkbox
    oci_cohere_command_a_checkbox = "oci_cohere/command-a" in llm_answer_checkbox
    oci_meta_llama_4_scout_checkbox = "oci_meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox
    openai_compatible_checkbox = OPENAI_COMPATIBLE_MODEL_KEY in llm_answer_checkbox

    context = search_result[['EMBED_ID', 'SOURCE', 'CONTENT']].to_dict('records')
    system_text = ""

    if use_image:
        fixed_image_message = """Vision 回答モードが有効です。

画像データをVisionモデルで解析して回答します。

テキストベースの回答をご希望の場合は、Vision 回答をオフにしてください。"""
        user_text = fixed_image_message
    else:
        user_text = get_langgpt_rag_prompt(context, query_text, include_citation, include_current_time,
                                           rag_prompt_template)

    if use_image:
        if oci_xai_grok_4_checkbox:
            oci_xai_grok_4_response = fixed_image_message
        if oci_cohere_command_a_checkbox:
            oci_cohere_command_a_response = fixed_image_message
        if oci_meta_llama_4_scout_checkbox:
            oci_meta_llama_4_scout_response = fixed_image_message
        if openai_compatible_checkbox:
            openai_compatible_response = fixed_image_message

        yield (
            gr.Markdown(value=oci_xai_grok_4_response),
            gr.Markdown(value=oci_cohere_command_a_response),
            gr.Markdown(value=oci_meta_llama_4_scout_response),
            gr.Markdown(value=openai_compatible_response),
        )
        return

    async for oci_xai_grok_4, oci_cohere_command_a, oci_meta_llama_4_scout, openai_compatible in chat(
            system_text,
            None,
            user_text,
            oci_cohere_command_a_user_text=user_text,
            oci_meta_llama_4_scout_user_image=None,
            oci_meta_llama_4_scout_user_text=user_text,
            openai_compatible_user_image=None,
            openai_compatible_user_text=user_text,
            oci_xai_grok_4_checkbox=oci_xai_grok_4_checkbox,
            oci_cohere_command_a_checkbox=oci_cohere_command_a_checkbox,
            oci_meta_llama_4_scout_checkbox=oci_meta_llama_4_scout_checkbox,
            openai_compatible_checkbox=openai_compatible_checkbox,
    ):
        oci_xai_grok_4_response += oci_xai_grok_4
        oci_cohere_command_a_response += oci_cohere_command_a
        oci_meta_llama_4_scout_response += oci_meta_llama_4_scout
        openai_compatible_response += openai_compatible
        yield (
            gr.Markdown(value=oci_xai_grok_4_response),
            gr.Markdown(value=oci_cohere_command_a_response),
            gr.Markdown(value=oci_meta_llama_4_scout_response),
            gr.Markdown(value=openai_compatible_response),
        )


async def append_citation(
        search_result,
        llm_answer_checkbox,
        include_citation,
        use_image,
        query_text,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        oci_xai_grok_4_answer_text,
        oci_cohere_command_a_answer_text,
        oci_meta_llama_4_scout_answer_text,
        openai_compatible_answer_text,
):
    """
    LLMの回答に引用情報を追加する
    """
    if use_image:
        include_citation = False
        print("Vision 回答がオンのため、append_citation内でinclude_citation=Falseに設定されました")

    has_error = False
    if not query_text:
        has_error = True
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        has_error = True
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        has_error = True

    if has_error or not include_citation:
        yield (
            gr.Markdown(value=oci_xai_grok_4_answer_text),
            gr.Markdown(value=oci_cohere_command_a_answer_text),
            gr.Markdown(value=oci_meta_llama_4_scout_answer_text),
            gr.Markdown(value=openai_compatible_answer_text),
        )
        return

    if "oci_xai/grok-4.3" in llm_answer_checkbox:
        oci_xai_grok_4_answer_text = extract_and_format(oci_xai_grok_4_answer_text, search_result)
    if "oci_cohere/command-a" in llm_answer_checkbox:
        oci_cohere_command_a_answer_text = extract_and_format(oci_cohere_command_a_answer_text, search_result)
    if "oci_meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox:
        oci_meta_llama_4_scout_answer_text = extract_and_format(oci_meta_llama_4_scout_answer_text, search_result)
    if OPENAI_COMPATIBLE_MODEL_KEY in llm_answer_checkbox:
        openai_compatible_answer_text = extract_and_format(openai_compatible_answer_text, search_result)

    yield (
        gr.Markdown(value=oci_xai_grok_4_answer_text),
        gr.Markdown(value=oci_cohere_command_a_answer_text),
        gr.Markdown(value=oci_meta_llama_4_scout_answer_text),
        gr.Markdown(value=openai_compatible_answer_text),
    )
