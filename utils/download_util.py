"""
ダウンロードファイル生成ユーティリティモジュール

このモジュールは、検索結果やLLM応答をExcelファイルとして生成し、
ダウンロード可能な形式で提供するための関数を提供します。
"""

import gradio as gr
import pandas as pd

from utils.openai_compatible_util import OPENAI_COMPATIBLE_MODEL_KEY, get_openai_compatible_llm_name
from utils.text_util import extract_citation, remove_base64_images_from_text


def generate_download_file(
        search_result,
        llm_answer_checkbox_group,
        include_citation,
        use_image,
        llm_evaluation_checkbox,
        query_text,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
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
    検索結果とLLM応答からダウンロード用のExcelファイルを生成する
    """
    if use_image:
        include_citation = False
        print("Vision 回答がオンのため、generate_download_file内でinclude_citation=Falseに設定されました")

    if not query_text:
        return gr.DownloadButton(value=None, visible=False)
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        return gr.DownloadButton(value=None, visible=False)
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        return gr.DownloadButton(value=None, visible=False)

    standard_answer_text = standard_answer_text if llm_evaluation_checkbox else ""
    df1 = pd.DataFrame({'クエリ': [query_text], '標準回答': [standard_answer_text]})
    df2 = search_result

    def prepare_model(model_key, response, evaluation):
        if model_key not in llm_answer_checkbox_group:
            return "", "", ""
        referenced_contexts = ""
        if include_citation:
            response, referenced_contexts = extract_citation(response)
        if not llm_evaluation_checkbox:
            evaluation = ""
        return response, referenced_contexts, evaluation

    oci_xai_grok_4_response, oci_xai_grok_4_referenced_contexts, oci_xai_grok_4_evaluation = prepare_model(
        "oci_xai/grok-4.3",
        oci_xai_grok_4_response,
        oci_xai_grok_4_evaluation
    )
    oci_cohere_command_a_response, oci_cohere_command_a_referenced_contexts, oci_cohere_command_a_evaluation = prepare_model(
        "oci_cohere/command-a",
        oci_cohere_command_a_response,
        oci_cohere_command_a_evaluation
    )
    oci_meta_llama_4_scout_response, oci_meta_llama_4_scout_referenced_contexts, oci_meta_llama_4_scout_evaluation = prepare_model(
        "oci_meta/llama-4-scout-17b-16e-instruct",
        oci_meta_llama_4_scout_response,
        oci_meta_llama_4_scout_evaluation
    )
    openai_compatible_response, openai_compatible_referenced_contexts, openai_compatible_evaluation = prepare_model(
        OPENAI_COMPATIBLE_MODEL_KEY,
        openai_compatible_response,
        openai_compatible_evaluation
    )

    df3 = pd.DataFrame(
        {
            'LLM モデル': [
                "oci_xai/grok-4.3",
                "oci_cohere/command-a",
                "oci_meta/llama-4-scout-17b-16e-instruct",
                get_openai_compatible_llm_name(),
            ],
            'LLM メッセージ': [
                oci_xai_grok_4_response,
                oci_cohere_command_a_response,
                oci_meta_llama_4_scout_response,
                openai_compatible_response,
            ],
            'Vision 回答': [
                remove_base64_images_from_text(oci_xai_grok_4_image_response),
                "",
                remove_base64_images_from_text(oci_meta_llama_4_scout_image_response),
                remove_base64_images_from_text(openai_compatible_image_response),
            ],
            '引用 Contexts': [
                oci_xai_grok_4_referenced_contexts,
                oci_cohere_command_a_referenced_contexts,
                oci_meta_llama_4_scout_referenced_contexts,
                openai_compatible_referenced_contexts,
            ],
            'LLM 評価結果': [
                oci_xai_grok_4_evaluation,
                oci_cohere_command_a_evaluation,
                oci_meta_llama_4_scout_evaluation,
                openai_compatible_evaluation,
            ]
        }
    )

    filepath = '/tmp/query_result.xlsx'
    with pd.ExcelWriter(filepath) as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
        df3.to_excel(writer, sheet_name='Sheet3', index=False)

    print(f"Excelファイルが {filepath} に保存されました")
    return gr.DownloadButton(value=filepath, visible=True)
