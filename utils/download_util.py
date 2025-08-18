"""
ダウンロードファイル生成ユーティリティモジュール

このモジュールは、検索結果やLLM応答をExcelファイルとして生成し、
ダウンロード可能な形式で提供するための関数を提供します。
"""

import gradio as gr
import pandas as pd

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
        oci_openai_o3_response,
        oci_xai_grok_4_response,
        oci_cohere_command_a_response,
        oci_meta_llama_4_scout_response,
        openai_gpt4o_response,
        azure_openai_gpt4o_response,
        oci_openai_o3_evaluation,
        oci_xai_grok_4_evaluation,
        oci_cohere_command_a_evaluation,
        oci_meta_llama_4_scout_evaluation,
        openai_gpt4o_evaluation,
        azure_openai_gpt4o_evaluation,
        oci_openai_o3_image_response,
        oci_meta_llama_4_scout_image_response,
        openai_gpt4o_image_response,
        azure_openai_gpt4o_image_response
):
    """
    検索結果とLLM応答からダウンロード用のExcelファイルを生成する

    Args:
        search_result: 検索結果のDataFrame
        llm_answer_checkbox_group: 選択されたLLMモデルのリスト
        include_citation: 引用を含めるかどうか
        use_image: Vision回答を使用するかどうか
        llm_evaluation_checkbox: LLM評価を含めるかどうか
        query_text: クエリテキスト
        doc_id_all_checkbox_input: 全ドキュメント選択フラグ
        doc_id_checkbox_group_input: 選択されたドキュメントIDのリスト
        standard_answer_text: 標準回答テキスト
        *_response: 各LLMモデルの応答
        *_evaluation: 各LLMモデルの評価結果
        *_image_response: 各LLMモデルのVision応答

    Returns:
        gr.DownloadButton: ダウンロードボタン
    """
    # Vision 回答がオンの場合、引用設定を固定でFalseにする
    if use_image:
        include_citation = False
        print("Vision 回答がオンのため、generate_download_file内でinclude_citation=Falseに設定されました")

    if not query_text:
        return gr.DownloadButton(value=None, visible=False)
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        return gr.DownloadButton(value=None, visible=False)
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        return gr.DownloadButton(value=None, visible=False)

    # サンプルDataFrameを作成
    if llm_evaluation_checkbox:
        standard_answer_text = standard_answer_text
    else:
        standard_answer_text = ""
    df1 = pd.DataFrame({'クエリ': [query_text], '標準回答': [standard_answer_text]})

    df2 = search_result

    if "oci_openai/o3" in llm_answer_checkbox_group:
        oci_openai_o3_response = oci_openai_o3_response
        oci_openai_o3_referenced_contexts = ""
        if include_citation:
            oci_openai_o3_response, oci_openai_o3_referenced_contexts = extract_citation(oci_openai_o3_response)
        if llm_evaluation_checkbox:
            oci_openai_o3_evaluation = oci_openai_o3_evaluation
        else:
            oci_openai_o3_evaluation = ""
    else:
        oci_openai_o3_response = ""
        oci_openai_o3_evaluation = ""
        oci_openai_o3_referenced_contexts = ""

    if "oci_xai/grok-4" in llm_answer_checkbox_group:
        oci_xai_grok_4_response = oci_xai_grok_4_response
        oci_xai_grok_4_referenced_contexts = ""
        if include_citation:
            oci_xai_grok_4_response, oci_xai_grok_4_referenced_contexts = extract_citation(oci_xai_grok_4_response)
        if llm_evaluation_checkbox:
            oci_xai_grok_4_evaluation = oci_xai_grok_4_evaluation
        else:
            oci_xai_grok_4_evaluation = ""
    else:
        oci_xai_grok_4_response = ""
        oci_xai_grok_4_evaluation = ""
        oci_xai_grok_4_referenced_contexts = ""

    if "oci_cohere/command-a" in llm_answer_checkbox_group:
        oci_cohere_command_a_response = oci_cohere_command_a_response
        oci_cohere_command_a_referenced_contexts = ""
        if include_citation:
            oci_cohere_command_a_response, oci_cohere_command_a_referenced_contexts = extract_citation(
                oci_cohere_command_a_response)
        if llm_evaluation_checkbox:
            oci_cohere_command_a_evaluation = oci_cohere_command_a_evaluation
        else:
            oci_cohere_command_a_evaluation = ""
    else:
        oci_cohere_command_a_response = ""
        oci_cohere_command_a_evaluation = ""
        oci_cohere_command_a_referenced_contexts = ""

    if "oci_meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox_group:
        oci_meta_llama_4_scout_response = oci_meta_llama_4_scout_response
        oci_meta_llama_4_scout_referenced_contexts = ""
        if include_citation:
            oci_meta_llama_4_scout_response, oci_meta_llama_4_scout_referenced_contexts = extract_citation(
                oci_meta_llama_4_scout_response)
        if llm_evaluation_checkbox:
            oci_meta_llama_4_scout_evaluation = oci_meta_llama_4_scout_evaluation
        else:
            oci_meta_llama_4_scout_evaluation = ""
    else:
        oci_meta_llama_4_scout_response = ""
        oci_meta_llama_4_scout_evaluation = ""
        oci_meta_llama_4_scout_referenced_contexts = ""

    if "openai/gpt-4o" in llm_answer_checkbox_group:
        openai_gpt4o_response = openai_gpt4o_response
        openai_gpt4o_referenced_contexts = ""
        if include_citation:
            openai_gpt4o_response, openai_gpt4o_referenced_contexts = extract_citation(openai_gpt4o_response)
        if llm_evaluation_checkbox:
            openai_gpt4o_evaluation = openai_gpt4o_evaluation
        else:
            openai_gpt4o_evaluation = ""
    else:
        openai_gpt4o_response = ""
        openai_gpt4o_evaluation = ""
        openai_gpt4o_referenced_contexts = ""

    if "azure_openai/gpt-4o" in llm_answer_checkbox_group:
        azure_openai_gpt4o_response = azure_openai_gpt4o_response
        azure_openai_gpt4o_referenced_contexts = ""
        if include_citation:
            azure_openai_gpt4o_response, azure_openai_gpt4o_referenced_contexts = extract_citation(
                azure_openai_gpt4o_response)
        if llm_evaluation_checkbox:
            azure_openai_gpt4o_evaluation = azure_openai_gpt4o_evaluation
        else:
            azure_openai_gpt4o_evaluation = ""
    else:
        azure_openai_gpt4o_response = ""
        azure_openai_gpt4o_evaluation = ""
        azure_openai_gpt4o_referenced_contexts = ""

    df3 = pd.DataFrame(
        {
            'LLM モデル':
                [
                    "oci_openai/o3",
                    "oci_xai/grok-4",
                    "oci_cohere/command-a",
                    "oci_meta/llama-4-scout-17b-16e-instruct",
                    "openai/gpt-4o",
                    "azure_openai/gpt-4o",
                ],
            'LLM メッセージ': [
                oci_openai_o3_response,
                oci_xai_grok_4_response,
                oci_cohere_command_a_response,
                oci_meta_llama_4_scout_response,
                openai_gpt4o_response,
                azure_openai_gpt4o_response,
            ],
            'Vision 回答': [
                remove_base64_images_from_text(oci_openai_o3_image_response),
                "",  # xai/grok-4 (Vision機能なし)
                "",  # cohere/command-a (Vision機能なし)
                remove_base64_images_from_text(oci_meta_llama_4_scout_image_response),
                remove_base64_images_from_text(openai_gpt4o_image_response),
                remove_base64_images_from_text(azure_openai_gpt4o_image_response),
            ],
            '引用 Contexts': [
                oci_openai_o3_referenced_contexts,
                oci_xai_grok_4_referenced_contexts,
                oci_cohere_command_a_referenced_contexts,
                oci_meta_llama_4_scout_referenced_contexts,
                openai_gpt4o_referenced_contexts,
                azure_openai_gpt4o_referenced_contexts,
            ],
            'LLM 評価結果': [
                oci_openai_o3_evaluation,
                oci_xai_grok_4_evaluation,
                oci_cohere_command_a_evaluation,
                oci_meta_llama_4_scout_evaluation,
                openai_gpt4o_evaluation,
                azure_openai_gpt4o_evaluation,
            ]
        }
    )

    # ファイルパスを定義
    filepath = '/tmp/query_result.xlsx'

    # ExcelWriterを使用して複数のDataFrameを異なるシートに書き込み
    with pd.ExcelWriter(filepath) as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
        df3.to_excel(writer, sheet_name='Sheet3', index=False)

    print(f"Excelファイルが {filepath} に保存されました")
    return gr.DownloadButton(value=filepath, visible=True)
