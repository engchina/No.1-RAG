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
        xai_grok_3_response,
        command_a_response,
        llama_4_maverick_response,
        llama_4_scout_response,
        llama_3_3_70b_response,
        llama_3_2_90b_vision_response,
        openai_gpt4o_response,
        openai_gpt4_response,
        azure_openai_gpt4o_response,
        azure_openai_gpt4_response,
        xai_grok_3_evaluation,
        command_a_evaluation,
        llama_4_maverick_evaluation,
        llama_4_scout_evaluation,
        llama_3_3_70b_evaluation,
        llama_3_2_90b_vision_evaluation,
        openai_gpt4o_evaluation,
        openai_gpt4_evaluation,
        azure_openai_gpt4o_evaluation,
        azure_openai_gpt4_evaluation,
        llama_4_maverick_image_response,
        llama_4_scout_image_response,
        llama_3_2_90b_vision_image_response,
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

    if "xai/grok-3" in llm_answer_checkbox_group:
        xai_grok_3_response = xai_grok_3_response
        xai_grok_3_referenced_contexts = ""
        if include_citation:
            xai_grok_3_response, xai_grok_3_referenced_contexts = extract_citation(xai_grok_3_response)
        if llm_evaluation_checkbox:
            xai_grok_3_evaluation = xai_grok_3_evaluation
        else:
            xai_grok_3_evaluation = ""
    else:
        xai_grok_3_response = ""
        xai_grok_3_evaluation = ""
        xai_grok_3_referenced_contexts = ""

    if "cohere/command-a" in llm_answer_checkbox_group:
        command_a_response = command_a_response
        command_a_referenced_contexts = ""
        if include_citation:
            command_a_response, command_a_referenced_contexts = extract_citation(command_a_response)
        if llm_evaluation_checkbox:
            command_a_evaluation = command_a_evaluation
        else:
            command_a_evaluation = ""
    else:
        command_a_response = ""
        command_a_evaluation = ""
        command_a_referenced_contexts = ""

    if "meta/llama-4-maverick-17b-128e-instruct-fp8" in llm_answer_checkbox_group:
        llama_4_maverick_response = llama_4_maverick_response
        llama_4_maverick_referenced_contexts = ""
        if include_citation:
            llama_4_maverick_response, llama_4_maverick_referenced_contexts = extract_citation(
                llama_4_maverick_response)
        if llm_evaluation_checkbox:
            llama_4_maverick_evaluation = llama_4_maverick_evaluation
        else:
            llama_4_maverick_evaluation = ""
    else:
        llama_4_maverick_response = ""
        llama_4_maverick_evaluation = ""
        llama_4_maverick_referenced_contexts = ""

    if "meta/llama-4-scout-17b-16e-instruct" in llm_answer_checkbox_group:
        llama_4_scout_response = llama_4_scout_response
        llama_4_scout_referenced_contexts = ""
        if include_citation:
            llama_4_scout_response, llama_4_scout_referenced_contexts = extract_citation(llama_4_scout_response)
        if llm_evaluation_checkbox:
            llama_4_scout_evaluation = llama_4_scout_evaluation
        else:
            llama_4_scout_evaluation = ""
    else:
        llama_4_scout_response = ""
        llama_4_scout_evaluation = ""
        llama_4_scout_referenced_contexts = ""

    if "meta/llama-3-3-70b" in llm_answer_checkbox_group:
        llama_3_3_70b_response = llama_3_3_70b_response
        llama_3_3_70b_referenced_contexts = ""
        if include_citation:
            llama_3_3_70b_response, llama_3_3_70b_referenced_contexts = extract_citation(llama_3_3_70b_response)
        if llm_evaluation_checkbox:
            llama_3_3_70b_evaluation = llama_3_3_70b_evaluation
        else:
            llama_3_3_70b_evaluation = ""
    else:
        llama_3_3_70b_response = ""
        llama_3_3_70b_evaluation = ""
        llama_3_3_70b_referenced_contexts = ""

    if "meta/llama-3-2-90b-vision" in llm_answer_checkbox_group:
        llama_3_2_90b_vision_response = llama_3_2_90b_vision_response
        llama_3_2_90b_vision_referenced_contexts = ""
        if include_citation:
            llama_3_2_90b_vision_response, llama_3_2_90b_vision_referenced_contexts = extract_citation(
                llama_3_2_90b_vision_response)
        if llm_evaluation_checkbox:
            llama_3_2_90b_vision_evaluation = llama_3_2_90b_vision_evaluation
        else:
            llama_3_2_90b_vision_evaluation = ""
    else:
        llama_3_2_90b_vision_response = ""
        llama_3_2_90b_vision_evaluation = ""
        llama_3_2_90b_vision_referenced_contexts = ""

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

    if "openai/gpt-4" in llm_answer_checkbox_group:
        openai_gpt4_response = openai_gpt4_response
        openai_gpt4_referenced_contexts = ""
        if include_citation:
            openai_gpt4_response, openai_gpt4_referenced_contexts = extract_citation(openai_gpt4_response)
        if llm_evaluation_checkbox:
            openai_gpt4_evaluation = openai_gpt4_evaluation
        else:
            openai_gpt4_evaluation = ""
    else:
        openai_gpt4_response = ""
        openai_gpt4_evaluation = ""
        openai_gpt4_referenced_contexts = ""

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

    if "azure_openai/gpt-4" in llm_answer_checkbox_group:
        azure_openai_gpt4_response = azure_openai_gpt4_response
        azure_openai_gpt4_referenced_contexts = ""
        if include_citation:
            azure_openai_gpt4_response, azure_openai_gpt4_referenced_contexts = extract_citation(
                azure_openai_gpt4_response)
        if llm_evaluation_checkbox:
            azure_openai_gpt4_evaluation = azure_openai_gpt4_evaluation
        else:
            azure_openai_gpt4_evaluation = ""
    else:
        azure_openai_gpt4_response = ""
        azure_openai_gpt4_evaluation = ""
        azure_openai_gpt4_referenced_contexts = ""

    df3 = pd.DataFrame(
        {
            'LLM モデル':
                [
                    "xai/grok-3",
                    "cohere/command-a",
                    "meta/llama-4-maverick-17b-128e-instruct-fp8",
                    "meta/llama-4-scout-17b-16e-instruct",
                    "meta/llama-3-3-70b",
                    "meta/llama-3-2-90b-vision",
                    "openai/gpt-4o",
                    "openai/gpt-4",
                    "azure_openai/gpt-4o",
                    "azure_openai/gpt-4"
                ],
            'LLM メッセージ': [
                xai_grok_3_response,
                command_a_response,
                llama_4_maverick_response,
                llama_4_scout_response,
                llama_3_3_70b_response,
                llama_3_2_90b_vision_response,
                openai_gpt4o_response,
                openai_gpt4_response,
                azure_openai_gpt4o_response,
                azure_openai_gpt4_response
            ],
            'Vision 回答': [
                "",  # xai/grok-3 (Vision機能なし)
                "",  # cohere/command-a (Vision機能なし)
                remove_base64_images_from_text(llama_4_maverick_image_response),
                remove_base64_images_from_text(llama_4_scout_image_response),
                "",  # meta/llama-3-3-70b (Vision機能なし)
                remove_base64_images_from_text(llama_3_2_90b_vision_image_response),
                remove_base64_images_from_text(openai_gpt4o_image_response),
                "",  # openai/gpt-4 (Vision機能なし)
                remove_base64_images_from_text(azure_openai_gpt4o_image_response),
                ""  # azure_openai/gpt-4 (Vision機能なし)
            ],
            '引用 Contexts': [
                xai_grok_3_referenced_contexts,
                command_a_referenced_contexts,
                llama_4_maverick_referenced_contexts,
                llama_4_scout_referenced_contexts,
                llama_3_3_70b_referenced_contexts,
                llama_3_2_90b_vision_referenced_contexts,
                openai_gpt4o_referenced_contexts,
                openai_gpt4_referenced_contexts,
                azure_openai_gpt4o_referenced_contexts,
                azure_openai_gpt4_referenced_contexts
            ],
            'LLM 評価結果': [
                xai_grok_3_evaluation,
                command_a_evaluation,
                llama_4_maverick_evaluation,
                llama_4_scout_evaluation,
                llama_3_3_70b_evaluation,
                llama_3_2_90b_vision_evaluation,
                openai_gpt4o_evaluation,
                openai_gpt4_evaluation,
                azure_openai_gpt4o_evaluation,
                azure_openai_gpt4_evaluation
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
