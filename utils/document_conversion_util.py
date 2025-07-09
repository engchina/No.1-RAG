import json
import os
import pandas as pd
import gradio as gr
from markitdown import MarkItDown
from langchain_community.chat_models import ChatOCIGenAI
from utils.common_util import get_region


def convert_excel_to_text_document(file_path):
    """
    ExcelファイルをJSONライン形式のテキストドキュメントに変換する
    
    Args:
        file_path: アップロードされたファイルのパス
        
    Returns:
        tuple: (クリアされたファイル入力, 変換されたファイル)
    """
    has_error = False
    if not file_path:
        has_error = True
        gr.Warning("ファイルを選択してください")
    if has_error:
        return gr.File(value=None)

    output_file_path = file_path.name + '.txt'
    df = pd.read_excel(file_path.name)
    data = df.to_dict('records')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for row in data:
            # 各行を処理してTimestampを文字列に変換
            processed_row = {}
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    processed_row[key] = str(value)
                else:
                    processed_row[key] = value
            json_line = json.dumps(processed_row, ensure_ascii=False)
            f.write(json_line + ' <FIXED_DELIMITER>\n')
    return (
        gr.File(),
        gr.File(value=output_file_path)
    )


def convert_to_markdown_document(file_path, use_llm, llm_prompt):
    """
    ファイルをMarkdownドキュメントに変換する
    
    Args:
        file_path: アップロードされたファイルのパス
        use_llm: LLMを使用するかどうか
        llm_prompt: LLMに送信するプロンプト
        
    Returns:
        tuple: (クリアされたファイル入力, 変換されたファイル)
    """
    has_error = False
    if not file_path:
        has_error = True
        gr.Warning("ファイルを選択してください")
    if has_error:
        return gr.File(value=None)

    output_file_path = file_path.name + '.md'
    md = MarkItDown()

    file_extension = os.path.splitext(file_path.name)[-1].lower()
    if file_extension in ['.jpg', '.jpeg', '.png', '.ppt', '.pptx'] and use_llm:
        region = get_region()
        model_id = "meta.llama-3.2-90b-vision-instruct"
        if region == "us-chicago-1":
            model_id = "meta.llama-4-scout-17b-16e-instruct"
        client = ChatOCIGenAI(
            model_id="meta.llama-3.2-90b-vision-instruct",
            provider="meta",
            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
        )
        md = MarkItDown(llm_client=client, llm_model="meta.llama-3.2-90b-vision-instruct")
        result = md.convert(
            file_path.name,
            llm_prompt=llm_prompt,
        )
        print(f"{result.text_content=}")
    else:
        result = md.convert(
            file_path.name,
        )
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(result.text_content)
    return (
        gr.File(),
        gr.File(value=output_file_path)
    )
