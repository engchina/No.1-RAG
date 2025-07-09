"""
テキスト処理ユーティリティモジュール

このモジュールは、テキストの処理、引用の抽出、base64画像の削除など、
テキスト関連の操作を行うための関数を提供します。
"""

import re
import json
import pandas as pd


def remove_base64_images_from_text(text):
    """
    テキストからbase64画像情報を削除する

    Args:
        text: 処理対象のテキスト

    Returns:
        base64画像情報が削除されたテキスト
    """
    if not text or not isinstance(text, str):
        return text

    # base64画像のパターンを定義（data:image/で始まるもの）
    base64_image_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'

    # Markdown形式の画像（![alt](data:image/...)）を削除
    markdown_image_pattern = r'!\[[^\]]*\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)'

    # HTML形式の画像（<img src="data:image/...">）を削除
    html_image_pattern = r'<img[^>]*src=["\']data:image/[^;]+;base64,[A-Za-z0-9+/=]+["\'][^>]*>'

    # 各パターンを削除
    text = re.sub(markdown_image_pattern, '', text)
    text = re.sub(html_image_pattern, '', text)
    text = re.sub(base64_image_pattern, '', text)

    return text


def extract_citation(input_str):
    """
    テキストから引用部分を抽出する

    Args:
        input_str: 処理対象のテキスト

    Returns:
        tuple: (回答部分, 引用部分) のタプル。見つからない場合は (None, None)
    """
    # 2つの部分の内容をマッチング
    pattern = '^(.*?)\n\n---回答内で参照されているコンテキスト---\n\n(.*?)$'
    match = re.search(pattern, input_str, re.DOTALL)
    if match:
        part1 = match.group(1).strip()
        part2 = match.group(2).strip()
        return part1, part2
    else:
        return None, None


def extract_and_format(input_str, search_result_df):
    """
    テキストからJSONデータを抽出し、検索結果と照合してフォーマットする

    Args:
        input_str: 処理対象のテキスト
        search_result_df: 検索結果のDataFrame

    Returns:
        str: フォーマットされたテキスト
    """
    json_arrays = re.findall(r'\[\n.*?\{.*?}\n.*?]', input_str, flags=re.DOTALL)
    if not json_arrays:
        return (
                input_str +
                f"\n\n"
                f"---回答内で参照されているコンテキスト---"
                f"\n\n"
                f"回答にコンテキストが存在しないか、コンテキストの形式が正しくありません。"
        )

    extracted = []
    for json_str in json_arrays:
        input_str = input_str.replace(json_str, '')
        json_str = json_str.replace('\n', '').replace('\r', '')
        data = json.loads(json_str)

        for item in data:
            print(f"{item=}")
            if isinstance(item, dict):
                if "EMBED_ID" in item and "SOURCE" in item:
                    extracted.append({
                        "EMBED_ID": item["EMBED_ID"],
                        "SOURCE": item["SOURCE"]
                    })

    formatted = (
            input_str +
            f"\n\n"
            f"---回答内で参照されているコンテキスト---"
            f"\n\n"
    )
    formatted += "[\n"
    for item in extracted:
        content = "N/A"
        if item["EMBED_ID"] and isinstance(item["EMBED_ID"], int) and item["SOURCE"]:
            content = search_result_df.loc[
                (search_result_df["EMBED_ID"].astype(int) == int(item["EMBED_ID"])) &
                (search_result_df["SOURCE"] == item["SOURCE"]),
                "CONTENT"
            ].values
            if len(content) > 0:
                content = content[0]
                content = content.replace('"', '\'')
                content = content.replace('\n', ' ').replace('\r', ' ')
        formatted += (
            '    {{\n'
            '        "EMBED_ID": {},\n'
            '        "SOURCE": "{}",\n'
            '        "CONTENT": "{}"\n'
            '    }},\n'
        ).format(item["EMBED_ID"], item["SOURCE"], content)
    if extracted:
        formatted = formatted.rstrip(",\n") + "\n"
    formatted += "]"

    return formatted
