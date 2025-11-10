"""
ドキュメントローダーユーティリティモジュール

このモジュールは、ドキュメントファイルの読み込み、処理、データベースへの保存を行うための関数を提供します。
Markdown形式とunstructured形式の両方のドキュメント処理をサポートしています。
"""

import chardet
import json
import os
import shutil
from typing import Tuple

import gradio as gr
import oracledb
from langchain_community.document_loaders import TextLoader
from unstructured.partition.auto import partition


def load_document(file_path, server_directory, document_metadata,
                  pool, default_collection_name, generate_unique_id_func):
    """
    ドキュメントファイルを読み込み、処理してデータベースに保存する
    
    Args:
        file_path: アップロードされたファイルのパス
        server_directory (str): サーバー上の保存ディレクトリ
        document_metadata (str): ドキュメントのメタデータ（key1=value1,key2=value2形式）
        pool: データベース接続プール
        default_collection_name (str): デフォルトコレクション名
        generate_unique_id_func: ユニークID生成関数
        
    Returns:
        Tuple[gr.Textbox, gr.Textbox, gr.Textbox, gr.Textbox]: 
            - 実行されたSQL文
            - 生成されたドキュメントID
            - ページ数
            - ドキュメント内容
    """
    print("in load_document() start...")
    has_error = False

    # ファイルパスの検証
    if not file_path:
        has_error = True
        gr.Warning("ファイルを選択してください")

    # メタデータの検証
    if document_metadata:
        document_metadata = document_metadata.strip()
        if "=" not in document_metadata or '"' in document_metadata or "'" in document_metadata or '\\' in document_metadata:
            has_error = True
            gr.Warning("メタデータの形式が正しくありません。key1=value1,key2=value2,... の形式で入力してください。")
        else:
            metadatas = document_metadata.split(",")
            for metadata in metadatas:
                if "=" not in metadata:
                    has_error = True
                    gr.Warning(
                        "メタデータの形式が正しくありません。key1=value1,key2=value2,... の形式で入力してください。")
                    break

    if has_error:
        return gr.Textbox(value=""), gr.Textbox(value=""), gr.Textbox(value=""), gr.Textbox(value="")

    # サーバーディレクトリの作成
    if not os.path.exists(server_directory):
        os.makedirs(server_directory)

    # ファイル情報の取得
    doc_id = generate_unique_id_func("doc_")
    file_name = os.path.basename(file_path.name)
    file_extension = os.path.splitext(file_name)
    if isinstance(file_extension, tuple):
        file_extension = file_extension[1]

    # ファイルをサーバーディレクトリにコピー
    server_path = os.path.join(server_directory, f"{doc_id}_{file_name}")
    shutil.copy(file_path.name, server_path)

    # ドキュメント内容の処理
    collection_cmeta = {}
    # .mdと.txtファイルの場合
    if file_extension.lower() in [".md", ".txt"]:
        with open(server_path, "rb") as f:
            raw_data = f.read(4096)
            result = chardet.detect(raw_data)
            detected_encoding = result["encoding"]

        # 検出されたエンコーディングを検証し、失敗した場合はフォールバック
        encoding = None

        # まず検出されたエンコーディングを試す（置信度が0.5以上の場合）
        if detected_encoding and result["confidence"] >= 0.5:
            try:
                with open(server_path, 'r', encoding=detected_encoding) as test_file:
                    test_file.read(1000)  # 最初の1000文字を読んで検証
                encoding = detected_encoding
                print(f"Validated detected encoding: {encoding}")
            except (UnicodeDecodeError, UnicodeError, LookupError):
                print(f"Detected encoding {detected_encoding} failed validation, trying fallbacks...")

        # 検出されたエンコーディングが使えない場合、フォールバックを試す
        if encoding is None:
            # 中国語、日本語、一般的なエンコーディングを含む包括的なリスト
            fallback_encodings = [
                'utf-8', 'utf-8-sig',  # UTF-8系（最も一般的）
                'gbk', 'gb18030', 'gb2312',  # 中国語エンコーディング
                'cp932', 'shift_jis', 'euc-jp', 'iso-2022-jp',  # 日本語エンコーディング
                'latin1', 'cp1252',  # 西欧系
                'big5'  # 繁体字中国語
            ]

            for test_encoding in fallback_encodings:
                try:
                    with open(server_path, 'r', encoding=test_encoding) as test_file:
                        test_file.read(1000)
                    encoding = test_encoding
                    print(f"Fallback encoding detected: {encoding}")
                    break
                except (UnicodeDecodeError, UnicodeError, LookupError):
                    continue

            if encoding is None:
                encoding = 'utf-8'
                print(f"All encodings failed, using default: {encoding} with error handling")

        try:
            loader = TextLoader(server_path, encoding=encoding)
            documents = loader.load()
            original_contents = "".join(doc.page_content for doc in documents)
            pages_count = len(documents)
        except UnicodeDecodeError:
            # エラーが発生した場合は、unstructured使用
            # https://docs.unstructured.io/open-source/core-functionality/overview
            pages_count = 1
            elements = partition(filename=server_path, strategy='fast',
                                languages=["jpn", "eng", "chi_sim"],
                                extract_image_block_types=["Table"],
                                extract_image_block_to_payload=False,
                                skip_infer_table_types=["pdf", "jpg", "png", "heic", "doc", "docx"])
            # for el in elements:
            #     print(f"{el=}")
            original_contents = " \n".join(el.text.replace('\x0b', '\n').replace('\x01', ' ') for el in elements)
    else:
        # その他のファイル形式の処理（unstructured使用）
        # https://docs.unstructured.io/open-source/core-functionality/overview
        pages_count = 1
        elements = partition(filename=server_path, strategy='fast',
                             languages=["jpn", "eng", "chi_sim"],
                             extract_image_block_types=["Table"],
                             extract_image_block_to_payload=False,
                             skip_infer_table_types=["pdf", "jpg", "png", "heic", "doc", "docx"])
        # for el in elements:
        #     print(f"{el=}")
        original_contents = " \n".join(el.text.replace('\x0b', '\n').replace('\x01', ' ') for el in elements)
    print(f"{original_contents=}")

    # メタデータの設定
    collection_cmeta['file_name'] = file_name
    collection_cmeta['source'] = server_path
    collection_cmeta['server_path'] = server_path

    # カスタムメタデータの追加
    if document_metadata:
        metadatas = document_metadata.split(",")
        for metadata in metadatas:
            key, value = metadata.split("=")
            collection_cmeta[key] = value

    # データベースへの保存
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.setinputsizes(**{"data": oracledb.BLOB})
            load_document_sql = f"""
 -- (Only for Reference) Insert to table {default_collection_name}_collection
 INSERT INTO {default_collection_name}_collection(id, data, cmetadata)
 VALUES (:id, to_blob(:data), :cmetadata) """

            # 出力用SQL文の生成（参照用）
            output_sql_text = load_document_sql.replace(":id", "'" + str(doc_id) + "'")
            output_sql_text = output_sql_text.replace(":data", "'...'")
            output_sql_text = output_sql_text.replace(":cmetadata", "'" + json.dumps(collection_cmeta) + "'") + ";"

            # データベースに挿入
            cursor.execute(load_document_sql, {
                'id': doc_id,
                'data': original_contents,
                'cmetadata': json.dumps(collection_cmeta)
            })
            conn.commit()

    return (
        gr.Textbox(value=output_sql_text.strip()),
        gr.Textbox(value=doc_id),
        gr.Textbox(value=str(pages_count)),
        gr.Textbox(value=original_contents[:1000] + " ..." if len(original_contents) > 1000 else original_contents)
    )
