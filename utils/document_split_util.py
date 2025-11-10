"""
ドキュメント分割処理ユーティリティモジュール

このモジュールは、ドキュメントの分割、画像ブロック処理、チャンク管理を行うための関数を提供します。
unstructured形式のドキュメント処理とembedding生成に特化しています。
"""

import chardet
import re
from typing import List, Dict, Any, Tuple

import gradio as gr
import oracledb
import pandas as pd
from langchain_community.document_loaders import TextLoader
from unstructured.partition.auto import partition

from utils.chunk_util import RecursiveCharacterTextSplitter
from utils.common_util import get_dict_value


def reset_document_chunks_result_dataframe():
    """
    ドキュメントチャンク結果データフレームをリセットする
    
    Returns:
        gr.Dataframe: 空のデータフレーム
    """
    return (
        gr.Dataframe(value=None, row_count=(1, "fixed"))
    )


def reset_document_chunks_result_detail():
    """
    ドキュメントチャンク結果詳細をリセットする
    
    Returns:
        Tuple[gr.Textbox, gr.Textbox]: 空のテキストボックス
    """
    return (
        gr.Textbox(value=""),
        gr.Textbox(value="")
    )


def process_text_chunks(unstructured_chunks: List[str]) -> List[Dict[str, Any]]:
    """
    unstructuredチャンクを処理してメタデータ付きチャンクリストに変換する
    
    Args:
        unstructured_chunks (List[str]): unstructuredで分割されたテキストチャンクのリスト
        
    Returns:
        List[Dict[str, Any]]: メタデータ付きチャンクのリスト
    """
    chunks = []
    chunk_id = 10001
    start_offset = 1

    for chunk in unstructured_chunks:
        chunk_length = len(chunk)
        if chunk_length == 0:
            continue
        chunks.append({
            'CHUNK_ID': chunk_id,
            'CHUNK_OFFSET': start_offset,
            'CHUNK_LENGTH': chunk_length,
            'CHUNK_DATA': chunk
        })

        # IDとオフセットを更新
        chunk_id += 1
        start_offset += chunk_length

    return chunks


def split_document_by_unstructured(doc_id, chunks_by, chunks_max_size,
                                   chunks_overlap_size,
                                   chunks_split_by, chunks_split_by_custom,
                                   chunks_language, chunks_normalize,
                                   chunks_normalize_options,
                                   pool, default_collection_name,
                                   get_server_path_func, generate_embedding_response_func):
    """
    unstructured形式のドキュメントを分割し、チャンクをデータベースに保存する

    Args:
        doc_id (str): ドキュメントID
        chunks_by: チャンク分割方法（未使用）
        chunks_max_size (int): チャンクの最大サイズ
        chunks_overlap_size (float): チャンクのオーバーラップサイズ（パーセンテージ）
        chunks_split_by: 分割方法（未使用）
        chunks_split_by_custom: カスタム分割方法（未使用）
        chunks_language: 言語設定（未使用）
        chunks_normalize: 正規化設定（未使用）
        chunks_normalize_options: 正規化オプション（未使用）
        pool: データベース接続プール
        default_collection_name (str): デフォルトコレクション名
        get_server_path_func: サーバーパス取得関数
        generate_embedding_response_func: 埋め込み生成関数

    Returns:
        Tuple[gr.Textbox, gr.Textbox, gr.Dataframe]:
            - 実行されたSQL文
            - チャンク数
            - チャンクデータフレーム
    """
    has_error = False
    if not doc_id:
        has_error = True
        gr.Warning("ドキュメントを選択してください")
    if has_error:
        return (
            gr.Textbox(value=""),
            gr.Textbox(value=""),
            gr.Dataframe(value=None, row_count=(1, "fixed"))
        )

    output_sql = ""
    server_path = get_server_path_func(doc_id)

    chunks_overlap = int(float(chunks_max_size) * (float(chunks_overlap_size) / 100))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunks_max_size - chunks_overlap,
        chunk_overlap=chunks_overlap
    )

    doc_data = ""
    # .mdと.txtファイルの場合
    if server_path.lower().endswith(('.md', '.txt')):
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
            doc_data = "\n".join(doc.page_content for doc in documents)
        except UnicodeDecodeError:
            # エラーが発生した場合は、unstructured使用
            elements = partition(filename=server_path, strategy='fast',
                                languages=["jpn", "eng", "chi_sim"],
                                extract_image_block_types=["Table"],
                                extract_image_block_to_payload=False,
                                skip_infer_table_types=["pdf", "jpg", "png", "heic", "doc", "docx"])

            # テーブルデータの処理（Claudeを使用）
            page_table_documents = {}
            prev_page_number = 0
            table_idx = 1

            for el in elements:
                page_number = el.metadata.page_number
                if prev_page_number != page_number:
                    prev_page_number = page_number
                    table_idx = 1
                if el.category == "Table":
                    table_id = f"page_{page_number}_table_{table_idx}"
                    print(f"{table_id=}")
                    print(f"{page_table_documents=}")
                    if page_table_documents:
                        page_table_document = get_dict_value(page_table_documents, table_id)
                        print(f"{page_table_document=}")
                        if page_table_document:
                            page_content = get_dict_value(page_table_document, "page_content")
                            table_to_markdown = get_dict_value(page_table_document, "table_to_markdown")
                            if page_content and table_to_markdown:
                                print(f"Before: {el.text=}")
                                el.text = page_content + "\n" + table_to_markdown + "\n"
                                print(f"After: {el.text=}")
                    table_idx += 1

            for element in elements:
                # element.textを文字列に変換してLOBオブジェクトのTypeErrorを回避
                element.text = str(element.text).replace('\x0b', '\n')
                element.text = str(element.text).replace('\x01', ' ')
            doc_data = " \n".join([str(element.text) for element in elements])

        # 画像ブロックが含まれている場合はOCRコンテンツを抽出
        if re.search(r'<!-- image_begin -->.*?<!-- image_end -->', doc_data, re.DOTALL):
            # すべてのOCRコンテンツブロックを抽出
            text_contexts = re.findall(r'<!-- image_ocr_content_begin -->(.*?)<!-- image_ocr_content_end -->', doc_data,
                                       re.DOTALL)
            doc_data = "\n".join(ocr.strip() for ocr in text_contexts if ocr.strip())
    else:
        # データベースからデータを取得できない場合は、ファイルを読み込む
        elements = partition(filename=server_path, strategy='fast',
                             languages=["jpn", "eng", "chi_sim"],
                             extract_image_block_types=["Table"],
                             extract_image_block_to_payload=False,
                             skip_infer_table_types=["pdf", "jpg", "png", "heic", "doc", "docx"])

        # テーブルデータの処理（Claudeを使用）
        page_table_documents = {}
        prev_page_number = 0
        table_idx = 1

        for el in elements:
            page_number = el.metadata.page_number
            if prev_page_number != page_number:
                prev_page_number = page_number
                table_idx = 1
            if el.category == "Table":
                table_id = f"page_{page_number}_table_{table_idx}"
                print(f"{table_id=}")
                print(f"{page_table_documents=}")
                if page_table_documents:
                    page_table_document = get_dict_value(page_table_documents, table_id)
                    print(f"{page_table_document=}")
                    if page_table_document:
                        page_content = get_dict_value(page_table_document, "page_content")
                        table_to_markdown = get_dict_value(page_table_document, "table_to_markdown")
                        if page_content and table_to_markdown:
                            print(f"Before: {el.text=}")
                            el.text = page_content + "\n" + table_to_markdown + "\n"
                            print(f"After: {el.text=}")
                table_idx += 1

        for element in elements:
            # element.textを文字列に変換してLOBオブジェクトのTypeErrorを回避
            element.text = str(element.text).replace('\x0b', '\n')
            element.text = str(element.text).replace('\x01', ' ')
        doc_data = " \n".join([str(element.text) for element in elements])

    unstructured_chunks = text_splitter.split_text(doc_data)
    chunks = process_text_chunks(unstructured_chunks)
    chunks_dataframe = pd.DataFrame(chunks)

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            delete_sql = f"""
-- Delete chunks
DELETE FROM {default_collection_name}_embedding WHERE doc_id = :doc_id """
            cursor.execute(delete_sql, [doc_id])
            output_sql += delete_sql.replace(':doc_id', "'" + str(doc_id) + "'").lstrip() + ";"

            save_chunks_sql = f"""
-- (Only for Reference) Insert chunks
INSERT INTO {default_collection_name}_embedding (
doc_id,
embed_id,
embed_data
)
VALUES (:doc_id, :embed_id, :embed_data) """
            output_sql += "\n" + save_chunks_sql.replace(':doc_id', "'" + str(doc_id) + "'"
                                                         ).replace(':embed_id', "'...'"
                                                                   ).replace(':embed_data', "'...'"
                                                                             ) + ";"
            print(f"{output_sql=}")
            # バッチ挿入用のデータを準備
            data_to_insert = [(doc_id, chunk['CHUNK_ID'], chunk['CHUNK_DATA']) for chunk in chunks]

            # バッチ挿入を実行
            cursor.executemany(save_chunks_sql, data_to_insert)
            conn.commit()

    return (
        gr.Textbox(value=output_sql),
        gr.Textbox(value=str(len(chunks_dataframe))),
        gr.Dataframe(value=chunks_dataframe, row_count=(len(chunks_dataframe), "fixed"))
    )


def on_select_split_document_chunks_result(evt: gr.SelectData, chunks_result_dataframe):
    """
    分割ドキュメントチャンク結果の選択イベントハンドラー

    Args:
        evt (gr.SelectData): 選択イベントデータ
        chunks_result_dataframe: チャンク結果データフレーム

    Returns:
        Tuple[gr.Textbox, gr.Textbox]: 選択されたチャンクのIDとデータ
    """
    if chunks_result_dataframe is None or len(chunks_result_dataframe) == 0:
        return gr.Textbox(value=""), gr.Textbox(value="")

    selected_row = evt.index[0]
    if selected_row < len(chunks_result_dataframe):
        chunk_id = chunks_result_dataframe.iloc[selected_row]['CHUNK_ID']
        chunk_data = chunks_result_dataframe.iloc[selected_row]['CHUNK_DATA']
        return gr.Textbox(value=str(chunk_id)), gr.Textbox(value=chunk_data)

    return gr.Textbox(value=""), gr.Textbox(value="")


def update_document_chunks_result_detail(doc_id, chunk_id, chunk_data,
                                         pool, default_collection_name,
                                         generate_embedding_response_func):
    """
    ドキュメントチャンク結果詳細を更新する

    Args:
        doc_id (str): ドキュメントID
        chunk_id (str): チャンクID
        chunk_data (str): チャンクデータ
        pool: データベース接続プール
        default_collection_name (str): デフォルトコレクション名
        generate_embedding_response_func: 埋め込み生成関数

    Returns:
        Tuple[gr.Textbox, gr.Textbox, gr.Textbox]:
            - 更新されたチャンクID
            - 空のテキストボックス
            - 更新されたチャンクデータ
    """
    chunk_data = chunk_data.strip()

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            update_sql = f"""
UPDATE {default_collection_name}_embedding
SET embed_data = :embed_data, embed_vector = :embed_vector
WHERE doc_id = :doc_id and embed_id = :embed_id
"""
            embed_vector = generate_embedding_response_func([chunk_data])[0]
            cursor.setinputsizes(embed_vector=oracledb.DB_TYPE_VECTOR)
            cursor.execute(update_sql,
                           {'doc_id': doc_id, 'embed_id': chunk_id, 'embed_data': chunk_data,
                            'embed_vector': embed_vector})
            conn.commit()

    return (
        gr.Textbox(value=chunk_id),
        gr.Textbox(),
        gr.Textbox(value=chunk_data),
    )


def update_document_chunks_result_detail_with_validation(doc_id, df: pd.DataFrame, chunk_id, chunk_data,
                                                         pool, default_collection_name,
                                                         generate_embedding_response_func):
    """
    ドキュメントチャンク結果詳細を更新する（バリデーション付き）

    この関数はmain.pyからの包装関数で、入力検証とデータフレーム更新を含みます。

    Args:
        doc_id (str): ドキュメントID
        df (pd.DataFrame): チャンク結果データフレーム
        chunk_id (str): チャンクID
        chunk_data (str): チャンクデータ
        pool: データベース接続プール
        default_collection_name (str): デフォルトコレクション名
        generate_embedding_response_func: 埋め込み生成関数

    Returns:
        Tuple[gr.Dataframe, gr.Textbox, gr.Textbox]:
            - 更新されたデータフレーム
            - 空のテキストボックス
            - 更新されたチャンクデータ
    """
    print("in update_document_chunks_result_detail_with_validation() start...")
    has_error = False
    if not doc_id:
        has_error = True
        gr.Warning("ドキュメントを選択してください")
    if not chunk_data or chunk_data.strip() == "":
        has_error = True
        gr.Warning("CHUNK_DATAを入力してください")
    if has_error:
        return (
            gr.Dataframe(),
            gr.Textbox(),
            gr.Textbox(),
        )

    # 既存の関数を呼び出し
    result = update_document_chunks_result_detail(doc_id, chunk_id, chunk_data,
                                                  pool, default_collection_name, generate_embedding_response_func)

    # データフレームを更新
    updated_df = df.copy()
    mask = updated_df['CHUNK_ID'] == int(chunk_id)
    updated_df.loc[mask, 'CHUNK_DATA'] = chunk_data
    updated_df.loc[mask, 'CHUNK_LENGTH'] = len(chunk_data)

    return (
        gr.Dataframe(value=updated_df),
        gr.Textbox(),
        gr.Textbox(value=chunk_data),
    )
