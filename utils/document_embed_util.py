"""
ドキュメント埋め込み処理ユーティリティモジュール

このモジュールは、ドキュメントの埋め込みベクトル生成と保存を行うための関数を提供します。
unstructured形式のドキュメント処理に特化しています。
"""

import json
import re
from typing import Tuple

import gradio as gr
import oracledb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from utils.embedding_util import generate_embedding_response, generate_image_embedding_response


def embed_save_document_by_unstructured(doc_id, chunks_by, chunks_max_size,
                                        chunks_overlap_size,
                                        chunks_split_by, chunks_split_by_custom,
                                        chunks_language, chunks_normalize,
                                        chunks_normalize_options,
                                        pool, default_collection_name,
                                        get_server_path_func):
    """
    unstructured形式のドキュメントに対して埋め込みベクトルを生成し、データベースに保存する
    
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
        
    Returns:
        Tuple[gr.Textbox, gr.Textbox, gr.Dataframe]: 
            - 実行されたSQL文
            - 空のテキストボックス
            - 空のデータフレーム
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

    # 画像ブロックを処理して{default_collection_name}_imageテーブルに保存
    server_path = get_server_path_func(doc_id)
    chunks_overlap = int(float(chunks_max_size) * (float(chunks_overlap_size) / 100))

    # .mdファイルの場合、コンテンツを取得して画像ブロックを処理
    if server_path.lower().endswith('.md'):
        loader = TextLoader(server_path)
        documents = loader.load()
        doc_data = "\n".join(doc.page_content for doc in documents)

        # 画像ブロックが含まれている場合は処理する
        if re.search(r'<!-- image_begin -->.*?<!-- image_end -->', doc_data, re.DOTALL):
            process_image_blocks(doc_id, doc_data, pool, default_collection_name,
                                 generate_embedding_response,
                                 chunk_size=chunks_max_size - chunks_overlap,
                                 chunk_overlap=chunks_overlap)

    output_sql = ""
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            # 既存の埋め込みデータを取得
            select_sql = f"""
SELECT doc_id, embed_id, embed_data FROM {default_collection_name}_embedding  WHERE doc_id = :doc_id
"""
            cursor.execute(select_sql, doc_id=doc_id)
            records = cursor.fetchall()
            embed_datas = [record[2] for record in records]

            # 埋め込みベクトルを生成
            embed_vectors = generate_embedding_response(embed_datas)

            # 埋め込みベクトルを更新するSQL
            update_sql = f"""
UPDATE {default_collection_name}_embedding
SET embed_vector = :embed_vector
WHERE doc_id = :doc_id and embed_id = :embed_id
"""

            # SQLサイズ設定
            cursor.setinputsizes(embed_vector=oracledb.DB_TYPE_VECTOR)

            # 出力用SQL文を生成（参照用）
            output_sql += update_sql.replace(':doc_id', "'" + str(doc_id) + "'"
                                             ).replace(':embed_id', "'" + str('...') + "'"
                                                       ).replace(':embed_vector', "'" + str('...') + "'").strip() + ";"
            print(f"{output_sql=}")

            # バッチで埋め込みベクトルを更新
            cursor.executemany(update_sql,
                               [{'doc_id': record[0], 'embed_id': record[1], 'embed_vector': embed_vector}
                                for record, embed_vector in zip(records, embed_vectors)])
            conn.commit()

    return (
        gr.Textbox(output_sql),
        gr.Textbox(),
        gr.Dataframe()
    )


def process_image_blocks(doc_id: str, doc_data: str, pool, default_collection_name: str,
                         generate_embedding_response_func, chunk_size: int = None, chunk_overlap: int = None):
    """
    画像ブロックを処理してデータベースに保存し、text_splitterを使用してデータを分割する
    
    Args:
        doc_id (str): ドキュメントID
        doc_data (str): ドキュメントデータ
        chunk_size (int, optional): チャンクサイズ
        chunk_overlap (int, optional): チャンクオーバーラップ
        pool: データベース接続プール
        default_collection_name (str): デフォルトコレクション名
        generate_embedding_response_func: 埋め込み生成関数
    """
    # 画像ブロックのパターンを検索
    image_blocks = re.findall(r'<!-- image_begin -->(.*?)<!-- image_end -->', doc_data, re.DOTALL)

    if not image_blocks:
        return

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            # 既存の画像データを削除
            delete_image_sql = f"""
DELETE FROM {default_collection_name}_image WHERE doc_id = :doc_id
"""
            cursor.execute(delete_image_sql, [doc_id])

            # 既存の画像embedding データを削除
            delete_image_embedding_sql = f"""
DELETE FROM {default_collection_name}_image_embedding WHERE doc_id = :doc_id
"""
            cursor.execute(delete_image_embedding_sql, [doc_id])

            # 画像データを挿入するSQL
            insert_image_sql = f"""
INSERT INTO {default_collection_name}_image (
    doc_id,
    img_id,
    text_data,
    vlm_data,
    base64_data
) VALUES (:doc_id, :img_id, :text_data, :vlm_data, :base64_data)
"""

            img_id = 1
            for image_block in image_blocks:
                # OCRコンテンツを抽出
                text_data_match = re.search(r'<!-- image_ocr_content_begin -->(.*?)<!-- image_ocr_content_end -->',
                                            image_block, re.DOTALL)
                text_data = text_data_match.group(1).strip() if text_data_match else ""

                # VLM説明を抽出
                vlm_data_match = re.search(
                    r'<!-- image_vlm_description_begin -->(.*?)<!-- image_vlm_description_end -->', image_block,
                    re.DOTALL)
                vlm_data = vlm_data_match.group(1).strip() if vlm_data_match else ""

                # Base64データを抽出
                base64_data_match = re.search(r'<!-- image_base64_begin -->(.*?)<!-- image_base64_end -->', image_block,
                                              re.DOTALL)
                base64_data = base64_data_match.group(1).strip() if base64_data_match else ""

                # base64_dataから前後のHTMLコメント記号を除去
                if base64_data:
                    # 前後の <!-- と --> を除去
                    base64_data = re.sub(r'^<!--\s*', '', base64_data)
                    base64_data = re.sub(r'\s*-->$', '', base64_data)
                    base64_data = base64_data.strip()

                # データベースに挿入
                cursor.execute(insert_image_sql, {
                    'doc_id': doc_id,
                    'img_id': img_id,
                    'text_data': text_data,
                    'vlm_data': vlm_data,
                    'base64_data': base64_data
                })

                print(f"画像データを保存しました: doc_id={doc_id}, img_id={img_id}")
                img_id += 1

            conn.commit()
            print(f"合計 {len(image_blocks)} 個の画像ブロックを処理しました")

            # 画像データのsplit処理を実行
            _process_image_data_splitting(doc_id, cursor, default_collection_name,
                                          generate_embedding_response_func, chunk_size, chunk_overlap)
            conn.commit()


def _process_image_data_splitting(doc_id: str, cursor, default_collection_name: str,
                                  generate_embedding_response_func, chunk_size: int = None, chunk_overlap: int = None):
    """
    画像テーブルのtext_dataとvlm_dataを個別にtext_splitterで分割し、image_embeddingテーブルに保存する
    
    Args:
        doc_id (str): ドキュメントID
        cursor: データベースカーソル
        chunk_size (int, optional): チャンクサイズ
        chunk_overlap (int, optional): チャンクオーバーラップ
        default_collection_name (str): デフォルトコレクション名
        generate_embedding_response_func: 埋め込み生成関数
    """
    # 画像データを取得
    select_image_sql = f"""
SELECT img_id, text_data, vlm_data, base64_data FROM {default_collection_name}_image
WHERE doc_id = :doc_id ORDER BY img_id
"""
    cursor.execute(select_image_sql, [doc_id])
    image_records = cursor.fetchall()

    if not image_records:
        print(f"画像データが見つかりません: doc_id={doc_id}")
        return

    # text_splitterを初期化
    if chunk_size is not None and chunk_overlap is not None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        # デフォルト設定を使用
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,  # デフォルトのchunk_size
            chunk_overlap=25  # デフォルトのoverlap（約10%）
        )

    # 画像embedding データを挿入するSQL
    insert_embedding_sql = f"""
INSERT INTO {default_collection_name}_image_embedding (
    doc_id,
    embed_id,
    embed_data,
    embed_vector,
    cmetadata,
    img_id
) VALUES (:doc_id, :embed_id, :embed_data, :embed_vector, :cmetadata, :img_id)
"""

    # text_data、vlm_data、base64_dataで独立したembed_idを使用
    text_embed_id = 10001
    vlm_embed_id = 30001
    base64_embed_id = 50001
    total_chunks = 0

    for img_id, text_data, vlm_data, base64_data in image_records:
        # LOBオブジェクトを文字列に変換
        text_data_str = text_data.read() if text_data else ""
        vlm_data_str = vlm_data.read() if vlm_data else ""

        # text_dataを個別に処理
        if text_data_str and text_data_str.strip():
            text_chunks = text_splitter.split_text(text_data_str)

            if text_chunks:
                # text_dataのchunkに対してembeddingを生成して保存
                text_chunk_texts = [chunk for chunk in text_chunks if chunk.strip()]
                if text_chunk_texts:
                    text_embed_vectors = generate_embedding_response_func(text_chunk_texts)

                    for i, (chunk_text, embed_vector) in enumerate(zip(text_chunk_texts, text_embed_vectors)):
                        # text_data用のメタデータを作成
                        cmetadata = json.dumps({
                            "img_id": img_id,
                            "chunk_index": i,
                            "source": "image_processing",
                            "data_type": "text_data",
                            "original_img_id": img_id
                        }, ensure_ascii=False)

                        cursor.setinputsizes(embed_vector=oracledb.DB_TYPE_VECTOR)
                        cursor.execute(insert_embedding_sql, {
                            'doc_id': doc_id,
                            'embed_id': text_embed_id,
                            'embed_data': chunk_text,
                            'embed_vector': embed_vector,
                            'cmetadata': cmetadata,
                            'img_id': img_id
                        })

                        print(
                            f"画像text_data embeddingデータを保存しました: doc_id={doc_id}, img_id={img_id}, embed_id={text_embed_id}")
                        text_embed_id += 1
                        total_chunks += 1

        # vlm_dataを個別に処理
        if vlm_data_str and vlm_data_str.strip():
            vlm_chunks = text_splitter.split_text(vlm_data_str)

            if vlm_chunks:
                # vlm_dataのchunkに対してembeddingを生成して保存
                vlm_chunk_texts = [chunk for chunk in vlm_chunks if chunk.strip()]
                if vlm_chunk_texts:
                    vlm_embed_vectors = generate_embedding_response_func(vlm_chunk_texts)

                    for i, (chunk_text, embed_vector) in enumerate(zip(vlm_chunk_texts, vlm_embed_vectors)):
                        # vlm_data用のメタデータを作成
                        cmetadata = json.dumps({
                            "img_id": img_id,
                            "chunk_index": i,
                            "source": "image_processing",
                            "data_type": "vlm_data",
                            "original_img_id": img_id
                        }, ensure_ascii=False)

                        cursor.setinputsizes(embed_vector=oracledb.DB_TYPE_VECTOR)
                        cursor.execute(insert_embedding_sql, {
                            'doc_id': doc_id,
                            'embed_id': vlm_embed_id,
                            'embed_data': chunk_text,
                            'embed_vector': embed_vector,
                            'cmetadata': cmetadata,
                            'img_id': img_id
                        })

                        print(
                            f"画像vlm_data embeddingデータを保存しました: doc_id={doc_id}, img_id={img_id}, embed_id={vlm_embed_id}")
                        vlm_embed_id += 1
                        total_chunks += 1

        # base64_dataを処理（チャンク分割は行わず、直接画像embeddingを生成）
        if base64_data:
            # LOBオブジェクトを文字列に変換
            base64_data_str = base64_data.read() if hasattr(base64_data, 'read') else str(base64_data)

            if base64_data_str and base64_data_str.strip():
                # base64データに対して画像embeddingを生成
                try:
                    # base64データがdata URIフォーマットでない場合は、適切なフォーマットに変換
                    formatted_base64_data = base64_data_str.strip()
                    if not formatted_base64_data.startswith('data:'):
                        # base64データの先頭文字から画像形式を推測
                        image_format = "jpeg"  # デフォルト
                        if formatted_base64_data.startswith('/9j/'):
                            image_format = "jpeg"
                        elif formatted_base64_data.startswith('iVBORw0KGgo'):
                            image_format = "png"
                        elif formatted_base64_data.startswith('R0lGODlh'):
                            image_format = "gif"
                        elif formatted_base64_data.startswith('UklGR'):
                            image_format = "webp"

                        formatted_base64_data = f"data:image/{image_format};base64,{formatted_base64_data}"

                    print(f"base64データフォーマット確認: {formatted_base64_data[:50]}...")
                    base64_embed_vectors = generate_image_embedding_response([formatted_base64_data])

                    if base64_embed_vectors:
                        # base64_data用のメタデータを作成
                        cmetadata = json.dumps({
                            "img_id": img_id,
                            "chunk_index": 0,  # base64データは分割しないため常に0
                            "source": "image_processing",
                            "data_type": "base64_data",
                            "original_img_id": img_id
                        }, ensure_ascii=False)

                        cursor.setinputsizes(embed_vector=oracledb.DB_TYPE_VECTOR)
                        cursor.execute(insert_embedding_sql, {
                            'doc_id': doc_id,
                            'embed_id': base64_embed_id,
                            'embed_data': "これは画像です",  # 固定値
                            'embed_vector': base64_embed_vectors[0],
                            'cmetadata': cmetadata,
                            'img_id': img_id
                        })

                        print(
                            f"画像base64_data embeddingデータを保存しました: doc_id={doc_id}, img_id={img_id}, embed_id={base64_embed_id}")
                        base64_embed_id += 1
                        total_chunks += 1

                except Exception as e:
                    print(f"base64データの画像embedding生成でエラーが発生しました: {e}")

        # 全てのデータが空の場合の警告
        base64_data_check = ""
        if base64_data:
            base64_data_check = base64_data.read() if hasattr(base64_data, 'read') else str(base64_data)

        if (not text_data_str or not text_data_str.strip()) and (not vlm_data_str or not vlm_data_str.strip()) and (
                not base64_data_check or not base64_data_check.strip()):
            print(f"画像 {img_id} にtext_data、vlm_data、base64_dataの全てが空です")

    print(f"画像データの分割処理が完了しました: 合計 {total_chunks} 個のchunkを生成")
