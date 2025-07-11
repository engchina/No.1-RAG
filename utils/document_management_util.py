"""
ドキュメント操作関連の関数を提供するモジュール

このモジュールには以下の機能が含まれています：
- ドキュメント検索 (search_document)
- ドキュメントチャット (chat_document)
- 引用追加 (append_citation)
- 画像処理ストリーミング (process_single_image_streaming, process_image_answers_streaming)
- クエリ結果挿入 (insert_query_result)
- ドキュメント削除 (delete_document)
"""

import os
import time
from itertools import combinations
from typing import List, Tuple

import gradio as gr
import pandas as pd
import requests
from oracledb import DatabaseError

from utils.common_util import get_region
from utils.rerank_util import rerank_documents_response


def get_doc_list(pool, default_collection_name: str) -> List[Tuple[str, str]]:
    """
    データベースからドキュメントリストを取得する
    
    Args:
        pool: データベース接続プール
        default_collection_name: コレクション名
        
    Returns:
        List[Tuple[str, str]]: ドキュメント名とIDのタプルのリスト
    """
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(f"""
SELECT
    JSON_VALUE(cmetadata, '$.file_name') name,
    id
FROM
    {default_collection_name}_collection
ORDER BY name """)
                return [(f"{row[0]}", row[1]) for row in cursor.fetchall()]
            except DatabaseError as de:
                return []


def get_server_path(pool, default_collection_name: str, doc_id: str) -> str:
    """
    ドキュメントIDからサーバーパスを取得する
    
    Args:
        pool: データベース接続プール
        default_collection_name: コレクション名
        doc_id: ドキュメントID
        
    Returns:
        str: サーバーパス
    """
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"""
SELECT JSON_VALUE(cmetadata, '$.server_path') AS server_path
FROM {default_collection_name}_collection
WHERE id = :doc_id """, doc_id=doc_id)
            return cursor.fetchone()[0]


def delete_document(pool, default_collection_name: str, server_directory, doc_ids):
    """
    指定されたドキュメントを削除する
    
    Args:
        pool: データベース接続プール
        default_collection_name: コレクション名
        server_directory: サーバーディレクトリ
        doc_ids: 削除するドキュメントIDのリスト
        
    Returns:
        tuple: Gradioコンポーネントのタプル
    """
    has_error = False
    if not server_directory:
        has_error = True
        gr.Warning("サーバー・ディレクトリを入力してください")
    print(f"{doc_ids=}")
    if not doc_ids or len(doc_ids) == 0 or (len(doc_ids) == 1 and doc_ids[0] == ''):
        has_error = True
        gr.Warning("ドキュメントを選択してください")
    if has_error:
        return (
            gr.Textbox(value=""),
            gr.Radio(),
            gr.CheckboxGroup()
        )

    output_sql = ""
    with pool.acquire() as conn, conn.cursor() as cursor:
        for doc_id in filter(bool, doc_ids):
            server_path = get_server_path(pool, default_collection_name, doc_id)
            if os.path.exists(server_path):
                os.remove(server_path)
                print(f"File {doc_id} deleted successfully")
            else:
                print(f"File {doc_id} not found")

            delete_embedding_sql = f"""
DELETE FROM {default_collection_name}_embedding
WHERE doc_id = :doc_id """
            delete_collection_sql = f"""
DELETE FROM {default_collection_name}_collection
WHERE id = :doc_id """
            delete_image_sql = f"""
DELETE FROM {default_collection_name}_image
WHERE doc_id = :doc_id """
            delete_image_embedding_sql = f"""
DELETE FROM {default_collection_name}_image_embedding
WHERE doc_id = :doc_id """

            output_sql += delete_embedding_sql.strip().replace(":doc_id", "'" + doc_id + "'") + ";\n"
            output_sql += delete_collection_sql.strip().replace(":doc_id", "'" + doc_id + "'") + ";\n"
            output_sql += delete_image_sql.strip().replace(":doc_id", "'" + doc_id + "'") + ";\n"
            output_sql += delete_image_embedding_sql.strip().replace(":doc_id", "'" + doc_id + "'") + ";\n"

            cursor.execute(delete_embedding_sql, doc_id=doc_id)
            cursor.execute(delete_collection_sql, doc_id=doc_id)
            cursor.execute(delete_image_sql, doc_id=doc_id)
            cursor.execute(delete_image_embedding_sql, doc_id=doc_id)

            conn.commit()

    doc_list = get_doc_list(pool, default_collection_name)
    return (
        gr.Textbox(value=output_sql),
        gr.Radio(doc_list),
        gr.CheckboxGroup(choices=doc_list, value=[]),
        gr.CheckboxGroup(choices=doc_list)
    )


def search_document(
        pool,
        default_collection_name: str,
        reranker_model_radio_input,
        reranker_top_k_slider_input,
        reranker_threshold_slider_input,
        threshold_value_slider_input,
        top_k_slider_input,
        doc_id_all_checkbox_input,
        doc_id_checkbox_group_input,
        text_search_checkbox_input,
        text_search_k_slider_input,
        document_metadata_text_input,
        query_text_input,
        sub_query1_text_input,
        sub_query2_text_input,
        sub_query3_text_input,
        partition_by_k_slider_input,
        answer_by_one_checkbox_input,
        extend_first_chunk_size_input,
        extend_around_chunk_size_input,
        use_image
):
    """
    類似度検索を使用して質問に関連する分割を取得する
    これは単純に「トップK」検索で、クエリとの埋め込み類似度に基づいてドキュメントを選択する

    Args:
        pool: データベース接続プール
        default_collection_name: コレクション名
        その他: 検索パラメータ

    Returns:
        tuple: 検索結果を含むGradioコンポーネントのタプル
    """
    # Vision 回答がオンの場合、特定のパラメータ値を強制使用
    if use_image:
        answer_by_one_checkbox_input = False
        extend_first_chunk_size_input = 0
        extend_around_chunk_size_input = 0
        print(
            f"Vision 回答モード: answer_by_one_checkbox={answer_by_one_checkbox_input}, extend_first_chunk_size={extend_first_chunk_size_input}, extend_around_chunk_size={extend_around_chunk_size_input}")

    has_error = False
    if not query_text_input:
        has_error = True
        # gr.Warning("クエリを入力してください")
    if not doc_id_all_checkbox_input and (not doc_id_checkbox_group_input or doc_id_checkbox_group_input == [""]):
        has_error = True
        gr.Warning("ドキュメントを選択してください")
    if document_metadata_text_input:
        document_metadata_text_input = document_metadata_text_input.strip()
        if "=" not in document_metadata_text_input or '"' in document_metadata_text_input or "'" in document_metadata_text_input or '\\' in document_metadata_text_input:
            has_error = True
            gr.Warning("メタデータの形式が正しくありません。key1=value1,key2=value2,... の形式で入力してください。")
        else:
            metadatas = document_metadata_text_input.split(",")
            for metadata in metadatas:
                if "=" not in metadata:
                    has_error = True
                    gr.Warning(
                        "メタデータの形式が正しくありません。key1=value1,key2=value2,... の形式で入力してください。")
                    break
    if has_error:
        return (
            gr.Textbox(value=""),
            gr.Markdown(
                "**検索結果数**: 0   |   **検索キーワード**: (0)[]",
                visible=True
            ),
            gr.Dataframe(
                value=None,
                row_count=(1, "fixed")
            )
        )

    def cut_lists(lists, limit=10):
        if not lists or len(lists) == 0:
            return lists
        if len(lists) > limit:
            # 文字列の長さでソート、長い文字列を優先的に削除
            sorted_indices = sorted(range(len(lists)), key=lambda i: len(lists[i]))
            lists = [lists[i] for i in sorted_indices[:limit]]
        return lists

    def generate_combinations(words_list):
        sampled_list = []
        n = len(words_list)
        threshold = 0.75
        if n < 3:
            sampled_list = words_list
            return sampled_list
        min_required = int(n * threshold)
        print(f"{n=}")
        print(f"{min_required=}")
        sampled_list += list(combinations(words_list, min_required))
        return sampled_list

    def process_lists(lists):
        contains_sql_list = []
        for lst in lists:
            if isinstance(lst, str):
                contains_sql_list.append(f"contains(de.embed_data, '{lst}') > 0")
            else:
                temp = " and ".join(str(item) for item in lst)
                if temp and temp not in contains_sql_list:
                    contains_sql_list.append(f"contains(de.embed_data, '{temp}') > 0")
        return ' OR '.join(contains_sql_list)

    def format_keywords(x):
        if len(x) > 0:
            formatted = '[' + ', '.join(x) + ']'
            return f"({len(x)}){formatted}"
        else:
            return "(0)[No Match]"

    def replace_newlines(text):
        if '\n\n' in text:
            text = text.replace('\n\n', '\n')
            return replace_newlines(text)
        else:
            return text

    region = get_region()

    query_text_input = query_text_input.strip()
    sub_query1_text_input = sub_query1_text_input.strip()
    sub_query2_text_input = sub_query2_text_input.strip()
    sub_query3_text_input = sub_query3_text_input.strip()

    # Use OracleAIVector
    unranked_docs = []
    threshold_value = threshold_value_slider_input
    top_k = top_k_slider_input
    doc_ids_str = "'" + "','".join([str(doc_id) for doc_id in doc_id_checkbox_group_input if doc_id]) + "'"
    print(f"{doc_ids_str=}")
    with_sql = """
               -- Select data
               WITH offsets AS
                        (SELECT level - (:extend_around_chunk_size / 2 + 1) AS offset
                         FROM dual
               CONNECT BY level <= (:extend_around_chunk_size + 1)
                   )
                        , selected_embed_ids AS
                        ( \
               """

    where_metadata_sql = ""
    if document_metadata_text_input:
        metadata_conditions = []
        metadatas = document_metadata_text_input.split(",")
        for i, metadata in enumerate(metadatas):
            if "=" not in metadata:
                continue
            key, value = metadata.split("=", 1)
            # 正しいJSONパス構文とパラメータバインディングを使用
            metadata_conditions.append(f"JSON_VALUE(dc.cmetadata, '$.\"{key}\"') = '{value}'")

        if metadata_conditions:
            where_metadata_sql = " AND (" + " AND ".join(metadata_conditions) + ") "
        print(f"{where_metadata_sql=}")

    # 注意：where_threshold_sqlとwhere_sqlは現在base_sql内で直接構築されています。use_imageに応じて異なるテーブル別名を使用する必要があるためです
    # Vision 回答がオンの場合、image_embeddingテーブルを使用、オフの場合はembeddingテーブルを使用
    if use_image:
        # Vision 回答がオンの場合、image_embeddingテーブルのみを使用
        base_sql = f"""
    SELECT ie.doc_id doc_id, ie.embed_id embed_id, VECTOR_DISTANCE(ie.embed_vector, (
        SELECT 
            TO_VECTOR(et.embed_vector) embed_vector
        FROM
            DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS(
                    :query_text,
                    JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-v4.0"}}')) t,
                    JSON_TABLE(t.column_value, '$[*]'
                    COLUMNS(
                        embed_id NUMBER PATH '$.embed_id',
                        embed_data VARCHAR2(4000) PATH '$.embed_data',
                        embed_vector CLOB PATH '$.embed_vector'
                    )
                )
            et), COSINE
        ) vector_distance
    FROM 
        {default_collection_name}_image_embedding ie, {default_collection_name}_collection dc
    WHERE 
        1 = 1
    AND 
        ie.doc_id = dc.id """ + ("""
    AND 
        ie.doc_id IN (
            SELECT TRIM(BOTH '''' FROM REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL)) AS doc_id
        FROM DUAL
        CONNECT BY REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL) IS NOT NULL
    ) """ if not doc_id_all_checkbox_input else "") + where_metadata_sql + f"""
    AND vector_distance(ie.embed_vector, (
        SELECT 
            TO_VECTOR(et.embed_vector) embed_vector
        FROM
            DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS(
                    :query_text,
                    JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-v4.0"}}')) t,
                    JSON_TABLE(t.column_value, '$[*]'
                    COLUMNS(
                        embed_id NUMBER PATH '$.embed_id',
                        embed_data VARCHAR2(4000) PATH '$.embed_data',
                        embed_vector CLOB PATH '$.embed_vector'
                    )
                )
            et), COSINE
        ) <= :threshold_value
    ORDER BY 
        vector_distance """
    else:
        # Vision 回答がオフの場合、embeddingテーブルのみを使用
        base_sql = f"""
    SELECT de.doc_id doc_id, de.embed_id embed_id, VECTOR_DISTANCE(de.embed_vector, (
        SELECT 
            TO_VECTOR(et.embed_vector) embed_vector
        FROM
            DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS(
                    :query_text,
                    JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-v4.0"}}')) t,
                    JSON_TABLE(t.column_value, '$[*]'
                    COLUMNS (
                        embed_id NUMBER PATH '$.embed_id',
                        embed_data VARCHAR2(4000) PATH '$.embed_data',
                        embed_vector CLOB PATH '$.embed_vector'
                    )
                )
            et), COSINE
        ) vector_distance
    FROM 
        {default_collection_name}_embedding de, {default_collection_name}_collection dc
    WHERE 
        1 = 1
    AND 
        de.doc_id = dc.id """ + ("""
    AND 
        de.doc_id IN (
            SELECT TRIM(BOTH '''' FROM REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL)) AS doc_id
            FROM DUAL
            CONNECT BY REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL) IS NOT NULL
        ) """ if not doc_id_all_checkbox_input else "") + where_metadata_sql + f"""
    AND 
        VECTOR_DISTANCE(de.embed_vector, (
            SELECT 
                TO_VECTOR(et.embed_vector) embed_vector
            FROM
                DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS(
                    :query_text,
                    JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-v4.0"}}')) t,
                    JSON_TABLE(t.column_value, '$[*]'
                    COLUMNS(
                        embed_id NUMBER PATH '$.embed_id',
                        embed_data VARCHAR2(4000) PATH '$.embed_data',
                        embed_vector CLOB PATH '$.embed_vector'
                    )
                )
            et), COSINE
        ) <= :threshold_value
    ORDER BY 
        vector_distance """
    base_sql += """
    FETCH FIRST :partition_by PARTITIONS BY doc_id, :top_k ROWS ONLY
    """ if partition_by_k_slider_input > 0 else """
    FETCH FIRST :top_k ROWS ONLY
    """
    select_sql = f"""
),
selected_embed_id_doc_ids AS
(
    SELECT DISTINCT s.embed_id + o.offset adjusted_embed_id, s.doc_id doc_id
    FROM selected_embed_ids s
    CROSS JOIN offsets o
),
selected_results AS
(
    SELECT s1.adjusted_embed_id adjusted_embed_id, s1.doc_id doc_id, NVL(s2.vector_distance, 999999.0) vector_distance
    FROM selected_embed_id_doc_ids s1
    LEFT JOIN selected_embed_ids s2
    ON s1.adjusted_embed_id = s2.embed_id AND s1.doc_id = s2.doc_id
),"""
    if use_image:
        # Vision 回答がオンの場合、image_embeddingテーブルを使用
        select_sql += f"""
aggregated_results AS
(
    SELECT 
        JSON_VALUE(dc.cmetadata, '$.file_name') name, ie.embed_id embed_id, ie.embed_data embed_data, ie.doc_id doc_id, MIN(s.vector_distance) vector_distance
    FROM 
        selected_results s, {default_collection_name}_image_embedding ie, {default_collection_name}_collection dc
    WHERE 
        s.adjusted_embed_id = ie.embed_id AND s.doc_id = dc.id and ie.doc_id = dc.id
    GROUP BY 
        ie.doc_id, name, ie.embed_id, ie.embed_data"""
    else:
        # 画像を使って回答がオフの場合、embeddingテーブルを使用
        select_sql += f"""
aggregated_results AS
(
    SELECT 
        JSON_VALUE(dc.cmetadata, '$.file_name') name, de.embed_id embed_id, de.embed_data embed_data, de.doc_id doc_id, MIN(s.vector_distance) vector_distance
    FROM 
        selected_results s, {default_collection_name}_embedding de, {default_collection_name}_collection dc
    WHERE 
        s.adjusted_embed_id = de.embed_id AND s.doc_id = dc.id and de.doc_id = dc.id
    GROUP BY 
        de.doc_id, name, de.embed_id, de.embed_data """

    select_sql += """
    ORDER BY
        vector_distance
)"""

    # use_image が true の場合、相邻 embed_id のデータ合併を行わない
    if use_image:
        select_sql += """
                      SELECT ar.name,
                             ar.embed_id,
                             ar.embed_data AS combined_embed_data,
                             ar.doc_id,
                             ar.vector_distance
                      FROM aggregated_results ar
                      ORDER BY ar.vector_distance """
    else:
        # use_image が false の場合、従来の相邻 embed_id データ合併ロジックを使用
        select_sql += """
,
ranked_data AS
(
    SELECT
        ar.embed_data,
        ar.doc_id,
        ar.embed_id,
        ar.vector_distance,
        ar.name,
        LAG(ar.embed_id) OVER (PARTITION BY ar.doc_id ORDER BY ar.embed_id) AS prev_embed_id,
        LEAD(ar.embed_id) OVER (PARTITION BY ar.doc_id ORDER BY ar.embed_id) AS next_embed_id
    FROM
        aggregated_results ar
),
grouped_data AS
(
    SELECT
        rd.embed_data,
        rd.doc_id,
        rd.embed_id,
        rd.vector_distance,
        rd.name,
        CASE
            WHEN rd.prev_embed_id IS NULL OR rd.embed_id - rd.prev_embed_id > 1 THEN 1
            ELSE 0
        END AS new_group
    FROM
        ranked_data rd
),
groups_marked AS
(
    SELECT
        gd.embed_data,
        gd.doc_id,
        gd.embed_id,
        gd.vector_distance,
        gd.name,
        SUM(gd.new_group) OVER (PARTITION BY gd.doc_id ORDER BY gd.embed_id) AS group_id
    FROM
        grouped_data gd
),
aggregated_data AS
(
    SELECT
        RTRIM(XMLCAST(XMLAGG(XMLELEMENT(e, ad.embed_data || ',') ORDER BY ad.embed_id) AS CLOB), ',') AS combined_embed_data,
        ad.doc_id,
        MIN(ad.embed_id) AS min_embed_id,
        MIN(ad.vector_distance) AS min_vector_distance,
        ad.name
    FROM
        groups_marked ad
    GROUP BY
        ad.doc_id, ad.group_id, ad.name
)
SELECT
    ad.name,
    ad.min_embed_id AS embed_id,
    ad.combined_embed_data,
    ad.doc_id,
    ad.min_vector_distance AS vector_distance
FROM
    aggregated_data ad
ORDER BY
    ad.min_vector_distance """

    query_texts = [":query_text"]
    if sub_query1_text_input:
        query_texts.append(":sub_query1")
    if sub_query2_text_input:
        query_texts.append(":sub_query2")
    if sub_query3_text_input:
        query_texts.append(":sub_query3")
    print(f"{query_texts=}")

    search_texts = []
    if text_search_checkbox_input:
        # Generate the combined SQL based on available query texts
        search_texts = requests.post(os.environ["GINZA_API_ENDPOINT"],
                                     json={'query_text': query_text_input, 'language': 'ja'}).json()
        search_text = ""
        if search_texts and len(search_texts) > 0:
            search_texts = cut_lists(search_texts, text_search_k_slider_input)
            generated_combinations = generate_combinations(search_texts)
            search_text = process_lists(generated_combinations)
            # search_texts = cut_lists(search_texts, text_search_k_slider_input)
            # search_text = process_lists(search_texts)
        if len(search_text) > 0:
            # where_sql += """
            #             AND (""" + search_text + """) """
            # 注意：ここのwhere_sqlは異なるテーブルクエリで使用されるため、テーブル別名をハードコードできません
            # この部分のロジックはfull_text_search_sqlで処理されます
            if use_image:
                # Vision 回答がオンの場合、image_embeddingテーブルのみを使用
                full_text_search_sql = f"""
    SELECT ie.doc_id doc_id, ie.embed_id embed_id, VECTOR_DISTANCE(ie.embed_vector, (
        SELECT 
            TO_VECTOR(et.embed_vector) embed_vector
        FROM
            DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS(
                    :query_text,
                    JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-v4.0"}}')) t,
                    JSON_TABLE(t.column_value, '$[*]'
                        COLUMNS (
                            embed_id NUMBER PATH '$.embed_id',
                            embed_data VARCHAR2(4000) PATH '$.embed_data',
                            embed_vector CLOB PATH '$.embed_vector'
                        )
                    )
                et), COSINE
            ) vector_distance
    FROM 
        {default_collection_name}_image_embedding ie, {default_collection_name}_collection dc
    WHERE 
        1 = 1
    AND 
        ie.doc_id = dc.id
    AND 
        CONTAINS(ie.embed_data, :search_texts, 1) > 0 """ + ("""
    AND 
        ie.doc_id IN (
            SELECT TRIM(BOTH '''' FROM REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL)) AS doc_id
            FROM DUAL
            CONNECT BY REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL) IS NOT NULL
        ) """ if not doc_id_all_checkbox_input else "") + where_metadata_sql + """
    ORDER BY 
        SCORE(1) DESC FETCH FIRST :top_k ROWS ONLY """
            else:
                # Vision 回答がオフの場合、embeddingテーブルのみを使用
                full_text_search_sql = f"""
    SELECT de.doc_id doc_id, de.embed_id embed_id, VECTOR_DISTANCE(de.embed_vector, (
        SELECT TO_VECTOR(et.embed_vector) embed_vector
        FROM
            DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS(
                    :query_text,
                    JSON('{{"provider": "ocigenai", "credential_name": "OCI_CRED", "url": "https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText", "model": "cohere.embed-v4.0"}}')) t,
                    JSON_TABLE(t.column_value, '$[*]'
                        COLUMNS(
                            embed_id NUMBER PATH '$.embed_id',
                            embed_data VARCHAR2(4000) PATH '$.embed_data',
                            embed_vector CLOB PATH '$.embed_vector'
                        )
                    )
                et), COSINE
            ) vector_distance
    FROM 
        {default_collection_name}_embedding de, {default_collection_name}_collection dc
    WHERE 
        1 = 1
    AND 
        de.doc_id = dc.id
    AND 
        CONTAINS(de.embed_data, :search_texts, 1) > 0 """ + ("""
    AND 
        de.doc_id IN (
            SELECT TRIM(BOTH '''' FROM REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL)) AS doc_id
            FROM DUAL
            CONNECT BY REGEXP_SUBSTR(:doc_ids, '''[^'']+''', 1, LEVEL) IS NOT NULL
        ) """ if not doc_id_all_checkbox_input else "") + where_metadata_sql + """
    ORDER BY 
        SCORE(1) DESC FETCH FIRST :top_k ROWS ONLY """
            complete_sql = (with_sql + """
    UNION
    """.join(
                f"({base_sql.replace(':query_text', one_query_text)})" for one_query_text in
                query_texts) + """
    UNION
    ( """
                            + full_text_search_sql + """
        ) """
                            + select_sql)
    else:
        print(f"{query_texts=}")
        complete_sql = with_sql + """
    UNION
    """.join(
            f"({base_sql.replace(':query_text', one_query_text)})" for one_query_text in
            query_texts) + select_sql
        print(f"{complete_sql=}")

    # Prepare parameters for SQL execution
    params = {
        "extend_around_chunk_size": extend_around_chunk_size_input,
        "query_text": query_text_input,
        "threshold_value": threshold_value,
        "top_k": top_k
    }
    if not doc_id_all_checkbox_input:
        params["doc_ids"] = doc_ids_str
    if partition_by_k_slider_input > 0:
        params["partition_by"] = partition_by_k_slider_input
    if text_search_checkbox_input and search_texts:
        params["search_texts"] = " ACCUM ".join(search_texts)

    # Add sub-query texts if they exist
    if sub_query1_text_input:
        params["sub_query1"] = sub_query1_text_input
    if sub_query2_text_input:
        params["sub_query2"] = sub_query2_text_input
    if sub_query3_text_input:
        params["sub_query3"] = sub_query3_text_input

    # For debugging: Print the final SQL command.
    # Assuming complete_sql is your SQL string with placeholders like :extend_around_chunk_size, :doc_ids, etc.
    query_sql_output = complete_sql

    # Manually replace placeholders with parameter values for debugging
    for key, value in params.items():
        if key == "doc_ids":
            value = "''" + "'',''".join([str(doc_id) for doc_id in doc_id_checkbox_group_input if doc_id]) + "''"
        placeholder = f":{key}"
        # For the purpose of display, ensure the value is properly quoted if it's a string
        display_value = f"'{value}'" if isinstance(value, str) else str(value)
        query_sql_output = query_sql_output.replace(placeholder, display_value)
    query_sql_output = query_sql_output.strip() + ";"

    # Now query_sql_output contains the SQL command with parameter values inserted
    print(f"\nQUERY_SQL_OUTPUT:\n{query_sql_output}")

    with (pool.acquire() as conn):
        with conn.cursor() as cursor:
            max_retries = 3
            retries = 0
            while retries < max_retries:
                try:
                    cursor.execute(complete_sql, params)
                    break
                except Exception as e:
                    print(f"Exception: {e}")
                    retries += 1
                    print(f"Error executing SQL query: {e}. Retrying ({retries}/{max_retries})...")
                    time.sleep(10 * retries)
                    if retries == max_retries:
                        gr.Warning("データベース処理中にエラーが発生しました。しばらくしてから再度お試しください。")
                        return (
                            gr.Textbox(value=""),
                            gr.Markdown(
                                "**検索結果数**: 0   |   **検索キーワード**: (0)[]",
                                visible=True
                            ),
                            gr.Dataframe(
                                value=None,
                                row_count=(1, "fixed")
                            )
                        )
            for row in cursor:
                # print(f"row: {row}")
                # use_image が true の場合、combined_embed_data は文字列なので .read() を呼ばない
                if use_image:
                    unranked_docs.append([row[0], row[1], row[2], row[3], row[4]])
                else:
                    unranked_docs.append([row[0], row[1], row[2].read(), row[3], row[4]])

            if len(unranked_docs) == 0:
                docs_dataframe = pd.DataFrame(
                    columns=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"])

                return (
                    gr.Textbox(value=query_sql_output.strip()),
                    gr.Markdown(
                        "**検索結果数**: " + str(len(docs_dataframe)) + "   |   **検索キーワード**: (" + str(
                            len(search_texts)) + ")[" + ', '.join(search_texts) + "]",
                        visible=True),
                    gr.Dataframe(
                        value=docs_dataframe,
                        wrap=True,
                        headers=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"],
                        column_widths=["4%", "50%", "6%", "8%", "6%", "8%"],
                        row_count=(1, "fixed"),
                    ),
                )

            # ToDo: In case of error
            if 'cohere/rerank' in reranker_model_radio_input:
                unranked = []
                for doc in unranked_docs:
                    unranked.append(doc[2])
                ranked_results = rerank_documents_response(query_text_input, unranked, reranker_model_radio_input)
                ranked_scores = [0.0] * len(unranked_docs)
                for result in ranked_results:
                    ranked_scores[result['index']] = result['relevance_score']
                docs_data = [{'CONTENT': doc[2],
                              'EMBED_ID': doc[1],
                              'SOURCE': str(doc[3]) + ":" + doc[0],
                              'DISTANCE': '-' if str(doc[4]) == '999999.0' else str(doc[4]),
                              'SCORE': ce_score} for doc, ce_score in zip(unranked_docs, ranked_scores)]
                docs_dataframe = pd.DataFrame(docs_data)
                docs_dataframe = docs_dataframe[
                    docs_dataframe['SCORE'] >= float(reranker_threshold_slider_input)
                    ].sort_values(by='SCORE', ascending=False).head(
                    reranker_top_k_slider_input)
            else:
                docs_data = [{'CONTENT': doc[2],
                              'EMBED_ID': doc[1],
                              'SOURCE': str(doc[3]) + ":" + doc[0],
                              'DISTANCE': '-' if str(doc[4]) == '999999.0' else str(doc[4]),
                              'SCORE': '-'} for doc in unranked_docs]
                docs_dataframe = pd.DataFrame(docs_data)

            print(f"{extend_first_chunk_size_input=}")
            if extend_first_chunk_size_input > 0 and len(docs_dataframe) > 0:
                filtered_doc_ids = "'" + "','".join(
                    [source.split(':')[0] for source in docs_dataframe['SOURCE'].tolist()]) + "'"
                print(f"{filtered_doc_ids=}")
                # 画像を使って回答の設定に応じて適切なテーブルを選択
                if use_image:
                    # 画像を使って回答がオンの場合、image_embeddingテーブルを使用
                    select_extend_first_chunk_sql = f"""
SELECT
    JSON_VALUE(dc.cmetadata, '$.file_name') name,
    MIN(ie.embed_id) embed_id,
    RTRIM(XMLCAST(XMLAGG(XMLELEMENT(e, ie.embed_data || ',') ORDER BY ie.embed_id) AS CLOB), ',') AS embed_data,
    ie.doc_id doc_id,
    '999999.0' vector_distance
FROM
    {default_collection_name}_image_embedding ie, {default_collection_name}_collection dc
WHERE
    ie.doc_id = dc.id AND
    ie.doc_id IN (:filtered_doc_ids) AND
    ie.embed_id <= :extend_first_chunk_size
GROUP BY
    ie.doc_id, name
ORDER BY 
    ie.doc_id """
                else:
                    # 画像を使って回答がオフの場合、embeddingテーブルを使用
                    select_extend_first_chunk_sql = f"""
SELECT
    JSON_VALUE(dc.cmetadata, '$.file_name') name,
    MIN(de.embed_id) embed_id,
    RTRIM(XMLCAST(XMLAGG(XMLELEMENT(e, de.embed_data || ',') ORDER BY de.embed_id) AS CLOB), ',') AS embed_data,
    de.doc_id doc_id,
    '999999.0' vector_distance
FROM
    {default_collection_name}_embedding de, {default_collection_name}_collection dc
WHERE
    de.doc_id = dc.id AND
    de.doc_id IN (:filtered_doc_ids) AND
    de.embed_id <= :extend_first_chunk_size
GROUP BY
    de.doc_id, name
ORDER BY 
    de.doc_id """
                select_extend_first_chunk_sql = (select_extend_first_chunk_sql
                                                 .replace(':filtered_doc_ids', filtered_doc_ids)
                                                 .replace(':extend_first_chunk_size',
                                                          str(extend_first_chunk_size_input)))
                print(f"{select_extend_first_chunk_sql=}")
                query_sql_output += "\n" + select_extend_first_chunk_sql.strip()
                cursor.execute(select_extend_first_chunk_sql)
                first_chunks_df = pd.DataFrame(columns=docs_dataframe.columns)
                for row in cursor:
                    # use_image が true の場合、embed_data は文字列なので .read() を呼ばない
                    content_data = row[2] if use_image else row[2].read()
                    new_data = pd.DataFrame(
                        {'CONTENT': content_data, 'EMBED_ID': row[1], 'SOURCE': str(row[3]) + ":" + row[0],
                         'DISTANCE': '-', 'SCORE': '-'},
                        index=[2])
                    first_chunks_df = pd.concat([new_data, first_chunks_df], ignore_index=True)
                print(f"{first_chunks_df=}")

                # 更新されたデータを格納するための空のDataFrameを作成
                updated_df = pd.DataFrame(columns=docs_dataframe.columns)

                # 各SOURCEの初期挿入位置を記録
                insert_positions = {}

                # 元データの各行を走査
                for index, row in docs_dataframe.iterrows():
                    source = row['SOURCE']

                    # 現在のSOURCEがまだ初期挿入位置を記録していない場合、現在の位置で初期化
                    if source not in insert_positions:
                        insert_positions[source] = len(updated_df)

                    # 新しいデータ中で現在のSOURCEと同じ行を見つける
                    same_source_new_data = first_chunks_df[first_chunks_df['SOURCE'] == source]

                    # 新しいデータ中で現在のSOURCEと同じ行を走査
                    for _, new_row in same_source_new_data.iterrows():
                        # 現在の行の前に新しいデータを挿入
                        updated_df = pd.concat([updated_df[:insert_positions[source]],
                                                pd.DataFrame(new_row).T,
                                                updated_df[insert_positions[source]:]])

                        # 現在のSOURCEの挿入位置を更新
                        insert_positions[source] += 1

                    # 現在の行をupdated_dfに追加
                    updated_df = pd.concat([updated_df[:insert_positions[source]],
                                            pd.DataFrame(row).T,
                                            updated_df[insert_positions[source]:]])

                    # 現在のSOURCEの挿入位置を更新
                    insert_positions[source] += 1
                docs_dataframe = updated_df

            conn.commit()

            docs_dataframe['KEY_WORDS'] = docs_dataframe['CONTENT'].apply(
                lambda x: [word for word in search_texts if word.lower() in x.lower()])
            docs_dataframe['KEY_WORDS'] = docs_dataframe['KEY_WORDS'].apply(format_keywords)
            docs_dataframe['CONTENT'] = docs_dataframe['CONTENT'].apply(replace_newlines)

            if answer_by_one_checkbox_input:
                docs_dataframe = docs_dataframe[docs_dataframe['SOURCE'] == docs_dataframe['SOURCE'].iloc[0]]

            docs_dataframe = docs_dataframe.reset_index(drop=True)
            docs_dataframe.insert(0, 'NO', pd.Series(range(1, len(docs_dataframe) + 1)))
            print(f"{docs_dataframe=}")

            if len(docs_dataframe) == 0:
                docs_dataframe = pd.DataFrame(
                    columns=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"])

            return (
                gr.Textbox(value=query_sql_output.strip()),
                gr.Markdown(
                    "**検索結果数**: " + str(len(docs_dataframe)) + "   |   **検索キーワード**: (" + str(
                        len(search_texts)) + ")[" + ', '.join(search_texts) + "]",
                    visible=True
                ),
                gr.Dataframe(
                    value=docs_dataframe,
                    wrap=True,
                    headers=["NO", "CONTENT", "EMBED_ID", "SOURCE", "DISTANCE", "SCORE", "KEY_WORDS"],
                    column_widths=["4%", "50%", "6%", "8%", "6%", "6%", "8%"],
                    row_count=(len(docs_dataframe) if len(docs_dataframe) > 0 else 1, "fixed")
                )
            )
