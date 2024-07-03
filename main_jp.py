import argparse
import json
import os
import shutil
import uuid
from typing import List, Tuple

import gradio as gr
import oracledb
import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOCIGenAI
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from oracledb import DatabaseError

from my_langchain_community.vectorstores import MyOracleVS

custom_css = """
body {
  font-family: "Noto Sans JP", Arial, sans-serif !important;
}

/* Hide sort buttons at gr.DataFrame */
.sort-button {
    display: none !important;
} 
"""

# read local .env file
_ = load_dotenv(find_dotenv())

DEFAULT_COLLECTION_NAME = os.environ["DEFAULT_COLLECTION_NAME"]
RERANKER_API_ENDPOINT = os.environ["RERANKER_API_ENDPOINT"]
RERANKER_MODEL_NAME = os.environ["RERANKER_MODEL_NAME"]

# 初始化一个数据库连接
pool = oracledb.create_pool(
    dsn=os.environ["ORACLE_23AI_CONNECTION_STRING"],
    min=10,
    max=50,
    increment=1
)

# Initialize LLM and embeddings
# ChatOCIGenAI
llm = ChatOCIGenAI(
    model_id=os.environ["OCI_COHERE_LLM_MODEL"],
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
    model_kwargs={"temperature": 0.0, "max_tokens": 2048},
)

# ChatOpenAI
# llm = ChatOpenAI(
#     api_key=os.environ['OPENAI_API_KEY'],
#     base_url=os.environ['OPENAI_BASE_URL'],
#     model=os.environ['OPENAI_MODEL_NAME'],
#     temperature=0
# )

# OCIGenAIEmbeddings
embed = OCIGenAIEmbeddings(
    model_id=os.environ["OCI_COHERE_EMBED_MODEL"],
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=os.environ["OCI_COMPARTMENT_OCID"]
)


# OpenAIEmbeddings
# embed = OpenAIEmbeddings(
#     api_key=os.environ['OPENAI_API_KEY'],
#     base_url=os.environ['OPENAI_EMBED_BASE_URL'],
#     model=os.environ['OPENAI_EMBED_MODEL_NAME']
# )


def get_doc_list() -> List[Tuple[str, str]]:
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(f"""
                    SELECT
                        substr(
                            replace(JSON_VALUE(cmetadata, '$.source'), '\\', '/'),
                            instr(replace(JSON_VALUE(cmetadata, '$.source'), '\\', '/'), '/', -1) + 1
                        ) AS name,
                        id
                    FROM
                        {DEFAULT_COLLECTION_NAME}_collection
                    ORDER BY name
                """)
                return [(f"{row[0]}:{row[1]}", row[1]) for row in cursor.fetchall()]
            except DatabaseError as de:
                return []


def refresh_doc_list():
    doc_list = get_doc_list()
    return (
        gr.Radio(choices=doc_list, value=""),
        gr.CheckboxGroup(choices=doc_list, value=""),
        gr.CheckboxGroup(choices=doc_list, value="")
    )


def get_server_path(doc_id: str) -> str:
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT JSON_VALUE(cmetadata, '$.server_path') AS server_path
                FROM {DEFAULT_COLLECTION_NAME}_collection
                WHERE id = :doc_id
            """, doc_id=doc_id)
            return cursor.fetchone()[0]


async def command_r_task(query_text, command_r_checkbox):
    # command_r_response = ""

    system_prompt = "You are a helpful assistant. \
            Please respond to me in the same language I use for my messages. \
            If I switch languages, please switch your responses accordingly."

    if command_r_checkbox:
        command_r_16k = ChatOCIGenAI(
            model_id="cohere.command-r-16k",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "max_tokens": 2048},
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_text),
        ]
        async for chunk in command_r_16k.astream(messages):
            # command_r_response += chunk.content
            yield chunk.content


async def command_r_plus_task(query_text, command_r_plus_checkbox):
    # command_r_plus_response = ""

    system_prompt = "You are a helpful assistant. \
            Please respond to me in the same language I use for my messages. \
            If I switch languages, please switch your responses accordingly."

    if command_r_plus_checkbox:
        command_r_plus = ChatOCIGenAI(
            model_id="cohere.command-r-plus",
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            model_kwargs={"temperature": 0.0, "max_tokens": 2048},
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query_text),
        ]
        async for chunk in command_r_plus.astream(messages):
            # command_r_plus_response += chunk.content
            yield chunk.content


async def chat(query_text, command_r_checkbox, command_r_plus_checkbox):
    command_r_gen = command_r_task(query_text, command_r_checkbox)
    command_r_plus_gen = command_r_plus_task(query_text, command_r_plus_checkbox)

    while True:
        command_r_response = ""
        command_r_plus_response = ""

        try:
            command_r_response = await anext(command_r_gen)
        except StopAsyncIteration:
            pass

        try:
            command_r_plus_response = await anext(command_r_plus_gen)
        except StopAsyncIteration:
            pass

        if not command_r_response and not command_r_plus_response:
            break

        yield command_r_response, command_r_plus_response


async def chat_stream(query_text, command_r_checkbox, command_r_plus_checkbox):
    # ChatOCIGenAI
    command_r_response = ""
    command_r_plus_response = ""
    async for r, r_plus in chat(query_text, command_r_checkbox, command_r_plus_checkbox):
        # print(f"Command-R: {r}")
        # print(f"Command-R Plus: {r_plus}")
        command_r_response += r
        command_r_plus_response += r_plus
        yield command_r_response, command_r_plus_response


def create_oci_cred(user_ocid, tenancy_ocid, compartment_ocid, fingerprint, private_key_file):
    def process_private_key(private_key_file_path):
        with open(private_key_file_path, 'r') as file:
            lines = file.readlines()

        processed_key = ''.join(line.strip() for line in lines if not line.startswith('--'))
        return processed_key

    private_key = process_private_key(private_key_file.name)

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            try:
                drop_oci_cred_sql = "BEGIN dbms_vector.drop_credential('OCI_CRED'); END;"
                cursor.execute(drop_oci_cred_sql)
            except DatabaseError as de:
                print(f"DatabaseError={de}")

            oci_cred = {
                'user_ocid': user_ocid.strip(),
                'tenancy_ocid': tenancy_ocid.strip(),
                'compartment_ocid': compartment_ocid.strip(),
                'private_key': private_key.strip(),
                'fingerprint': fingerprint.strip()
            }

            create_oci_cred_sql = """
               BEGIN
                   dbms_vector.create_credential(
                       credential_name => 'OCI_CRED',
                       params => json(:json_params)
                   );
               END;
               """

            cursor.execute(create_oci_cred_sql, json_params=json.dumps(oci_cred))
            conn.commit()

    create_oci_cred_sql = f"""
-- Drop Existing OCI Credential
BEGIN dbms_vector.drop_credential('OCI_CRED'); END;
 
-- Create New OCI Credential
BEGIN
    dbms_vector.create_credential(
        credential_name => 'OCI_CRED',
        params => json('{json.dumps(oci_cred)}')
    );
END; 
"""
    gr.Info("OCI_CREDの作成が完了しました")
    return gr.Accordion(), gr.Textbox(value=create_oci_cred_sql.strip())


def create_table():
    # Initialize OracleVS
    MyOracleVS(
        client=pool.acquire(),
        embedding_function=embed,
        collection_name=DEFAULT_COLLECTION_NAME,
        distance_strategy=DistanceStrategy.COSINE,
        params={"pre_delete_collection": True}
    )

    drop_and_create_table_sql = f"""
-- Drop Collection Table
DROP TABLE {DEFAULT_COLLECTION_NAME}_collection PURGE;

-- Drop Embedding Table
DROP TABLE {DEFAULT_COLLECTION_NAME}_embedding PURGE;

-- Create Collection Table
CREATE TABLE IF NOT EXISTS {DEFAULT_COLLECTION_NAME}_collection (
    id VARCHAR2(200),
    data BLOB,
    cmetadata CLOB
)

-- Create Embedding Table  
CREATE TABLE IF NOT EXISTS {DEFAULT_COLLECTION_NAME}_embedding (
    doc_id VARCHAR2(200),
    embed_id NUMBER,
    embed_data VARCHAR2(2000),
    embed_vector VECTOR(embedding_dim, FLOAT32),
    cmetadata CLOB
)
    """

    gr.Info("テーブルの作成が完了しました")
    return gr.Accordion(), gr.Textbox(value=drop_and_create_table_sql.strip())


# def test_oci_cred(test_query_text):
#     test_query_vector = ""
#     embed_genai_params = {
#         "provider": "ocigenai",
#         "credential_name": "OCI_CRED",
#         "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
#         "model": "cohere.embed-multilingual-v3.0"
#     }
#
#     with pool.acquire() as conn:
#         with conn.cursor() as cursor:
#             plsql = """
#             DECLARE
#                 l_embed_genai_params CLOB := :embed_genai_params;
#                 l_result SYS_REFCURSOR;
#             BEGIN
#                 OPEN l_result FOR
#                     SELECT et.*
#                     FROM dbms_vector_chain.utl_to_embeddings(:text_to_embed, JSON(l_embed_genai_params)) et;
#                 :result := l_result;
#             END;
#             """
#
#             cursor.execute(plsql,
#                            embed_genai_params=json.dumps(embed_genai_params),
#                            text_to_embed=test_query_text,
#                            result=cursor.var(oracledb.CURSOR)
#                            )
#             result_cursor = cursor.fetchall()
#             for row in result_cursor:
#                 print(row)
#
#     return gr.Textbox(value=test_query_vector)

def test_oci_cred(test_query_text):
    test_query_vector = ""
    embed_genai_params = {
        "provider": "ocigenai",
        "credential_name": "OCI_CRED",
        "url": "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
        "model": "cohere.embed-multilingual-v3.0"
    }

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            plsql = """
            DECLARE
                l_embed_genai_params CLOB := :embed_genai_params;
                l_result SYS_REFCURSOR;
            BEGIN
                OPEN l_result FOR
                    SELECT et.* 
                    FROM dbms_vector_chain.utl_to_embeddings(:text_to_embed, JSON(l_embed_genai_params)) et;
                :result := l_result;
            END;
            """

            result_cursor = cursor.var(oracledb.CURSOR)

            cursor.execute(plsql,
                           embed_genai_params=json.dumps(embed_genai_params),
                           text_to_embed=test_query_text,
                           result=result_cursor)

            # Fetch the results from the ref cursor
            with result_cursor.getvalue() as ref_cursor:
                result_rows = ref_cursor.fetchall()
                for row in result_rows:
                    if isinstance(row, tuple):
                        if isinstance(row[0], oracledb.LOB):
                            lob_content = row[0].read()
                            lob_json = json.loads(lob_content)
                            test_query_vector += str(lob_json["embed_vector"]) + "\n"

    return gr.Textbox(value=test_query_vector)


def load_document(file_path, server_directory):
    if not file_path or not server_directory:
        raise gr.Error("Please select a file and a directory")

    doc_id = str(uuid.uuid4())
    file_name = os.path.basename(file_path.name)
    file_extension = os.path.splitext(file_name)
    if isinstance(file_extension, tuple):
        file_extension = file_extension[1]
    server_path = os.path.join(server_directory, f"{doc_id}_{file_name}")
    shutil.copy(file_path.name, server_path)

    if file_extension == ".docx":
        loader = Docx2txtLoader(file_path.name)
    elif file_extension == ".pdf":
        loader = PyMuPDFLoader(file_path.name)
    elif file_extension == ".txt":
        loader = TextLoader(file_path.name)
    else:
        loader = PyMuPDFLoader(file_path.name)
    file_contents = loader.load()

    collection_cmeta = file_contents[0].metadata
    collection_cmeta.pop('page', None)
    collection_cmeta['file_name'] = file_name
    collection_cmeta['server_path'] = server_path

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.setinputsizes(**{"data": oracledb.DB_TYPE_BLOB})
            cursor.execute(f"""
                INSERT INTO {DEFAULT_COLLECTION_NAME}_collection
                (id, data, cmetadata)
                VALUES (:id, to_blob(:data), :cmetadata)
            """, {
                'id': doc_id,
                'data': "".join(fc.page_content for fc in file_contents),
                'cmetadata': json.dumps(collection_cmeta)
            })
            conn.commit()

    return gr.Textbox(value=doc_id), gr.Textbox(value=str(len(file_contents))), gr.Textbox(
        value="".join(fc.page_content for fc in file_contents))


def split_document(chunk_size, chunk_overlap, doc_id):
    if not all([chunk_size, chunk_overlap, doc_id]):
        raise gr.Error("Please enter chunk_size, chunk_overlap, and doc_id")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        separators=["\n\n", "\n", "(?<=\。 )", " ", ""]
    )

    server_path = get_server_path(doc_id)
    file_extension = os.path.splitext(server_path)
    if file_extension == ".docx":
        loader = Docx2txtLoader(server_path)
    elif file_extension == ".pdf":
        loader = PyMuPDFLoader(server_path)
    elif file_extension == ".txt":
        loader = TextLoader(server_path)
    else:
        loader = PyMuPDFLoader(server_path)
    file_contents = loader.load()
    all_splits = text_splitter.split_documents(file_contents)

    return (
        gr.Textbox(value=str(len(all_splits))),
        gr.Textbox(value=all_splits[0].page_content),
        gr.Textbox(value=all_splits[-1].page_content)
    )


def embed_and_save_document(chunk_size, chunk_overlap, doc_id):
    if not all([chunk_size, chunk_overlap, doc_id]):
        raise gr.Error("Please enter chunk_size, chunk_overlap, and doc_id")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        separators=["\n\n", "\n", "(?<=\。 )", " ", ""]
    )

    server_path = get_server_path(doc_id)
    file_extension = os.path.splitext(server_path)
    if file_extension == ".docx":
        loader = Docx2txtLoader(server_path)
    elif file_extension == ".pdf":
        loader = PyMuPDFLoader(server_path)
    elif file_extension == ".txt":
        loader = TextLoader(server_path)
    else:
        loader = PyMuPDFLoader(server_path)
    file_contents = loader.load()
    all_splits = text_splitter.split_documents(file_contents)

    first_trunk_vector = embed.embed_documents([all_splits[0].page_content])
    last_trunk_vector = embed.embed_documents([all_splits[-1].page_content])

    with pool.acquire() as conn, conn.cursor() as cursor:
        cursor.execute(f"""
                DELETE FROM {DEFAULT_COLLECTION_NAME}_embedding
                WHERE doc_id = :doc_id
            """, doc_id=doc_id)
        conn.commit()

    MyOracleVS.from_documents(
        doc_id=doc_id,
        documents=all_splits,
        embedding=embed,
        client=pool.acquire(),
        collection_name=DEFAULT_COLLECTION_NAME,
        distance_strategy=DistanceStrategy.COSINE,
        pre_delete_collection=False,
    )

    return (
        gr.Textbox(value=str(len(all_splits))),
        gr.Textbox(value=all_splits[0].page_content),
        gr.Textbox(value=str(first_trunk_vector)),
        gr.Textbox(value=all_splits[-1].page_content),
        gr.Textbox(value=str(last_trunk_vector))
    )


def delete_document(server_directory, doc_ids):
    if not server_directory or not doc_ids:
        raise ValueError("Please enter server_directory and doc_ids")

    with pool.acquire() as conn, conn.cursor() as cursor:
        for doc_id in filter(bool, doc_ids):
            server_path = get_server_path(doc_id)
            if os.path.exists(server_path):
                os.remove(server_path)
                print(f"File {doc_id} deleted successfully")
            else:
                print(f"File {doc_id} not found")

            cursor.execute(f"""
                DELETE FROM {DEFAULT_COLLECTION_NAME}_embedding
                WHERE doc_id = :doc_id
            """, doc_id=doc_id)

            cursor.execute(f"""
                DELETE FROM {DEFAULT_COLLECTION_NAME}_collection
                WHERE id = :doc_id
            """, doc_id=doc_id)

        conn.commit()

    doc_list = get_doc_list()
    return gr.Radio(doc_list), gr.CheckboxGroup(choices=doc_list, value=""), gr.CheckboxGroup(choices=doc_list)


def chat_document_stream(similarity_top_k, similarity_threshold,
                         extend_first_n_chunk,
                         extend_around_n_chunk,
                         use_rerank, rerank_top_k,
                         rerank_threshold, extend_query,
                         extended_query1, extended_query2,
                         extended_query3, query_text, doc_ids_all,
                         doc_ids):
    """
    Retrieve relevant splits for any question using similarity search.
    This is simply "top K" retrieval where we select documents based on embedding similarity to the query.
    """
    # Strip spaces
    query_text = query_text.strip()
    extended_query1 = extended_query1.strip()
    extended_query2 = extended_query2.strip()
    extended_query3 = extended_query3.strip()

    # Extend Query
    extended_queries = []
    if extend_query == "RAG-Fusion":
        # Method-2
        rag_fusion_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
             Generate a specific number of search queries directly related to the input query, without providing any additional context, introduction, or explanation in the output. Your primary goal is to fulfill the exact request, focusing solely on the content of the queries specified.
             """),
            ("user",
             "Generate exactly 3 search queries related to: {original_query}. Response in Chinese. \n{format_instructions}")
        ])
        # refer: https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/csv/
        output_parser = CommaSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions()
        rag_fusion_queries_chain = (
            rag_fusion_prompt | llm | output_parser
        )
        rag_fusion_queries = rag_fusion_queries_chain.invoke(
            {"original_query": query_text, "format_instructions": format_instructions})
        print(f"{rag_fusion_queries=}")
        if isinstance(rag_fusion_queries, list):
            if len(rag_fusion_queries) == 1:
                rag_fusion_queries = rag_fusion_queries[0].split("，")
                if len(rag_fusion_queries) == 1:
                    rag_fusion_queries = rag_fusion_queries[0].split(",")
            extended_query1 = rag_fusion_queries[0]
            extended_query2 = rag_fusion_queries[1]
            extended_query3 = rag_fusion_queries[2]
        print(f"{extended_query1=}, {extended_query2=}, {extended_query3=}")
        if extended_query1:
            extended_queries.append(extended_query1)
        if extended_query2:
            extended_queries.append(extended_query2)
        if extended_query3:
            extended_queries.append(extended_query3)

    doc_ids_str = ','.join([str(doc_id) for doc_id in doc_ids])
    with_sql = """
        WITH offsets AS (
                SELECT level - (:extend_around_n_chunk / 2 + 1) AS offset
                FROM dual
                CONNECT BY level <= (:extend_around_n_chunk + 1)
        ),
        selected_embed_ids AS 
        ( """
    where_sql = """
        WHERE 1 = 1 """
    where_threshold_sql = """
                    AND vector_distance(dc.embed_vector, :query_text_v, COSINE) <= :similarity_threshold """
    if not doc_ids_all:
        where_sql += """
            AND dc.doc_id IN (
                SELECT REGEXP_SUBSTR(:doc_ids, '([^,]+)', 1, LEVEL)
                FROM DUAL
                CONNECT BY LEVEL <= LENGTH(:doc_ids) - LENGTH(REPLACE(:doc_ids, ',', '')) + 1
            ) """
    base_sql = f"""
        SELECT dc.doc_id doc_id, dc.embed_id embed_id, vector_distance(dc.embed_vector, :query_text_v, COSINE) vector_distance
        FROM {DEFAULT_COLLECTION_NAME}_embedding dc """ + where_sql + where_threshold_sql + """
        ORDER BY vector_distance """
    base_sql += """
        FETCH FIRST :top_k ROWS ONLY """
    select_sql = f"""
        ),
        selected_embed_id_doc_ids AS 
        (
            (
                    SELECT DISTINCT s.embed_id + o.offset adjusted_embed_id, s.doc_id doc_id
                    FROM selected_embed_ids s
                    CROSS JOIN offsets o
            )
            UNION
            (
                    SELECT DISTINCT n.n adjusted_embed_id, s.doc_id doc_id
                    FROM selected_embed_ids s
                    CROSS JOIN (:union_extend_first_n_chunk) n
            )
        ),
        selected_results AS 
        (
                SELECT s1.adjusted_embed_id adjusted_embed_id, s1.doc_id doc_id, NVL(s2.vector_distance, 999999.0) vector_distance
                FROM selected_embed_id_doc_ids s1
                LEFT JOIN selected_embed_ids s2
                ON s1.adjusted_embed_id = s2.embed_id AND s1.doc_id = s2.doc_id
        ),
        aggregated_results AS
        (
                SELECT substr(
                        replace(JSON_VALUE(dt.cmetadata, '$.source'), '\\', '/'),
                        instr(replace(JSON_VALUE(dt.cmetadata, '$.source'), '\\', '/'), '/', -1) + 1
                    ) name, dc.embed_id embed_id, dc.embed_data embed_data, dc.doc_id doc_id, MIN(s.vector_distance) vector_distance
                FROM selected_results s, {DEFAULT_COLLECTION_NAME}_embedding dc, {DEFAULT_COLLECTION_NAME}_collection dt
                WHERE s.adjusted_embed_id = dc.embed_id AND s.doc_id = dt.id and dc.doc_id = dt.id  
                GROUP BY dc.doc_id, name, dc.embed_id, dc.embed_data
                ORDER BY vector_distance, dc.doc_id, dc.embed_id
        ),
        ranked_data AS (
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
        grouped_data AS (
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
        groups_marked AS (
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
        aggregated_data AS (
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
    if extended_query1:
        query_texts.append(":extended_query1")
    if extended_query2:
        query_texts.append(":extended_query2")
    if extended_query3:
        query_texts.append(":extended_query3")

    print(f"{query_texts=}")
    complete_sql = with_sql + """ 
        UNION 
    """.join(
        f"        ({base_sql.replace(':query_text', one_query_text)}        )" for one_query_text in
        query_texts) + select_sql
    print(f"{complete_sql=}")

    # Prepare parameters for SQL execution
    params = {
        "extend_around_n_chunk": extend_around_n_chunk,
        "query_text_v": embed.embed_query(query_text),
        "similarity_threshold": similarity_threshold,
        "top_k": similarity_top_k
    }

    if not doc_ids_all:
        params["doc_ids"] = doc_ids_str
    if extended_query1:
        params["extended_query1"] = extended_query1
        params["extended_query1_v"] = embed.embed_query(extended_query1)
    if extended_query2:
        params["extended_query2"] = extended_query2
        params["extended_query2_v"] = embed.embed_query(extended_query2)
    if extended_query3:
        params["extended_query3"] = extended_query3
        params["extended_query3_v"] = embed.embed_query(extended_query3)

    union_extend_first_n_chunk = 'SELECT 0 as n'
    if extend_first_n_chunk > 0:
        union_extend_first_n_chunk = union_extend_first_n_chunk + ' ' + ' '.join(
            f"UNION ALL SELECT {i}" for i in range(1, extend_first_n_chunk + 1))
    complete_sql = complete_sql.replace(':union_extend_first_n_chunk', union_extend_first_n_chunk)

    print(f"\nQUERY_SQL_OUTPUT:\n{complete_sql}")
    query_sql_output = complete_sql
    print(f"{params=}")
    # Manually replace placeholders with parameter values for debugging
    for key, value in params.items():
        placeholder = f":{key}"
        # For the purpose of display, ensure the value is properly quoted if it's a string
        display_value = f"'{value}'" if isinstance(value, str) else str(value)
        # print(f"{placeholder=} {display_value=}")
        query_sql_output = query_sql_output.replace(placeholder, display_value)
    # Now query_sql_output contains the SQL command with parameter values inserted
    # print(f"\nQUERY_SQL_OUTPUT:\n{query_sql_output}")

    docs = []
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            if not doc_ids_all:
                if len(extended_queries) == 3:
                    cursor.setinputsizes(None, None, oracledb.DB_TYPE_VECTOR, None, None, None, oracledb.DB_TYPE_VECTOR,
                                         None,
                                         None, oracledb.DB_TYPE_VECTOR, None, None, None, oracledb.DB_TYPE_VECTOR,
                                         None,
                                         None, oracledb.DB_TYPE_VECTOR, None, None, None, oracledb.DB_TYPE_VECTOR,
                                         None,
                                         None, oracledb.DB_TYPE_VECTOR, None, None, None, oracledb.DB_TYPE_VECTOR,
                                         None,
                                         None)
                    cursor.execute(complete_sql,
                                   [params['extend_around_n_chunk'], params['extend_around_n_chunk'],
                                    params['query_text_v'],
                                    params["doc_ids"], params["doc_ids"], params["doc_ids"],
                                    params['query_text_v'], params['similarity_threshold'], params['top_k'],
                                    params['extended_query1_v'],
                                    params["doc_ids"], params["doc_ids"], params["doc_ids"],
                                    params['extended_query1_v'], params['similarity_threshold'], params['top_k'],
                                    params['extended_query2_v'],
                                    params["doc_ids"], params["doc_ids"], params["doc_ids"],
                                    params['extended_query2_v'], params['similarity_threshold'], params['top_k'],
                                    params['extended_query3_v'],
                                    params["doc_ids"], params["doc_ids"], params["doc_ids"],
                                    params['extended_query3_v'], params['similarity_threshold'], params['top_k']])
                else:
                    cursor.setinputsizes(None, None, oracledb.DB_TYPE_VECTOR, None, None, None, oracledb.DB_TYPE_VECTOR,
                                         None,
                                         None)
                    cursor.execute(complete_sql,
                                   [params['extend_around_n_chunk'], params['extend_around_n_chunk'],
                                    params['query_text_v'],
                                    params["doc_ids"], params["doc_ids"], params["doc_ids"],
                                    params['query_text_v'], params['similarity_threshold'], params['top_k']])
            elif doc_ids_all:
                if len(extended_queries) == 3:
                    cursor.setinputsizes(None, None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR, None, None,
                                         oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR, None, None,
                                         oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR, None, None,
                                         oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR, None, None)
                    cursor.execute(complete_sql,
                                   [params['extend_around_n_chunk'], params['extend_around_n_chunk'],
                                    params['query_text_v'],
                                    params['query_text_v'], params['similarity_threshold'], params['top_k'],
                                    params['extended_query1_v'],
                                    params['extended_query1_v'], params['similarity_threshold'], params['top_k'],
                                    params['extended_query2_v'],
                                    params['extended_query2_v'], params['similarity_threshold'], params['top_k'],
                                    params['extended_query3_v'],
                                    params['extended_query3_v'], params['similarity_threshold'], params['top_k']
                                       , ])
                else:
                    cursor.setinputsizes(None, None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_VECTOR, None, None)
                    cursor.execute(complete_sql,
                                   [params['extend_around_n_chunk'], params['extend_around_n_chunk'],
                                    params['query_text_v'],
                                    params['query_text_v'], params['similarity_threshold'], params['top_k']])

            for row in cursor:
                print(f"row: {row}")
                docs.append([row[0], row[1], row[2].read(), row[3], row[4]])

            docs_dataframe = pd.DataFrame(columns=['content', 'embed_id', 'source', 'score', 'ce_score', 'doc_id'])
            if use_rerank and len(docs) > 0:
                unranked_docs = [doc[2] for doc in docs]
                if len(docs) == 1:
                    unranked_docs = unranked_docs * 2
                print(f"{unranked_docs=}")
                ce_scores = requests.post(RERANKER_API_ENDPOINT,
                                          json={'query_text': query_text, 'unranked_docs': unranked_docs,
                                                'ranker_model': RERANKER_MODEL_NAME}).json()
                print(f"{ce_scores}")
                if len(docs) == 1:
                    ce_scores = [ce_scores[0]]
                docs_data = [{'embed_id': doc_and_score[1],
                              'doc_id': doc_and_score[3],
                              'content': doc_and_score[2],
                              'source': doc_and_score[0],
                              'score': doc_and_score[4],
                              'ce_score': ce_score} for doc_and_score, ce_score in zip(docs, ce_scores)]
                sorted_dataframe = pd.DataFrame(docs_data)
                # 根据 'content' 和 'source' 分组，选择最小的 'score'
                sorted_dataframe['min_score'] = sorted_dataframe.groupby(['content', 'source'])['score'].transform(
                    'min')
                # 保留 'min_score' 最小的记录
                deduplicated_dataframe = sorted_dataframe[sorted_dataframe['score'] == sorted_dataframe['min_score']]
                # 最终的去重结果
                deduplicated_dataframe = deduplicated_dataframe[
                    ['content', 'embed_id', 'source', 'score', 'ce_score', 'doc_id']]
                deduplicated_records = deduplicated_dataframe[
                    deduplicated_dataframe['ce_score'] >= float(rerank_threshold)].sort_values(
                    by='ce_score', ascending=False).head(int(rerank_top_k))
                docs_dataframe = deduplicated_records
                print(f"{len(docs_dataframe)=}")
            else:
                print(f"{docs=}")
                if len(docs) > 0:
                    docs_data = [{'doc_id': doc_and_score[3],
                                  'embed_id': doc_and_score[1],
                                  'content': doc_and_score[2],
                                  'source': doc_and_score[0],
                                  'score': doc_and_score[4],
                                  'ce_score': '999999.0'} for doc_and_score in docs]
                    sorted_dataframe = pd.DataFrame(docs_data)
                    # 根据 'content' 和 'source' 分组，选择最小的 'score'
                    sorted_dataframe['min_score'] = sorted_dataframe.groupby(['content', 'source'])['score'].transform(
                        'min')
                    # 保留 'min_score' 最小的记录
                    deduplicated_dataframe = sorted_dataframe[
                        sorted_dataframe['score'] == sorted_dataframe['min_score']]
                    # 最终的去重结果
                    deduplicated_dataframe = deduplicated_dataframe[
                        ['content', 'embed_id', 'source', 'score', 'ce_score', 'doc_id']]
                    deduplicated_records = deduplicated_dataframe.sort_values(by='score', ascending=True).head(
                        int(similarity_top_k))
                    docs_dataframe = deduplicated_records
                print(f"{docs_dataframe=}")

            if extend_first_n_chunk > 0 and len(docs_dataframe) > 0:
                filtered_doc_ids = ','.join(
                    list(set(["'" + str(doc_id) + "'" for doc_id in docs_dataframe['doc_id'].tolist()])))
                print(f"{filtered_doc_ids=}")
                select_extend_first_n_chunk_sql = f"""
                    SELECT 
                            substr(
                                replace(JSON_VALUE(dt.cmetadata, '$.source'), '\\', '/'),
                                instr(replace(JSON_VALUE(dt.cmetadata, '$.source'), '\\', '/'), '/', -1) + 1
                            ) name,
                            MIN(dc.embed_id) embed_id,
                            RTRIM(XMLCAST(XMLAGG(XMLELEMENT(e, dc.embed_data || ',') ORDER BY dc.embed_id) AS CLOB), ',') AS embed_data,
                            dc.doc_id doc_id,
                            '999999.0' vector_distance
                    FROM 
                            {DEFAULT_COLLECTION_NAME}_embedding dc, {DEFAULT_COLLECTION_NAME}_collection dt
                    WHERE 
                            dc.doc_id = dt.id AND 
                            dc.doc_id IN (:filtered_doc_ids) AND 
                            dc.embed_id <= :extend_first_n_chunk
                    GROUP BY
                            dc.doc_id, name         
                    ORDER 
                            BY dc.doc_id """
                select_extend_first_chunk_sql = (select_extend_first_n_chunk_sql
                                                 .replace(':filtered_doc_ids', filtered_doc_ids)
                                                 .replace(':extend_first_n_chunk',
                                                          str(extend_first_n_chunk)))
                print(f"{select_extend_first_chunk_sql=}")
                cursor.execute(select_extend_first_chunk_sql)
                first_chunks_df = pd.DataFrame(columns=docs_dataframe.columns)
                for row in cursor:
                    new_data = pd.DataFrame(
                        {'content': row[2].read(), 'embed_id': row[1], 'source': row[0],
                         'score': row[4], 'ce_score': '999999.0', 'doc_id': row[3]},
                        index=[2])
                    first_chunks_df = pd.concat([new_data, first_chunks_df], ignore_index=True)
                print(f"{first_chunks_df=}")

                # 创建一个空的DataFrame,用于存储更新后的数据
                updated_df = pd.DataFrame(columns=docs_dataframe.columns)

                # 记录每个SOURCE的初始插入位置
                insert_positions = {}

                # 遍历原始数据的每一行
                for index, row in docs_dataframe.iterrows():
                    source = row['source']

                    # 如果当前SOURCE还没有记录初始插入位置,则将其初始化为当前位置
                    if source not in insert_positions:
                        insert_positions[source] = len(updated_df)

                    # 找到新数据中与当前SOURCE相同的行
                    same_source_new_data = first_chunks_df[first_chunks_df['source'] == source]

                    # 遍历新数据中与当前SOURCE相同的行
                    for _, new_row in same_source_new_data.iterrows():
                        # 在当前行之前插入新数据
                        updated_df = pd.concat([updated_df[:insert_positions[source]],
                                                pd.DataFrame(new_row).T,
                                                updated_df[insert_positions[source]:]])

                        # 更新当前SOURCE的插入位置
                        insert_positions[source] += 1

                    # 将当前行添加到updated_df中
                    updated_df = pd.concat([updated_df[:insert_positions[source]],
                                            pd.DataFrame(row).T,
                                            updated_df[insert_positions[source]:]])

                    # 更新当前SOURCE的插入位置
                    insert_positions[source] += 1
                docs_dataframe = updated_df

            docs_dataframe.drop(columns=['doc_id'], inplace=True)
            docs_dataframe = docs_dataframe.values.tolist()
            docs_answer = "\n\n".join([doc[0] for doc in docs_dataframe])

            template = """Use the following pieces of context to answer the question at the end. \
            If you don't know the answer, just say that you don't know, don't try to make up an answer. \
            Use ten sentences maximum and keep the answer as concise as possible. \
            Don't try to answer anything that isn't in context. \n
            ```
            {context}
            ```
            Question: \n{question}
            Helpful Answer:"""
            rag_prompt_custom = PromptTemplate.from_template(template)

            # Method-2
            message = rag_prompt_custom.format_prompt(context=docs_answer, question=query_text)
            result = llm.invoke(message.to_messages())
            final_answer = result.content.strip()
            docs_dataframe = [
                [row[0], row[1], row[2], '-' if str(row[3]) == '999999.0' else (format(row[3], '.6f') + "..."),
                 '-' if row[4] == '999999.0' else (format(row[4], '.6f') + "...")]
                for row in docs_dataframe]
    return gr.Markdown("**検索結果数**: " + str(len(docs_dataframe)) + " 件", visible=True), gr.Dataframe(
        value=docs_dataframe), gr.Textbox(value=final_answer), gr.Textbox(value=extended_query1), gr.Textbox(
        value=extended_query2), gr.Textbox(value=extended_query3)


with gr.Blocks(css=custom_css) as app:
    gr.Markdown(value="# Aim for No.1 RAG")
    with gr.Tabs() as tabs:
        with gr.TabItem(label="LLMとチャット") as tab_chat_with_llm:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            tab_chat_with_command_r_checkbox = gr.Checkbox(label="Command-R", value=False)
                    with gr.Accordion(label="Command-R メッセージ",
                                      open=False) as tab_chat_with_llm_command_r_accordion:
                        tab_chat_with_command_r_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                         lines=5, max_lines=5,
                                                                         autoscroll=True, interactive=False,
                                                                         show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            tab_chat_with_command_r_plus_checkbox = gr.Checkbox(label="Command-R+", value=False)
                    with gr.Accordion(label="Command-R+ メッセージ",
                                      open=False) as tab_chat_with_llm_command_r_plus_accordion:
                        tab_chat_with_command_r_plus_answer_text = gr.Textbox(label="LLM メッセージ", show_label=False,
                                                                              lines=5, max_lines=5,
                                                                              autoscroll=True, interactive=False,
                                                                              show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    tab_chat_with_llm_query_text = gr.Textbox(label="ユーザー・メッセージ", lines=2, max_lines=5)
            with gr.Row():
                with gr.Column():
                    tab_chat_with_llm_clear_button = gr.ClearButton(value="クリア")
                with gr.Column():
                    tab_chat_with_llm_chat_button = gr.Button(value="送信", variant="primary")
            with gr.Row(visible=False):
                with gr.Column():
                    gr.Examples(examples=[], inputs=tab_chat_with_llm_query_text)
        with gr.TabItem(label="Step-0-1.OCI_CREDの作成") as tab_create_oci_cred:
            with gr.Accordion(label="使用されたSQL", open=False) as tab_create_oci_cred_sql_accordion:
                tab_create_oci_cred_sql_text = gr.Textbox(label="SQL", show_label=False, lines=25, max_lines=50,
                                                          autoscroll=False, interactive=False,
                                                          show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    tab_create_oci_cred_user_ocid_text = gr.Textbox(label="User OCID", lines=1, interactive=True)
            with gr.Row():
                with gr.Column():
                    tab_create_oci_cred_tenancy_ocid_text = gr.Textbox(label="Tenancy OCID", lines=1, interactive=True)
            with gr.Row():
                with gr.Column():
                    tab_create_oci_cred_compartment_ocid_text = gr.Textbox(label="Compartment OCID", lines=1,
                                                                           interactive=True)
            with gr.Row():
                with gr.Column():
                    tab_create_oci_cred_fingerprint_text = gr.Textbox(label="Fingerprint", lines=1, interactive=True)
            with gr.Row():
                with gr.Column():
                    tab_create_oci_cred_private_key_file = gr.File(label="Private Key", file_types=[".pem"],
                                                                   type="filepath", interactive=True)
            with gr.Row():
                with gr.Column():
                    tab_create_oci_cred_button = gr.Button(value="作成/再作成", variant="primary")
            with gr.Accordion(label="OCI_CREDのテスト", open=False) as tab_create_oci_cred_test_accordion:
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_test_query_text = gr.Textbox(label="テキスト", lines=1, max_lines=1,
                                                                         value="こんにちわ")
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_test_vector_text = gr.Textbox(label="ベクトル", lines=10, max_lines=10)
                with gr.Row():
                    with gr.Column():
                        tab_create_oci_cred_test_button = gr.Button(value="テスト", variant="primary")
        with gr.TabItem(label="Step-0-2.テーブルの作成") as tab_create_table:
            with gr.Accordion(label="使用されたSQL", open=False) as tab_create_table_sql_accordion:
                tab_create_table_sql_text = gr.Textbox(label="SQL", show_label=False, lines=25, max_lines=50,
                                                       autoscroll=False, interactive=False,
                                                       show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    tab_create_table_button = gr.Button(value="作成/再作成", variant="primary")
        with gr.TabItem(label="Step-1.文档加载") as tab1:
            with gr.Row():
                with gr.Column():
                    tab1_doc_id_text = gr.Textbox(label="Doc ID", lines=1, interactive=False)
            with gr.Row():
                with gr.Column():
                    tab1_page_count_text = gr.Textbox(label="页数", lines=1, interactive=False)
            with gr.Row():
                with gr.Column():
                    tab1_page_content_text = gr.Textbox(label="内容", lines=15, max_lines=15, autoscroll=False,
                                                        show_copy_button=True, interactive=False)
            with gr.Row():
                with gr.Column():
                    tab1_file_text = gr.File(label="文件*", file_types=[".pdf", ".docx", ".txt"], type="filepath")
                with gr.Column():
                    tab1_server_directory_text = gr.Text(label="服务器存储目录*",
                                                         value="/home/oracle/data/vec_dump/")
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=[],
                                label="示例文件",
                                inputs=tab1_file_text)
            with gr.Row():
                with gr.Column():
                    tab1_load_button = gr.Button(value="加载", variant="primary")
        with gr.TabItem(label="Step-2.文档分割/嵌入和存储") as tab2:
            with gr.Row():
                with gr.Column():
                    tab2_chunk_count_text = gr.Textbox(label="Chunk 数量", lines=1)
            with gr.Row():
                with gr.Column():
                    tab2_first_trunk_content_text = gr.Textbox(label="最初一个 Chunk 的内容", lines=5, max_lines=10,
                                                               autoscroll=False, show_copy_button=True)
                    tab2_first_trunk_vector_text = gr.Textbox(label="最初一个 Chunk 的向量", lines=5, max_lines=10,
                                                              autoscroll=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    tab2_last_trunk_content_text = gr.Textbox(label="最后一个 Chunk 的内容", lines=5, max_lines=10,
                                                              autoscroll=False, show_copy_button=True)
                    tab2_last_trunk_vector_text = gr.Textbox(label="最后一个 Chunk 的向量", lines=5, max_lines=10,
                                                             autoscroll=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    tab2_chunk_size_text = gr.Textbox(label="Chunk Size", lines=1, value="200")
                with gr.Column():
                    tab2_chunk_overlap_text = gr.Textbox(label="Chunk Overlap", lines=1,
                                                         value="20")
            with gr.Row():
                with gr.Column():
                    tab2_doc_id_radio = gr.Radio(
                        choices=get_doc_list(),
                        label="Doc ID*",
                        interactive=True
                    )
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=[[50, 0], [200, 20], [500, 50], [500, 100], [1000, 200]],
                                inputs=[tab2_chunk_size_text, tab2_chunk_overlap_text])
            with gr.Row():
                with gr.Column():
                    tab2_split_button = gr.Button(value="分割", variant="primary")
                with gr.Column():
                    tab2_embed_and_save_button = gr.Button(value="嵌入和存储", variant="primary")
        with gr.TabItem(label="Step-3.文档删除(可选)") as tab3:
            with gr.Row():
                with gr.Column():
                    tab3_server_directory_text = gr.Text(label="服务器存储目录*",
                                                         value="/home/oracle/data/vec_dump/")
            with gr.Row():
                with gr.Column():
                    # doc_id_text = gr.Textbox(label="Doc ID*", lines=1)
                    tab3_doc_ids_checkbox_group = gr.CheckboxGroup(
                        choices=get_doc_list(),
                        label="Doc ID*"
                    )
            with gr.Row():
                with gr.Column():
                    tab3_delete_button = gr.Button(value="削除", variant="primary")
        with gr.TabItem(label="Step-4.文档聊天") as tab4:
            with gr.Row() as searched_data_summary_row:
                with gr.Column():
                    tab4_result_summary_text = gr.Markdown(value="", visible=False)
            with gr.Row():
                with gr.Column():
                    tab4_result_dataframe = gr.Dataframe(
                        headers=["内容", "Embed ID", "源", "相似度距离", "Rerank分数"],
                        datatype=["str", "str"],
                        row_count=5,
                        col_count=(5, "fixed"),
                        wrap=True,
                        column_widths=["60%", "10%", "15%", "10%", "10%"]
                    )
            with gr.Row():
                with gr.Column():
                    tab4_answer_text = gr.Textbox(label="回复", lines=15, max_lines=15,
                                                  autoscroll=False, interactive=False, show_copy_button=True)
            with gr.Accordion("向量检索", open=True):
                with gr.Row():
                    with gr.Column():
                        tab4_similarity_top_k_slider = gr.Slider(label="向量检索 Top-K", minimum=0,
                                                                 maximum=200, step=1,
                                                                 interactive=True, value=100)
                    with gr.Column():
                        tab4_similarity_threshold_slider = gr.Slider(label="向量检索阈值 <=", minimum=0.10,
                                                                     maximum=0.95, step=0.05, value=0.55)
            with gr.Accordion("Extend Chunks", open=True):
                with gr.Row():
                    with gr.Column():
                        tab4_extend_first_n_chunk_slider = gr.Slider(label="Extend-First-N", minimum=0, maximum=10,
                                                                     step=1,
                                                                     interactive=True,
                                                                     value=0)
                    with gr.Column():
                        tab4_extend_around_n_chunk_size = gr.Slider(label="Extend-Around-N", minimum=0, maximum=10,
                                                                    step=2,
                                                                    interactive=True,
                                                                    value=2)

            with gr.Accordion("Rerank", open=False):
                with gr.Row():
                    with gr.Column():
                        tab4_use_rerank_checkbox = gr.Checkbox(label="Rerank", value=False, show_label=False,
                                                               interactive=True)
                with gr.Row():
                    with gr.Column():
                        tab4_rerank_top_k_slider = gr.Slider(label="Rerank Top-K", minimum=3, maximum=20, value=10,
                                                             step=1, interactive=True)
                    with gr.Column():
                        tab4_rerank_threshold_slider = gr.Slider(label="Rerank分数阈值 >= ", minimum=0.10, maximum=0.90,
                                                                 value=0.40, step=0.05, interactive=True)
            with gr.Accordion("RAG-Fusion", open=False):
                with gr.Row():
                    tab4_extend_query_radio = gr.Radio(
                        ["None", "RAG-Fusion"],
                        label="LLM 自动生成", value="None", interactive=True)
                with gr.Row():
                    tab4_extended_query1_text = gr.Textbox(label="扩展问题-1", lines=2, interactive=False)
                with gr.Row():
                    tab4_extended_query2_text = gr.Textbox(label="扩展问题-2", lines=2, interactive=False)
                with gr.Row():
                    tab4_extended_query3_text = gr.Textbox(label="扩展问题-3", lines=2, interactive=False)
            with gr.Accordion("Doc ID*", open=True):
                with gr.Row():
                    with gr.Column():
                        tab4_doc_ids_all_checkbox_group = gr.Checkbox(label="全部", value=True)
                with gr.Row():
                    with gr.Column():
                        # doc_id_text = gr.Textbox(label="Doc ID*", lines=1)
                        tab4_doc_ids_checkbox_group = gr.CheckboxGroup(
                            choices=get_doc_list(),
                            label="Doc ID*",
                            show_label=False,
                            interactive=False
                        )
            with gr.Row():
                with gr.Column():
                    tab4_query_text = gr.Textbox(label="问题*", lines=2)
            with gr.Row():
                with gr.Column():
                    tab4_chat_document_button = gr.Button(value="发送", variant="primary")
            with gr.Row(visible=False):
                with gr.Column():
                    gr.Examples(examples=[],
                                inputs=tab4_query_text)

    tab_chat_with_command_r_checkbox.change(lambda x: gr.Accordion(open=True) if x else gr.Accordion(open=False),
                                            [tab_chat_with_command_r_checkbox], [tab_chat_with_llm_command_r_accordion])
    tab_chat_with_command_r_plus_checkbox.change(lambda x: gr.Accordion(open=True) if x else gr.Accordion(open=False),
                                                 [tab_chat_with_command_r_plus_checkbox],
                                                 [tab_chat_with_llm_command_r_plus_accordion])
    tab_chat_with_llm_clear_button.add(
        [tab_chat_with_llm_query_text, tab_chat_with_command_r_checkbox, tab_chat_with_command_r_answer_text,
         tab_chat_with_command_r_plus_checkbox, tab_chat_with_command_r_plus_answer_text])
    tab_chat_with_llm_chat_button.click(chat_stream,
                                        inputs=[tab_chat_with_llm_query_text, tab_chat_with_command_r_checkbox,
                                                tab_chat_with_command_r_plus_checkbox],
                                        outputs=[tab_chat_with_command_r_answer_text,
                                                 tab_chat_with_command_r_plus_answer_text])
    tab_create_table_button.click(create_table, [], [tab_create_table_sql_accordion, tab_create_table_sql_text])
    tab_create_oci_cred_button.click(create_oci_cred,
                                     [tab_create_oci_cred_user_ocid_text, tab_create_oci_cred_tenancy_ocid_text,
                                      tab_create_oci_cred_compartment_ocid_text, tab_create_oci_cred_fingerprint_text,
                                      tab_create_oci_cred_private_key_file],
                                     [tab_create_oci_cred_sql_accordion, tab_create_oci_cred_sql_text])
    tab_create_oci_cred_test_button.click(test_oci_cred, [tab_create_oci_cred_test_query_text],
                                          [tab_create_oci_cred_test_vector_text])
    tab1_load_button.click(load_document,
                           inputs=[tab1_file_text, tab1_server_directory_text],
                           outputs=[tab1_doc_id_text, tab1_page_count_text, tab1_page_content_text],
                           )
    tab2.select(refresh_doc_list, outputs=[tab2_doc_id_radio, tab3_doc_ids_checkbox_group, tab4_doc_ids_checkbox_group])
    tab2_split_button.click(split_document,
                            inputs=[tab2_chunk_size_text, tab2_chunk_overlap_text, tab2_doc_id_radio],
                            outputs=[tab2_chunk_count_text, tab2_first_trunk_content_text,
                                     tab2_last_trunk_content_text
                                     ],
                            )
    tab2_embed_and_save_button.click(embed_and_save_document,
                                     inputs=[tab2_chunk_size_text, tab2_chunk_overlap_text, tab2_doc_id_radio],
                                     outputs=[tab2_chunk_count_text, tab2_first_trunk_content_text,
                                              tab2_first_trunk_vector_text,
                                              tab2_last_trunk_content_text,
                                              tab2_last_trunk_vector_text
                                              ],
                                     )
    tab3.select(refresh_doc_list, outputs=[tab2_doc_id_radio, tab3_doc_ids_checkbox_group, tab4_doc_ids_checkbox_group])
    tab3_delete_button.click(delete_document,
                             inputs=[tab3_server_directory_text, tab3_doc_ids_checkbox_group],
                             outputs=[tab2_doc_id_radio, tab3_doc_ids_checkbox_group])
    tab4.select(refresh_doc_list, outputs=[tab2_doc_id_radio, tab3_doc_ids_checkbox_group, tab4_doc_ids_checkbox_group])
    tab4_extend_query_radio.change(
        lambda x: (gr.Textbox(value=None, interactive=False), gr.Textbox(value=None, interactive=False),
                   gr.Textbox(value=None, interactive=False)) if x == "None" else (
            gr.Textbox(value=None, interactive=False), gr.Textbox(value=None, interactive=False),
            gr.Textbox(value=None, interactive=False)),
        tab4_extend_query_radio, [tab4_extended_query1_text, tab4_extended_query2_text, tab4_extended_query3_text])
    tab4_doc_ids_all_checkbox_group.change(
        lambda x: gr.CheckboxGroup(interactive=False, value="") if x else gr.CheckboxGroup(
            interactive=True, value=""),
        tab4_doc_ids_all_checkbox_group, tab4_doc_ids_checkbox_group)
    tab4_chat_document_button.click(chat_document_stream,
                                    inputs=[
                                        tab4_similarity_top_k_slider,
                                        tab4_similarity_threshold_slider,
                                        tab4_extend_first_n_chunk_slider,
                                        tab4_extend_around_n_chunk_size,
                                        tab4_use_rerank_checkbox,
                                        tab4_rerank_top_k_slider, tab4_rerank_threshold_slider,
                                        tab4_extend_query_radio,
                                        tab4_extended_query1_text, tab4_extended_query2_text,
                                        tab4_extended_query3_text, tab4_query_text, tab4_doc_ids_all_checkbox_group,
                                        tab4_doc_ids_checkbox_group
                                    ],
                                    outputs=[tab4_result_summary_text, tab4_result_dataframe, tab4_answer_text,
                                             tab4_extended_query1_text, tab4_extended_query2_text,
                                             tab4_extended_query3_text])

app.queue()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        max_threads=200,
    )
