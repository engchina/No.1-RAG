import os
from langchain_community.embeddings import OCIGenAIEmbeddings
from oracledb import DatabaseError
from utils.common_util import get_region


def create_table(pool, default_collection_name):
    """
    Create database tables and indexes for RAG system.

    Args:
        pool: Oracle database connection pool
        default_collection_name: Name of the default collection

    Returns:
        str: SQL text output for reference
    """
    # Get embedding function and dimension first
    region = get_region()
    embed = OCIGenAIEmbeddings(
        model_id=os.environ["OCI_COHERE_EMBED_MODEL"],
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
        compartment_id=os.environ["OCI_COMPARTMENT_OCID"]
    )

    # Get embedding dimension by creating a test embedding
    test_embedding = embed.embed_query("test")
    embedding_dim = len(test_embedding)

    # Initialize output_sql_text
    output_sql_text = ""

    # SQL statements for RAG QA tables
    drop_rag_qa_result_sql = "DROP TABLE IF EXISTS rag_qa_result"

    create_rag_qa_result_sql = """CREATE TABLE IF NOT EXISTS rag_qa_result (
        id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        query_id VARCHAR2(100),
        query VARCHAR2(4000),
        standard_answer VARCHAR2(30000),
        sql CLOB,
        created_date TIMESTAMP DEFAULT TO_TIMESTAMP(
            TO_CHAR(SYSTIMESTAMP, 'YYYY-MM-DD HH24:MI:SS'),
            'YYYY-MM-DD HH24:MI:SS'
        )
    )"""

    drop_rag_qa_feedback_sql = "DROP TABLE IF EXISTS rag_qa_feedback"

    create_rag_qa_feedback_sql = """CREATE TABLE IF NOT EXISTS rag_qa_feedback (
        id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        query_id VARCHAR2(100),
        llm_name VARCHAR2(100),
        llm_answer CLOB,
        vlm_answer CLOB,
        ragas_evaluation_result CLOB,
        human_evaluation_result VARCHAR2(20),
        user_comment VARCHAR2(30000),
        created_date TIMESTAMP DEFAULT TO_TIMESTAMP(
            TO_CHAR(SYSTIMESTAMP, 'YYYY-MM-DD HH24:MI:SS'),
            'YYYY-MM-DD HH24:MI:SS'
        )
    )"""

    # Index creation SQL statements
    create_index_sql = f"""CREATE INDEX {default_collection_name}_embed_data_idx
ON {default_collection_name}_embedding(embed_data)
INDEXTYPE IS CTXSYS.CONTEXT
PARAMETERS ('LEXER world_lexer sync (every "freq=minutely; interval=1")')"""

    create_image_index_sql = f"""CREATE INDEX {default_collection_name}_image_embed_data_idx
ON {default_collection_name}_image_embedding(embed_data)
INDEXTYPE IS CTXSYS.CONTEXT
PARAMETERS ('LEXER world_lexer sync (every "freq=minutely; interval=1")')"""

    # Drop table SQL statements
    drop_collection_table_sql = f"DROP TABLE IF EXISTS {default_collection_name}_collection PURGE"
    drop_embedding_table_sql = f"DROP TABLE IF EXISTS {default_collection_name}_embedding PURGE"
    drop_image_table_sql = f"DROP TABLE IF EXISTS {default_collection_name}_image PURGE"
    drop_image_embedding_table_sql = f"DROP TABLE IF EXISTS {default_collection_name}_image_embedding PURGE"

    # Create table SQL statements
    create_collection_table_sql = f"""CREATE TABLE IF NOT EXISTS {default_collection_name}_collection (
        id VARCHAR2(200),
        data BLOB,
        cmetadata CLOB
    )"""

    create_embedding_table_sql = f"""CREATE TABLE IF NOT EXISTS {default_collection_name}_embedding (
        doc_id VARCHAR2(200),
        embed_id NUMBER,
        embed_data VARCHAR2(4000),
        embed_vector VECTOR({embedding_dim}, FLOAT32),
        cmetadata CLOB
    )"""

    create_image_table_sql = f"""CREATE TABLE IF NOT EXISTS {default_collection_name}_image (
        doc_id VARCHAR2(200),
        img_id NUMBER,
        text_data CLOB,
        vlm_data CLOB,
        base64_data CLOB
    )"""

    create_image_embedding_table_sql = f"""CREATE TABLE IF NOT EXISTS {default_collection_name}_image_embedding (
        doc_id VARCHAR2(200),
        embed_id NUMBER,
        embed_data VARCHAR2(4000),
        embed_vector VECTOR({embedding_dim}, FLOAT32),
        cmetadata CLOB,
        img_id NUMBER
    )"""

    # Preference management SQL statements
    check_preference_sql = """SELECT PRE_NAME
FROM CTX_PREFERENCES
WHERE PRE_NAME = 'WORLD_LEXER'
    AND PRE_OWNER = USER"""

    drop_preference_plsql = """BEGIN
    CTX_DDL.DROP_PREFERENCE('world_lexer');
END;"""

    create_preference_plsql = """BEGIN
    CTX_DDL.CREATE_PREFERENCE('world_lexer','WORLD_LEXER');
END;"""

    # Build SQL output text for reference
    output_sql_text = _build_sql_output_text(
        default_collection_name,
        drop_collection_table_sql,
        drop_embedding_table_sql,
        drop_image_table_sql,
        drop_image_embedding_table_sql,
        drop_rag_qa_result_sql,
        drop_rag_qa_feedback_sql,
        create_collection_table_sql,
        create_embedding_table_sql,
        create_image_table_sql,
        create_image_embedding_table_sql,
        create_index_sql,
        create_image_index_sql,
        create_rag_qa_result_sql,
        create_rag_qa_feedback_sql,
        drop_preference_plsql,
        create_preference_plsql
    )

    # Execute database operations
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            # Drop indexes first
            _drop_indexes(cursor, default_collection_name)

            # Drop all tables
            _drop_tables(
                cursor,
                default_collection_name,
                drop_collection_table_sql,
                drop_embedding_table_sql,
                drop_image_table_sql,
                drop_image_embedding_table_sql,
                drop_rag_qa_result_sql,
                drop_rag_qa_feedback_sql
            )

            # Create all tables
            _create_tables(
                cursor,
                default_collection_name,
                create_collection_table_sql,
                create_embedding_table_sql,
                create_image_table_sql,
                create_image_embedding_table_sql,
                create_rag_qa_result_sql,
                create_rag_qa_feedback_sql
            )

            # Handle preferences (must be done before creating indexes that depend on them)
            _handle_preferences(
                cursor,
                check_preference_sql,
                drop_preference_plsql,
                create_preference_plsql
            )

            # Create indexes (after preferences are created)
            _create_indexes(
                cursor,
                default_collection_name,
                create_index_sql,
                create_image_index_sql
            )

            conn.commit()

    return output_sql_text


def _build_sql_output_text(default_collection_name, *sql_statements):
    """Build formatted SQL output text for reference."""
    (drop_collection_table_sql, drop_embedding_table_sql, drop_image_table_sql,
     drop_image_embedding_table_sql, drop_rag_qa_result_sql, drop_rag_qa_feedback_sql,
     create_collection_table_sql, create_embedding_table_sql, create_image_table_sql,
     create_image_embedding_table_sql, create_index_sql, create_image_index_sql,
     create_rag_qa_result_sql, create_rag_qa_feedback_sql, drop_preference_plsql,
     create_preference_plsql) = sql_statements

    output_sql_text = f"""-- Drop Indexes
DROP INDEX IF EXISTS {default_collection_name}_embed_data_idx;
DROP INDEX IF EXISTS {default_collection_name}_image_embed_data_idx;

-- Drop All Tables
{drop_collection_table_sql};
{drop_embedding_table_sql};
{drop_image_table_sql};
{drop_image_embedding_table_sql};
{drop_rag_qa_result_sql};
{drop_rag_qa_feedback_sql};

-- Create All Tables
{create_collection_table_sql};
{create_embedding_table_sql};
{create_image_table_sql};
{create_image_embedding_table_sql};

-- Create RAG QA Tables
{create_rag_qa_result_sql};
{create_rag_qa_feedback_sql};

-- Handle Preferences (must be done before creating indexes)
{drop_preference_plsql}
{create_preference_plsql}

-- Create Indexes (after preferences are created)
{create_index_sql};
{create_image_index_sql};"""

    return output_sql_text


def _drop_indexes(cursor, default_collection_name):
    """Drop database indexes with existence check."""
    indexes = [
        f"{default_collection_name}_embed_data_idx",
        f"{default_collection_name}_image_embed_data_idx"
    ]

    for index_name in indexes:
        try:
            cursor.execute(
                f"SELECT COUNT(*) FROM USER_INDEXES WHERE INDEX_NAME = '{index_name.upper()}'"
            )
            if cursor.fetchone()[0] > 0:
                cursor.execute(f"DROP INDEX {index_name}")
                print(f"インデックス {index_name} を削除しました")
        except DatabaseError as e:
            print(f"インデックス {index_name} の削除エラー: {e}")


def _drop_tables(cursor, default_collection_name, *drop_sqls):
    """Drop database tables with existence check."""
    (drop_collection_table_sql, drop_embedding_table_sql, drop_image_table_sql,
     drop_image_embedding_table_sql, drop_rag_qa_result_sql, drop_rag_qa_feedback_sql) = drop_sqls

    tables = [
        (f"{default_collection_name}_collection", drop_collection_table_sql),
        (f"{default_collection_name}_embedding", drop_embedding_table_sql),
        (f"{default_collection_name}_image", drop_image_table_sql),
        (f"{default_collection_name}_image_embedding", drop_image_embedding_table_sql),
        ("rag_qa_result", drop_rag_qa_result_sql),
        ("rag_qa_feedback", drop_rag_qa_feedback_sql)
    ]

    for table_name, drop_sql in tables:
        try:
            if table_name.startswith("RAG_QA"):
                cursor.execute(drop_sql)
                print(f"テーブル {table_name} を削除しました")
            else:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                cursor.execute(drop_sql)
                print(f"テーブル {table_name} を削除しました")
        except DatabaseError as e:
            if e.args[0].code == 942:  # Table or view does not exist
                print(f"テーブル {table_name} は存在しません")
            else:
                print(f"テーブル {table_name} の削除エラー: {e}")


def _create_tables(cursor, default_collection_name, *create_sqls):
    """Create database tables."""
    (create_collection_table_sql, create_embedding_table_sql, create_image_table_sql,
     create_image_embedding_table_sql, create_rag_qa_result_sql, create_rag_qa_feedback_sql) = create_sqls

    tables = [
        (f"{default_collection_name}_collection", create_collection_table_sql),
        (f"{default_collection_name}_embedding", create_embedding_table_sql),
        (f"{default_collection_name}_image", create_image_table_sql),
        (f"{default_collection_name}_image_embedding", create_image_embedding_table_sql),
        ("rag_qa_result", create_rag_qa_result_sql),
        ("rag_qa_feedback", create_rag_qa_feedback_sql)
    ]

    for table_name, create_sql in tables:
        try:
            cursor.execute(create_sql)
            print(f"テーブル {table_name} を作成しました")
        except DatabaseError as e:
            print(f"テーブル {table_name} の作成エラー: {e}")


def _create_indexes(cursor, default_collection_name, create_index_sql, create_image_index_sql):
    """Create database indexes."""
    indexes = [
        (f"{default_collection_name}_embed_data_idx", create_index_sql),
        (f"{default_collection_name}_image_embed_data_idx", create_image_index_sql)
    ]

    for index_name, create_sql in indexes:
        try:
            cursor.execute(create_sql)
            print(f"インデックス {index_name} を作成しました")
        except DatabaseError as e:
            print(f"インデックス {index_name} の作成エラー: {e}")


def _handle_preferences(cursor, check_preference_sql, drop_preference_plsql, create_preference_plsql):
    """Handle Oracle Text preferences."""
    try:
        cursor.execute(check_preference_sql)
        if cursor.fetchone():
            cursor.execute(drop_preference_plsql)
            print("Preference 'WORLD_LEXER' を削除しました")
        else:
            print("Preference 'WORLD_LEXER' は存在しません")
    except DatabaseError as e:
        print(f"Preference 'WORLD_LEXER' の削除エラー: {e}")

    try:
        cursor.execute(create_preference_plsql)
        print("Preference 'WORLD_LEXER' を作成しました")
    except DatabaseError as e:
        print(f"Preference 'WORLD_LEXER' の作成エラー: {e}")
