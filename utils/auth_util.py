"""
認証ユーティリティモジュール

このモジュールは、OCI、OpenAI-Compatible、Langfuseなどの
各種サービスの認証情報を設定するための関数を提供します。
"""

import json
import os
import re
import shutil
from pathlib import Path

import gradio as gr
import oracledb
from dotenv import find_dotenv, load_dotenv, set_key, unset_key
from oracledb import DatabaseError

from .common_util import get_region


def create_oci_cred(user_ocid, tenancy_ocid, fingerprint, private_key_file, region, pool=None):
    """
    OCI認証情報を設定する

    Args:
        user_ocid: ユーザーOCID
        tenancy_ocid: テナンシーOCID
        fingerprint: フィンガープリント
        private_key_file: 秘密鍖ファイル
        region: リージョン
        pool: データベース接続プール

    Returns:
        tuple: (Accordion, Textbox) のタプル
    """

    def process_private_key(private_key_file_path):
        with open(private_key_file_path, 'r') as file:
            lines = file.readlines()

        processed_key = ''.join(line.strip() for line in lines if not line.startswith('--'))
        return processed_key

    has_error = False
    if not user_ocid:
        has_error = True
        gr.Warning("User OCIDを入力してください")
    if not tenancy_ocid:
        has_error = True
        gr.Warning("Tenancy OCIDを入力してください")
    if not fingerprint:
        has_error = True
        gr.Warning("Fingerprintを入力してください")
    if not private_key_file:
        has_error = True
        gr.Warning("Private Keyを入力してください")
    if not region:
        has_error = True
        gr.Warning("Regionを選択してください")

    if has_error:
        return gr.Accordion(), gr.Textbox()

    user_ocid = user_ocid.strip()
    tenancy_ocid = tenancy_ocid.strip()
    fingerprint = fingerprint.strip()
    region = region.strip()

    # set up OCI config
    BASE_DIR = Path(__file__).resolve().parent.parent
    if not os.path.exists("/root/.oci"):
        os.makedirs("/root/.oci")
    if not os.path.exists("/root/.oci/config"):
        config_src = BASE_DIR / ".oci" / "config"
        shutil.copy(str(config_src), "/root/.oci/config")
    oci_config_path = find_dotenv("/root/.oci/config")
    key_file_path = '/root/.oci/oci_api_key.pem'
    set_key(oci_config_path, "user", user_ocid, quote_mode="never")
    set_key(oci_config_path, "tenancy", tenancy_ocid, quote_mode="never")
    set_key(oci_config_path, "region", region, quote_mode="never")
    set_key(oci_config_path, "fingerprint", fingerprint, quote_mode="never")
    set_key(oci_config_path, "key_file", key_file_path, quote_mode="never")
    shutil.copy(private_key_file.name, key_file_path)
    load_dotenv(oci_config_path)

    # set up OCI Credential on database
    private_key = process_private_key(private_key_file.name)

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            try:
                # Define the PL/SQL statement
                append_acl_sql = """
    BEGIN
      DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
        host => '*',
        ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                           principal_name => 'admin',
                           principal_type => xs_acl.ptype_db));
    END;
                    """

                # Execute the PL/SQL statement
                cursor.execute(append_acl_sql)
            except DatabaseError as de:
                print(f"DatabaseError={de}")

            try:
                acl_genai_sql = f"""
    BEGIN
      DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
        host => 'inference.generativeai.{region}.oci.oraclecloud.com',
        ace  => xs$ace_type(privilege_list => xs$name_list('http'),
                            principal_name => 'admin',
                            principal_type => xs_acl.ptype_db));
    END;
                """
                cursor.execute(acl_genai_sql)
            except DatabaseError as de:
                print(f"DatabaseError={de}")

            try:
                drop_oci_cred_sql = "BEGIN dbms_vector.drop_credential('OCI_CRED'); END;"
                cursor.execute(drop_oci_cred_sql)
            except DatabaseError as de:
                print(f"DatabaseError={de}")

            oci_cred = {
                'user_ocid': user_ocid,
                'tenancy_ocid': tenancy_ocid,
                'compartment_ocid': os.environ["OCI_COMPARTMENT_OCID"],
                'private_key': private_key.strip(),
                'fingerprint': fingerprint
            }

            create_oci_cred_sql = """
    BEGIN
       dbms_vector.create_credential(
           credential_name => 'OCI_CRED',
           params => json(:json_params)
       );
    END; """

            cursor.execute(create_oci_cred_sql, json_params=json.dumps(oci_cred))
            conn.commit()

    create_oci_cred_sql = f"""
    -- Append Host ACE
    BEGIN
      DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
        host => '*',
        ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                           principal_name => 'admin',
                           principal_type => xs_acl.ptype_db));
    END;

    -- Append OCI GenAI Host ACE
    BEGIN
      DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
        host => 'inference.generativeai.{region}.oci.oraclecloud.com',
        ace  => xs$ace_type(privilege_list => xs$name_list('http'),
                            principal_name => 'admin',
                            principal_type => xs_acl.ptype_db));
    END;

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
    gr.Info("OCI API Keyの設定が完了しました")
    return gr.Accordion(), gr.Textbox(value=create_oci_cred_sql.strip())


def create_openai_cred(openai_cred_base_url, openai_cred_api_key, openai_cred_model, openai_cred_project):
    """
    OpenAI-Compatible認証情報を設定する

    Args:
        openai_cred_base_url: OpenAI-Compatible Base URL
        openai_cred_api_key: OpenAI-Compatible API Key
        openai_cred_model: OpenAI-Compatible Model
        openai_cred_project: OpenAI-Compatible Project

    Returns:
        tuple: (Base URL Textbox, API Key Textbox, Model Textbox, Project Textbox) のタプル
    """
    has_error = False
    if not openai_cred_base_url:
        has_error = True
        gr.Warning("OpenAI-Compatible Base URLを入力してください")
    if not openai_cred_api_key:
        has_error = True
        gr.Warning("OpenAI-Compatible API Keyを入力してください")
    if not openai_cred_model:
        has_error = True
        gr.Warning("OpenAI-Compatible Modelを入力してください")
    if has_error:
        return gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.Textbox()
    openai_cred_base_url = openai_cred_base_url.strip()
    openai_cred_api_key = openai_cred_api_key.strip()
    openai_cred_model = openai_cred_model.strip()
    openai_cred_project = (openai_cred_project or "").strip()
    env_path = find_dotenv()
    os.environ["OPENAI_BASE_URL"] = openai_cred_base_url
    os.environ["OPENAI_API_KEY"] = openai_cred_api_key
    os.environ["OPENAI_MODEL"] = openai_cred_model
    set_key(env_path, "OPENAI_BASE_URL", openai_cred_base_url, quote_mode="never")
    set_key(env_path, "OPENAI_API_KEY", openai_cred_api_key, quote_mode="never")
    set_key(env_path, "OPENAI_MODEL", openai_cred_model, quote_mode="never")
    if openai_cred_project:
        os.environ["OPENAI_PROJECT"] = openai_cred_project
        set_key(env_path, "OPENAI_PROJECT", openai_cred_project, quote_mode="never")
    else:
        os.environ.pop("OPENAI_PROJECT", None)
        unset_key(env_path, "OPENAI_PROJECT")
    load_dotenv(env_path)
    gr.Info("OpenAI-Compatibleの設定が完了しました")
    return (
        gr.Textbox(value=openai_cred_base_url),
        gr.Textbox(value=openai_cred_api_key),
        gr.Textbox(value=openai_cred_model),
        gr.Textbox(value=openai_cred_project)
    )


def create_langfuse_cred(langfuse_cred_secret_key, langfuse_cred_public_key, langfuse_cred_host):
    """
    Langfuse認証情報を設定する

    Args:
        langfuse_cred_secret_key: Langfuse Secret Key
        langfuse_cred_public_key: Langfuse Public Key
        langfuse_cred_host: Langfuse Host

    Returns:
        tuple: (Secret Key Textbox, Public Key Textbox, Host Textbox) のタプル
    """
    has_error = False
    if not langfuse_cred_secret_key:
        has_error = True
        gr.Warning("Langfuse Secret Keyを入力してください")
    if not langfuse_cred_public_key:
        has_error = True
        gr.Warning("Langfuse Public Keyを入力してください")
    if not langfuse_cred_host:
        has_error = True
        gr.Warning("Langfuse Hostを入力してください")
    if has_error:
        return gr.Textbox(), gr.Textbox(), gr.Textbox()
    langfuse_cred_secret_key = langfuse_cred_secret_key.strip()
    langfuse_cred_public_key = langfuse_cred_public_key.strip()
    langfuse_cred_host = langfuse_cred_host.strip()
    env_path = find_dotenv()
    os.environ["LANGFUSE_SECRET_KEY"] = langfuse_cred_secret_key
    os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_cred_public_key
    os.environ["LANGFUSE_HOST"] = langfuse_cred_host
    set_key(env_path, "LANGFUSE_SECRET_KEY", langfuse_cred_secret_key, quote_mode="never")
    set_key(env_path, "LANGFUSE_PUBLIC_KEY", langfuse_cred_public_key, quote_mode="never")
    set_key(env_path, "LANGFUSE_HOST", langfuse_cred_host, quote_mode="never")
    load_dotenv(env_path)
    gr.Info("Langfuse API Keyの設定が完了しました")
    return (
        gr.Textbox(value=langfuse_cred_secret_key),
        gr.Textbox(value=langfuse_cred_public_key),
        gr.Textbox(value=langfuse_cred_host)
    )


def do_auth(username, password):
    """
    データベース接続文字列を使用してユーザー認証を行う

    Args:
        username: ユーザー名
        password: パスワード

    Returns:
        bool: 認証が成功した場合True、失敗した場合False
    """
    dsn = os.environ["ORACLE_23AI_CONNECTION_STRING"]
    pattern = r"^([^/]+)/([^@]+)@"
    match = re.match(pattern, dsn)

    if match:
        if username.lower() == match.group(1).lower() and password == match.group(2):
            return True
    return False


def test_oci_cred(test_query_text, pool):
    """
    OCI認証情報をテストする

    Args:
        test_query_text: テスト用のクエリテキスト
        pool: データベース接続プール

    Returns:
        gr.Textbox: ベクトル結果を含むTextbox
    """
    test_query_vector = ""
    region = get_region()
    embed_genai_params = {
        "provider": "ocigenai",
        "credential_name": "OCI_CRED",
        "url": f"https://inference.generativeai.{region}.oci.oraclecloud.com/20231130/actions/embedText",
        "model": "cohere.embed-v4.0"
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
END; """

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
