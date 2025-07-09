"""
システム設定とチェックユーティリティモジュール

このモジュールは、Langfuse、データベース接続プール、その他のシステムコンポーネントの
設定と健康状態チェックを行うための関数を提供します。
"""

import logging

# ログ設定
logger = logging.getLogger(__name__)


# Langfuse関連の関数は utils/langfuse_util.py に移動されました:
# - check_langfuse_availability
# - create_safe_langfuse_handler
# - get_safe_stream_config


def check_database_pool_health(pool):
    """
    データベース接続プールの健康状態をチェックする

    Args:
        pool: データベース接続プール

    Returns:
        bool: プールが正常な場合True、問題がある場合False
    """
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM DUAL")
                result = cursor.fetchone()
                return result is not None
    except Exception as e:
        logger.error(f"データベース接続プールの健康チェックに失敗しました: {e}")
        return False
