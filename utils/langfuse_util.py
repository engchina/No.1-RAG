"""
Langfuseユーティリティモジュール

このモジュールは、Langfuseサービスの可用性チェック、安全なハンドラー作成、
ストリーミング設定の管理など、Langfuse関連の機能を提供します。
"""

import logging
import os
from urllib.parse import urljoin

import requests
from langfuse.callback import CallbackHandler

# ログ設定
logger = logging.getLogger(__name__)


def check_langfuse_availability():
    """
    Langfuse サービスの可用性を事前に確認する

    Returns:
        bool: Langfuse サービスが利用可能な場合は True、そうでなければ False
    """
    try:
        # 環境変数の存在確認
        required_env_vars = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"]
        for var in required_env_vars:
            if not os.environ.get(var):
                logger.warning(f"Langfuse 環境変数 {var} が設定されていません")
                return False

        # Langfuse クライアントの初期化テスト
        host = os.environ["LANGFUSE_HOST"].rstrip('/')

        # ヘルスチェックエンドポイントを試行
        health_url = urljoin(host, "/api/public/health")

        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info("Langfuse サービスが利用可能です")
                return True
            else:
                logger.warning(f"Langfuse ヘルスチェック失敗: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Langfuse サービスへの接続に失敗しました: {e}")
            return False

    except Exception as e:
        logger.warning(f"Langfuse 可用性チェック中にエラーが発生しました: {e}")
        return False


def create_safe_langfuse_handler():
    """
    安全なlangfuse handlerを作成する
    エラーが発生してもstream処理を中断しないようにする

    Returns:
        CallbackHandler or None: 正常に作成できた場合はCallbackHandler、エラーの場合はNone
    """
    try:
        # 事前にLangfuseサービスの可用性をチェック
        if not check_langfuse_availability():
            logger.warning("Langfuse サービスが利用できないため、callback を無効にします")
            return None

        return CallbackHandler(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=os.environ["LANGFUSE_HOST"],
        )
    except Exception as e:
        logger.warning(f"Langfuse handlerの作成に失敗しました: {e}")
        return None


def get_safe_stream_config(model_name=None):
    """
    安全なstream設定を取得する
    langfuse handlerが利用できない場合は空の設定を返す

    Args:
        model_name: モデル名（メタデータ用）

    Returns:
        dict: stream設定
    """
    try:
        langfuse_handler = create_safe_langfuse_handler()
        if langfuse_handler is None:
            logger.info(f"Langfuse が利用できないため、{model_name} のストリーミングは callback なしで実行されます")
            return {}

        config = {"callbacks": [langfuse_handler]}
        if model_name:
            config["metadata"] = {"ls_model_name": model_name}
        logger.info(f"Langfuse callback が有効になりました: {model_name}")
        return config
    except Exception as e:
        logger.warning(f"Stream設定の作成に失敗しました: {e}")
        return {}


def validate_langfuse_config():
    """
    Langfuse設定の妥当性を検証する

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        # 必要な環境変数の確認
        required_vars = ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"]
        missing_vars = []

        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)

        if missing_vars:
            error_msg = f"必要な環境変数が設定されていません: {', '.join(missing_vars)}"
            return False, error_msg

        # ホストURLの形式チェック
        host = os.environ["LANGFUSE_HOST"]
        if not (host.startswith("http://") or host.startswith("https://")):
            return False, "LANGFUSE_HOSTはhttp://またはhttps://で始まる必要があります"

        # キーの長さチェック（基本的な妥当性）
        secret_key = os.environ["LANGFUSE_SECRET_KEY"]
        public_key = os.environ["LANGFUSE_PUBLIC_KEY"]

        if len(secret_key) < 10:
            return False, "LANGFUSE_SECRET_KEYが短すぎます"

        if len(public_key) < 10:
            return False, "LANGFUSE_PUBLIC_KEYが短すぎます"

        return True, None

    except Exception as e:
        return False, f"設定検証中にエラーが発生しました: {e}"


def get_langfuse_status():
    """
    Langfuseの現在の状態を取得する

    Returns:
        dict: Langfuseの状態情報
    """
    status = {
        "config_valid": False,
        "service_available": False,
        "handler_ready": False,
        "error_message": None
    }

    try:
        # 設定の妥当性チェック
        config_valid, config_error = validate_langfuse_config()
        status["config_valid"] = config_valid

        if not config_valid:
            status["error_message"] = config_error
            return status

        # サービスの可用性チェック
        status["service_available"] = check_langfuse_availability()

        # ハンドラーの作成テスト
        handler = create_safe_langfuse_handler()
        status["handler_ready"] = handler is not None

        if status["config_valid"] and status["service_available"] and status["handler_ready"]:
            status["error_message"] = None
        elif not status["service_available"]:
            status["error_message"] = "Langfuseサービスに接続できません"
        elif not status["handler_ready"]:
            status["error_message"] = "Langfuseハンドラーの作成に失敗しました"

    except Exception as e:
        status["error_message"] = f"状態取得中にエラーが発生しました: {e}"

    return status
