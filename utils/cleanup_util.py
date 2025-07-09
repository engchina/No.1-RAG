"""
リソースクリーンアップユーティリティモジュール

このモジュールは、LLMクライアント、HTTP接続プール、その他のシステムリソースの
安全なクリーンアップを行うための関数を提供します。
"""

import asyncio
import gc
import warnings
import tracemalloc


def cleanup_llm_client(llm_client):
    """
    LLMクライアントのリソースを安全にクリーンアップする（同期版）

    Args:
        llm_client: クリーンアップするLLMクライアント
    """
    if llm_client is None:
        return

    try:
        # OpenAI系クライアントの場合
        if hasattr(llm_client, 'client'):
            if hasattr(llm_client.client, 'close'):
                # 同期的にクローズ
                if hasattr(llm_client.client.close, '__call__'):
                    try:
                        llm_client.client.close()
                    except Exception as e:
                        print(f"OpenAI クライアントの同期クローズ中にエラー: {e}")

        # _clientアトリビュートを持つ場合
        elif hasattr(llm_client, '_client'):
            if hasattr(llm_client._client, 'close'):
                try:
                    llm_client._client.close()
                except Exception as e:
                    print(f"_client クローズ中にエラー: {e}")

            # OCI GenAI系クライアントのセッション処理
            if hasattr(llm_client._client, '_session'):
                if hasattr(llm_client._client._session, 'close'):
                    try:
                        llm_client._client._session.close()
                    except Exception as e:
                        print(f"OCI セッションクローズ中にエラー: {e}")

        print(f"LLMクライアント {type(llm_client).__name__} のリソースクリーンアップが完了しました")

    except Exception as cleanup_error:
        print(f"LLMクライアントのクリーンアップ中に予期しないエラーが発生しました: {cleanup_error}")


async def cleanup_llm_client_async(llm_client):
    """
    LLMクライアントのリソースを安全にクリーンアップする（非同期版）

    Args:
        llm_client: クリーンアップするLLMクライアント
    """
    if llm_client is None:
        return

    try:
        # OpenAI系クライアントの場合（非同期クローズ）
        if hasattr(llm_client, 'client'):
            # HTTPXクライアントの場合
            if hasattr(llm_client.client, 'aclose'):
                try:
                    await llm_client.client.aclose()
                    print(f"OpenAI HTTPXクライアントの非同期クローズが完了しました")
                except Exception as e:
                    # OpenAI API type エラーは無視
                    if "Ambiguous use of module client" not in str(e):
                        print(f"OpenAI クライアントの非同期クローズ中にエラー: {e}")
            elif hasattr(llm_client.client, 'close'):
                try:
                    llm_client.client.close()
                    print(f"OpenAI クライアントの同期クローズが完了しました")
                except Exception as e:
                    # OpenAI API type エラーは無視
                    if "Ambiguous use of module client" not in str(e):
                        print(f"OpenAI クライアントの同期クローズ中にエラー: {e}")

            # 追加のHTTP接続プール清理
            if hasattr(llm_client.client, '_client'):
                if hasattr(llm_client.client._client, 'aclose'):
                    try:
                        await llm_client.client._client.aclose()
                        print(f"OpenAI 内部HTTPクライアントの非同期クローズが完了しました")
                    except Exception as e:
                        if "Ambiguous use of module client" not in str(e):
                            print(f"OpenAI 内部HTTPクライアントクローズ中にエラー: {e}")

        # _clientアトリビュートを持つ場合（OCI GenAI等）
        elif hasattr(llm_client, '_client'):
            if hasattr(llm_client._client, 'aclose'):
                try:
                    await llm_client._client.aclose()
                    print(f"OCI _client の非同期クローズが完了しました")
                except Exception as e:
                    print(f"_client 非同期クローズ中にエラー: {e}")
            elif hasattr(llm_client._client, 'close'):
                try:
                    llm_client._client.close()
                    print(f"OCI _client の同期クローズが完了しました")
                except Exception as e:
                    print(f"_client クローズ中にエラー: {e}")

            # OCI GenAI系クライアントのセッション処理
            if hasattr(llm_client._client, '_session'):
                if hasattr(llm_client._client._session, 'aclose'):
                    try:
                        await llm_client._client._session.aclose()
                        print(f"OCI セッションの非同期クローズが完了しました")
                    except Exception as e:
                        print(f"OCI セッション非同期クローズ中にエラー: {e}")
                elif hasattr(llm_client._client._session, 'close'):
                    try:
                        llm_client._client._session.close()
                        print(f"OCI セッションの同期クローズが完了しました")
                    except Exception as e:
                        print(f"OCI セッションクローズ中にエラー: {e}")

        # 追加のリソース清理：HTTPコネクションプールの強制クリーンアップ（軽量版）
        try:
            await force_cleanup_http_connections(llm_client)
        except Exception as force_cleanup_error:
            # 強制クリーンアップのエラーは詳細を表示しない
            if "Ambiguous use of module client" not in str(force_cleanup_error):
                print(f"HTTP接続プール強制クリーンアップ中にエラー: {force_cleanup_error}")

        print(f"LLMクライアント {type(llm_client).__name__} の非同期リソースクリーンアップが完了しました")

    except Exception as cleanup_error:
        # OpenAI API type エラーは無視
        if "Ambiguous use of module client" not in str(cleanup_error):
            print(f"LLMクライアントの非同期クリーンアップ中に予期しないエラーが発生しました: {cleanup_error}")


async def force_cleanup_http_connections(llm_client):
    """
    HTTPコネクションプールを強制的にクリーンアップする

    Args:
        llm_client: LLMクライアント
    """
    try:
        # httpxライブラリのコネクションプールを探してクリーンアップ
        import httpx

        # ガベージコレクションを実行してオブジェクトを収集
        gc.collect()

        # httpxクライアントを探してクリーンアップ（最大3つまで）
        cleaned_count = 0
        for obj in gc.get_objects():
            if cleaned_count >= 3:  # 処理数を制限
                break

            if isinstance(obj, httpx.AsyncClient):
                try:
                    if not obj.is_closed:
                        await obj.aclose()
                        cleaned_count += 1
                        print(f"未閉鎖のHTTPXクライアント #{cleaned_count} をクリーンアップしました")
                except Exception as e:
                    # OpenAI API type エラーは無視
                    if "Ambiguous use of module client" not in str(e):
                        print(f"HTTPXクライアントクリーンアップ中にエラー: {e}")
            elif isinstance(obj, httpx.Client):
                try:
                    if not obj.is_closed:
                        obj.close()
                        cleaned_count += 1
                        print(f"未閉鎖の同期HTTPXクライアント #{cleaned_count} をクリーンアップしました")
                except Exception as e:
                    # OpenAI API type エラーは無視
                    if "Ambiguous use of module client" not in str(e):
                        print(f"同期HTTPXクライアントクリーンアップ中にエラー: {e}")

    except ImportError:
        # httpxがインストールされていない場合は無視
        pass
    except Exception as e:
        # OpenAI API type エラーは無視
        if "Ambiguous use of module client" not in str(e):
            print(f"HTTP接続プール強制クリーンアップ中にエラー: {e}")


async def cleanup_all_http_connections():
    """
    システム全体のHTTP接続プールをクリーンアップする
    """
    try:
        import httpx
        import aiohttp

        print("システム全体のHTTP接続プールクリーンアップを開始...")

        # ガベージコレクションを実行
        gc.collect()

        cleaned_count = 0

        # すべてのオブジェクトをスキャンしてHTTPクライアントを探す
        for obj in gc.get_objects():
            try:
                # httpxクライアント
                if isinstance(obj, httpx.AsyncClient):
                    if not obj.is_closed:
                        await obj.aclose()
                        cleaned_count += 1
                        print(f"HTTPXクライアント #{cleaned_count} をクリーンアップしました")
                elif isinstance(obj, httpx.Client):
                    if not obj.is_closed:
                        obj.close()
                        cleaned_count += 1
                        print(f"同期HTTPXクライアント #{cleaned_count} をクリーンアップしました")

                # aiohttpクライアント
                elif hasattr(obj, '__class__') and 'aiohttp' in str(obj.__class__):
                    if hasattr(obj, 'close') and not getattr(obj, 'closed', True):
                        try:
                            await obj.close()
                            cleaned_count += 1
                            print(f"aiohttpクライアント #{cleaned_count} をクリーンアップしました")
                        except Exception:
                            pass

            except Exception as e:
                # 個別のオブジェクトクリーンアップエラーは無視
                pass

        # 最終ガベージコレクション
        gc.collect()

        print(f"HTTP接続プールクリーンアップ完了: {cleaned_count}個のクライアントを処理しました")

    except ImportError:
        print("HTTP接続プールクリーンアップ: 必要なライブラリがインストールされていません")
    except Exception as e:
        print(f"システム全体のHTTP接続プールクリーンアップ中にエラー: {e}")


async def lightweight_cleanup():
    """
    軽量なリソースクリーンアップ（画像処理後に使用）
    """
    try:
        # 軽量なガベージコレクション
        collected = gc.collect()
        if collected > 0:
            print(f"軽量クリーンアップ: {collected} オブジェクトを回収しました")

        # 基本的なHTTP接続チェック（重い処理は避ける）
        try:
            import httpx
            # 明らかに閉じられていないクライアントのみチェック
            cleaned_count = 0
            for obj in gc.get_objects():
                if isinstance(obj, httpx.AsyncClient) and hasattr(obj, 'is_closed'):
                    if not obj.is_closed:
                        try:
                            await obj.aclose()
                            cleaned_count += 1
                            print(f"未閉鎖のHTTPXクライアント #{cleaned_count} をクリーンアップしました")
                        except Exception as close_error:
                            # OpenAI API type エラーなどは無視
                            if "Ambiguous use of module client" not in str(close_error):
                                print(f"HTTPXクライアントクローズ中にエラー: {close_error}")
                        if cleaned_count >= 3:  # 最大3つまで処理
                            break
        except ImportError:
            pass
        except Exception as http_error:
            # HTTP関連のエラーは詳細を表示しない
            print(f"HTTP接続チェック中にエラーが発生しました（無視されます）")

    except Exception as e:
        print(f"軽量クリーンアップ中にエラー: {e}")


def enable_resource_warnings():
    """
    リソース警告を有効にして詳細な情報を取得する
    """
    import warnings

    # ResourceWarningを有効にする
    warnings.filterwarnings("always", category=ResourceWarning)

    # tracemallocを有効にしてメモリ追跡を開始
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        print("リソース追跡が有効になりました")

    # カスタムwarning処理を設定
    def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
        if category == ResourceWarning:
            print(f"🚨 リソース警告: {message}")
            print(f"   ファイル: {filename}:{lineno}")
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                print(f"   メモリ使用量: 現在={current / 1024 / 1024:.1f}MB, ピーク={peak / 1024 / 1024:.1f}MB")

    # 警告ハンドラーを設定
    warnings.showwarning = custom_warning_handler


async def final_resource_cleanup():
    """
    プログラム終了時の最終リソースクリーンアップ
    """
    print("\n=== 最終リソースクリーンアップを開始 ===")

    try:
        # HTTP接続プールの全体クリーンアップ
        await cleanup_all_http_connections()

        # 追加のリソースクリーンアップ
        # 現在のタスクを取得（自分自身は除外）
        current_task = asyncio.current_task()
        tasks = [task for task in asyncio.all_tasks()
                 if not task.done() and task != current_task]

        if tasks:
            print(f"実行中のタスクを終了します: {len(tasks)}個")
            for task in tasks:
                try:
                    task.cancel()
                    await asyncio.sleep(0.1)  # タスクのキャンセルを待つ
                except Exception as task_error:
                    print(f"タスクキャンセル中にエラー: {task_error}")

        # 強制ガベージコレクション
        collected = gc.collect()
        print(f"ガベージコレクション: {collected} オブジェクトを回収しました")

        # メモリ統計を表示
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            print(f"最終メモリ使用量: 現在={current / 1024 / 1024:.1f}MB, ピーク={peak / 1024 / 1024:.1f}MB")
            tracemalloc.stop()

        print("=== 最終リソースクリーンアップ完了 ===\n")

    except Exception as e:
        print(f"最終リソースクリーンアップ中にエラー: {e}")
        import traceback
        traceback.print_exc()
