"""
画像処理関連の関数を提供するモジュール

このモジュールには以下の機能が含まれています：
- 単一画像ストリーミング処理 (process_single_image_streaming)
- 画像回答ストリーミング処理 (process_image_answers_streaming)
- 複数画像ストリーミング処理 (process_multiple_images_streaming)
"""

import asyncio
import os
import time

import gradio as gr
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from my_langchain_community.chat_models import ChatOCIGenAI
from utils.cleanup_util import lightweight_cleanup, cleanup_llm_client_async
from utils.common_util import get_region
from utils.image_util import compress_image_for_display
from utils.prompts_util import get_image_qa_prompt
from utils.system_util import check_database_pool_health


async def process_single_image_streaming(image_url, query_text, llm_answer_checkbox_group, target_models, image_index,
                                         doc_id, img_id, custom_image_prompt=None):
    """
    単一画像を選択されたLLMモデルで処理し、ストリーミング形式で回答を返す

    Args:
        image_url: 画像のURL
        query_text: クエリテキスト
        llm_answer_checkbox_group: 選択されたLLMモデルのリスト
        target_models: 対象モデルのリスト
        image_index: 画像のインデックス
        doc_id: ドキュメントID
        img_id: 画像ID
        custom_image_prompt: カスタム画像プロンプトテンプレート

    Yields:
        dict: 各モデルの部分的な回答を含む辞書
    """
    if custom_image_prompt:
        custom_image_prompt = custom_image_prompt.replace('{{query_text}}', '{query_text}')

    region = get_region()

    # 各モデルのタスクジェネレーターを作成
    async def create_model_task(model):
        llm = None  # LLMインスタンスを初期化
        try:
            if model not in llm_answer_checkbox_group:
                # 選択されていないモデルは即座に完了を通知
                yield "TASK_DONE"
                return

            print(f"=== {model} での処理開始 ===")
            start_time = time.time()

            # モデルに応じてLLMインスタンスを作成
            if model == "meta/llama-4-scout-17b-16e-instruct":
                llm = ChatOCIGenAI(
                    model_id="meta.llama-4-scout-17b-16e-instruct",
                    provider="meta",
                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                    compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
                    model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
                )
            elif model == "openai/gpt-4o":
                load_dotenv(find_dotenv())
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    top_p=0.75,
                    seed=42,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ["OPENAI_BASE_URL"],
                )
            elif model == "azure_openai/gpt-4o":
                load_dotenv(find_dotenv())
                llm = AzureChatOpenAI(
                    deployment_name="gpt-4o",
                    temperature=0,
                    top_p=0.75,
                    seed=42,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_GPT_4O"],
                    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_GPT_4O"],
                )
            else:
                print(f"未対応のモデル: {model}")
                yield "TASK_DONE"
                return

            # プロンプトを作成
            if custom_image_prompt:
                prompt_text = custom_image_prompt.format(query_text=query_text)
            else:
                prompt_text = get_image_qa_prompt(query_text)

            # メッセージを作成
            messages = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                )
            ]

            print(f"画像 {image_index + 1} (doc_id: {doc_id}, img_id: {img_id}) を {model} で処理中...")

            # 表示用に画像を圧縮
            compressed_image_url = compress_image_for_display(image_url)
            # ヘッダー情報を最初にyield（圧縮された画像を使用）
            header_text = f"\n\n---\n\n**画像 {image_index} (doc_id: {doc_id}, img_id: {img_id}) による回答：**\n\n![画像]({compressed_image_url})\n\n"
            yield header_text

            # ストリーミングで回答を取得
            # Avoid for: Error uploading media: HTTPConnectionPool(host='minio', port=9000)
            # async for chunk in llm.astream(messages, config={"callbacks": [langfuse_handler],
            #                                                  "metadata": {"ls_model_name": model}}):
            async for chunk in llm.astream(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    yield chunk.content

            end_time = time.time()
            inference_time = end_time - start_time
            print(f"\n\n推論時間: {inference_time:.2f}秒")
            print(f"=== {model} での処理完了 ===\n")

            # 推論時間を追加
            yield f"\n\n推論時間: {inference_time:.2f}秒\n\n"
            yield "TASK_DONE"

        except Exception as e:
            print(f"モデル {model} でエラーが発生しました: {e}")
            yield f"\n\nエラーが発生しました: {str(e)}\n\n"
            yield "TASK_DONE"
        finally:
            # LLMインスタンスのクリーンアップ
            if llm:
                try:
                    # 軽量なHTTP接続クリーンアップ
                    await lightweight_cleanup()
                except Exception as cleanup_error:
                    print(f"クリーンアップ中にエラーが発生しました: {cleanup_error}")

    # 各モデルのタスクを作成
    tasks = {}
    for model in target_models:
        if model in llm_answer_checkbox_group:
            tasks[model] = create_model_task(model)

    # 各モデルの現在の回答を追跡
    current_responses = {model: "" for model in target_models}
    completed_tasks = set()

    # すべてのタスクが完了するまでループ
    while len(completed_tasks) < len(tasks):
        for model, task_gen in tasks.items():
            if model in completed_tasks:
                continue

            try:
                # 各タスクから次のチャンクを取得
                chunk = await task_gen.__anext__()
                if chunk == "TASK_DONE":
                    completed_tasks.add(model)
                    print(f"タスク {model} が完了しました")
                else:
                    current_responses[model] += chunk

                # 現在の状態を辞書として返す
                yield current_responses.copy()

            except StopAsyncIteration:
                completed_tasks.add(model)
                print(f"タスク {model} が完了しました (StopAsyncIteration)")
            except Exception as e:
                print(f"タスク {model} でエラーが発生しました: {e}")
                completed_tasks.add(model)

        # 短い待機時間を追加してCPU使用率を下げる
        await asyncio.sleep(0.01)

    print("すべてのタスクが完了しました")

    # 軽量なHTTP接続クリーンアップ
    await lightweight_cleanup()

    # ガベージコレクションを強制実行してリソースを解放
    import gc
    gc.collect()
    print("単一画像処理のリソースクリーンアップが完了しました")


async def process_image_answers_streaming(
        pool,
        default_collection_name,
        search_result,
        use_image,
        single_image_processing,
        llm_answer_checkbox_group,
        query_text,
        llama_4_scout_image_answer_text,
        openai_gpt4o_image_answer_text,
        azure_openai_gpt4o_image_answer_text,
        image_limit_k=5,
        custom_image_prompt=None,
):
    """
    Vision 回答がオンの場合、検索結果から画像データを取得し、
    選択されたVisionモデルで画像処理を行い、ストリーミング形式で回答を出力する

    処理の流れ：
    1. 検索結果からdoc_idとembed_idのペアを抽出（search_resultの順序を保持）
    2. データベースから対応する画像のbase64データを取得（search_resultの相関性順序を維持）
    3. image_limit_kパラメータで上位k件に制限
    4. 取得した画像を各選択されたLLMモデルで並行処理
    5. ストリーミング形式で回答を出力

    注意：パフォーマンスと応答時間を考慮し、処理する画像数はimage_limit_kパラメータで制限されています。
    重要：search_resultの相関性順序（相関性の高い順）を保持して画像を処理します。

    Args:
        pool: データベース接続プール
        default_collection_name: デフォルトコレクション名
        search_result: 検索結果
        use_image: Vision 回答するかどうか
        single_image_processing: 画像を1枚ずつ処理するかどうか
        llm_answer_checkbox_group: 選択されたLLMモデルのリスト
        query_text: クエリテキスト
        llama_4_scout_image_answer_text: Llama 4 Scout のVision 回答テキスト
        openai_gpt4o_image_answer_text: OpenAI GPT-4o のVision 回答テキスト
        azure_openai_gpt4o_image_answer_text: Azure OpenAI GPT-4o のVision 回答テキスト
        image_limit_k: 処理する画像の最大数（1-10）
        custom_image_prompt: カスタム画像プロンプトテンプレート

    Yields:
        tuple: 各モデルの更新されたVision 回答を含むGradio Markdownコンポーネントのタプル
    """
    print("process_image_answers_streaming() 開始...")

    # データベース接続プールの健康状態をチェック
    if not check_database_pool_health(pool):
        print("データベース接続プールに問題があります")
        yield (
            gr.Markdown(value=llama_4_scout_image_answer_text),
            gr.Markdown(value=openai_gpt4o_image_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
        )
        return

    # Vision 回答がオフの場合は何もしない
    if not use_image:
        print("Vision 回答がオフのため、base64_data取得をスキップします")
        yield (
            gr.Markdown(value=llama_4_scout_image_answer_text),
            gr.Markdown(value=openai_gpt4o_image_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
        )
        return

    # 検索結果が空の場合は何もしない
    if search_result.empty or (len(search_result) > 0 and search_result.iloc[0]['CONTENT'] == ''):
        print("検索結果が空のため、base64_data取得をスキップします")
        yield (
            gr.Markdown(value=llama_4_scout_image_answer_text),
            gr.Markdown(value=openai_gpt4o_image_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
        )
        return

    # 指定されたLLMモデルがチェックされているかを確認
    target_models = [
        "meta/llama-4-scout-17b-16e-instruct",
        "openai/gpt-4o",
        "azure_openai/gpt-4o"
    ]

    # llm_answer_checkbox_groupに指定されたモデルのいずれかが含まれているかチェック
    has_target_model = any(model in llm_answer_checkbox_group for model in target_models)

    if not has_target_model:
        print(
            "対象のLLMモデル（llama-4-scout, gpt-4o）がチェックされていないため、base64_data取得をスキップします")
        yield (
            gr.Markdown(value=llama_4_scout_image_answer_text),
            gr.Markdown(value=openai_gpt4o_image_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
        )
        return

    print("条件を満たしているため、base64_dataを取得します...")

    try:
        # 検索結果からdoc_idとembed_idを取得（search_resultの順序を保持）
        doc_embed_pairs = []
        doc_embed_order_map = {}  # (doc_id, embed_id) -> search_result内の順序
        for idx, (_, row) in enumerate(search_result.iterrows()):
            source = row['SOURCE']
            embed_id = row['EMBED_ID']
            if ':' in source:
                doc_id = source.split(':')[0]
                doc_embed_pair = (doc_id, embed_id)
                if doc_embed_pair not in doc_embed_pairs:
                    doc_embed_pairs.append(doc_embed_pair)
                    doc_embed_order_map[doc_embed_pair] = idx  # search_result内の順序を記録

        if not doc_embed_pairs:
            print("検索結果からdoc_idとembed_idを取得できませんでした")
            yield (
                gr.Markdown(value=llama_4_scout_image_answer_text),
                gr.Markdown(value=openai_gpt4o_image_answer_text),
                gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
            )
            return

        print(f"取得したdoc_id, embed_idペア数: {len(doc_embed_pairs)}")
        print(f"最初の5ペア: {doc_embed_pairs[:5]}")

        # データベースからdistinct base64_dataを取得
        try:
            with pool.acquire() as conn:
                with conn.cursor() as cursor:
                    # まず_image_embeddingテーブルからimg_idを取得
                    embed_where_conditions = []
                    for doc_id, embed_id in doc_embed_pairs:
                        embed_where_conditions.append(f"(doc_id = '{doc_id}' AND embed_id = {embed_id})")

                    embed_where_clause = " OR ".join(embed_where_conditions)

                    # _image_embeddingテーブルからimg_idを取得
                    get_img_ids_sql = f"""
                        SELECT doc_id, embed_id, img_id
                        FROM (
                            SELECT
                                doc_id,
                                embed_id,
                                img_id,
                                ROW_NUMBER() OVER (PARTITION BY doc_id, img_id ORDER BY embed_id ASC) as rn
                            FROM {default_collection_name}_image_embedding
                            WHERE ({embed_where_clause})
                            AND img_id IS NOT NULL
                        ) subquery
                        WHERE rn = 1
                    """

                    print(f"img_id取得SQL: {get_img_ids_sql}")
                    cursor.execute(get_img_ids_sql)

                    # 結果を辞書に格納してsearch_resultの順序で並び替え
                    img_results = {}
                    for row in cursor:
                        doc_id = row[0]
                        embed_id = row[1]
                        img_id = row[2]
                        doc_embed_pair = (doc_id, embed_id)
                        if doc_embed_pair in doc_embed_order_map:
                            order_idx = doc_embed_order_map[doc_embed_pair]
                            if order_idx not in img_results:
                                img_results[order_idx] = []
                            img_results[order_idx].append((doc_id, img_id))
                            print(
                                f"見つかったペア: doc_id={doc_id}, embed_id={embed_id}, img_id={img_id}, order={order_idx}")

                    # search_resultの順序でdoc_img_pairsを構築
                    doc_img_pairs = []
                    for order_idx in sorted(img_results.keys()):
                        for doc_id, img_id in img_results[order_idx]:
                            doc_img_pair = (doc_id, img_id)
                            if doc_img_pair not in doc_img_pairs:
                                doc_img_pairs.append(doc_img_pair)

                    if not doc_img_pairs:
                        print("_image_embeddingテーブルからimg_idを取得できませんでした")
                        yield (
                            gr.Markdown(value=llama_4_scout_image_answer_text),
                            gr.Markdown(value=openai_gpt4o_image_answer_text),
                            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
                        )
                        return

                    print(f"取得したdoc_id, img_idペア数: {len(doc_img_pairs)}")

                    # image_limit_kを適用してdoc_img_pairsを制限
                    limited_doc_img_pairs = doc_img_pairs[:image_limit_k]
                    print(f"image_limit_k={image_limit_k}適用後のペア数: {len(limited_doc_img_pairs)}")

                    # 次に_imageテーブルからbase64_dataを取得
                    img_where_conditions = []
                    for doc_id, img_id in limited_doc_img_pairs:
                        img_where_conditions.append(f"(doc_id = '{doc_id}' AND img_id = {img_id})")

                    if not img_where_conditions:
                        print("処理対象の画像がありません")
                        yield (
                            gr.Markdown(value=llama_4_scout_image_answer_text),
                            gr.Markdown(value=openai_gpt4o_image_answer_text),
                            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
                        )
                        return

                    img_where_clause = " OR ".join(img_where_conditions)

                    # base64画像データを取得（search_resultの順序を保持）
                    select_sql = f"""
                    SELECT base64_data, doc_id, img_id
                    FROM {default_collection_name}_image
                    WHERE ({img_where_clause})
                    AND base64_data IS NOT NULL
                    """

                    print(f"実行するSQL: {select_sql}")
                    cursor.execute(select_sql)

                    # 結果を辞書に格納
                    image_data_dict = {}
                    for row in cursor:
                        if row[0] is not None:
                            try:
                                # CLOBオブジェクトの読み取り
                                if hasattr(row[0], 'read'):
                                    base64_string = row[0].read()
                                    # 10MB制限チェック
                                    if len(base64_string) > 10 * 1024 * 1024:
                                        print(f"Base64データが大きすぎます（{len(base64_string)}文字）、スキップします")
                                        continue
                                else:
                                    base64_string = str(row[0])

                                if row[2] is not None:
                                    # img_idはNUMBER型なので、直接使用するか安全に変換
                                    img_id = row[2] if isinstance(row[2], (int, float)) else int(row[2])
                                    doc_id = row[1]
                                    image_data_dict[(doc_id, img_id)] = (base64_string, doc_id, img_id)

                            except Exception as e:
                                print(f"CLOB読み取りまたはimg_id処理エラー: {e}")
                                continue

                    # search_resultの順序でbase64_data_listを構築
                    base64_data_list = []
                    for doc_id, img_id in limited_doc_img_pairs:
                        if (doc_id, img_id) in image_data_dict:
                            base64_data_list.append(image_data_dict[(doc_id, img_id)])
                            print(f"search_result順序で画像を追加: doc_id={doc_id}, img_id={img_id}")

                    print(f"取得したbase64_dataの数: {len(base64_data_list)} (search_result順序を保持)")

                    # 初期化：現在のVision 回答テキストを保持（累積用）
                    accumulated_llama_4_scout_text = llama_4_scout_image_answer_text
                    accumulated_openai_gpt4o_text = openai_gpt4o_image_answer_text
                    accumulated_azure_openai_gpt4o_text = azure_openai_gpt4o_image_answer_text

                    # 処理方式を選択：1枚ずつ処理 vs 全画像まとめて処理 vs ファイル単位で処理
                    if base64_data_list:
                        if single_image_processing == "1枚ずつ処理":
                            # 1枚ずつ処理モード
                            print(f"単一画像処理開始: {len(base64_data_list)}枚の画像を1枚ずつ処理中...")

                            # 各base64_dataに対してLLMで処理
                            for i, (base64_data, doc_id, img_id) in enumerate(base64_data_list, 1):
                                print(f"画像 {i} (doc_id: {doc_id}, img_id: {img_id}) を処理中...")

                                # base64データをdata:image/png;base64,{base64_data}形式に変換
                                image_url = f"data:image/png;base64,{base64_data}"

                                # 各モデルの現在の画像に対する回答を保持
                                current_image_llama_4_scout = ""
                                current_image_openai_gpt4o = ""
                                current_image_azure_openai_gpt4o = ""

                                # 選択されたLLMモデルに対して処理を実行し、結果をストリーミングで取得
                                async for llm_results in process_single_image_streaming(
                                        image_url,
                                        query_text,
                                        llm_answer_checkbox_group,
                                        target_models,
                                        i,
                                        doc_id,
                                        img_id,
                                        custom_image_prompt
                                ):
                                    # 各LLMの結果を現在の画像の回答として更新
                                    if "meta/llama-4-scout-17b-16e-instruct" in llm_results:
                                        current_image_llama_4_scout = llm_results["meta/llama-4-scout-17b-16e-instruct"]

                                    if "openai/gpt-4o" in llm_results:
                                        current_image_openai_gpt4o = llm_results["openai/gpt-4o"]

                                    if "azure_openai/gpt-4o" in llm_results:
                                        current_image_azure_openai_gpt4o = llm_results["azure_openai/gpt-4o"]

                                    # 累積テキストと現在の画像の回答を結合して表示
                                    current_llama_4_scout_text = accumulated_llama_4_scout_text + current_image_llama_4_scout
                                    current_openai_gpt4o_text = accumulated_openai_gpt4o_text + current_image_openai_gpt4o
                                    current_azure_openai_gpt4o_text = accumulated_azure_openai_gpt4o_text + current_image_azure_openai_gpt4o

                                    # 更新されたVision 回答結果をyield
                                    yield (
                                        gr.Markdown(value=current_llama_4_scout_text),
                                        gr.Markdown(value=current_openai_gpt4o_text),
                                        gr.Markdown(value=current_azure_openai_gpt4o_text)
                                    )

                                # 現在の画像の処理が完了したら、累積テキストに追加
                                accumulated_llama_4_scout_text += current_image_llama_4_scout
                                accumulated_openai_gpt4o_text += current_image_openai_gpt4o
                                accumulated_azure_openai_gpt4o_text += current_image_azure_openai_gpt4o
                        elif single_image_processing == "全画像まとめて処理":
                            # 全画像まとめて処理モード
                            print(f"複数画像一括処理開始: {len(base64_data_list)}枚の画像を一括処理中...")

                            # img_idによる昇順ソート
                            sorted_base64_data_list = sorted(base64_data_list,
                                                             key=lambda x: int(x[2]) if x[2] is not None else 0)
                            print(f"全画像をimg_idで昇順ソート完了: {len(sorted_base64_data_list)}枚")

                            # 各モデルの回答を保持
                            current_llama_4_scout = ""
                            current_openai_gpt4o = ""
                            current_azure_openai_gpt4o = ""

                            # 複数画像を一括処理
                            async for llm_results in process_multiple_images_streaming(
                                    sorted_base64_data_list,
                                    query_text,
                                    llm_answer_checkbox_group,
                                    target_models,
                                    custom_image_prompt
                            ):
                                # 各LLMの結果を複数画像の回答として更新
                                if "meta/llama-4-scout-17b-16e-instruct" in llm_results:
                                    current_llama_4_scout = llm_results["meta/llama-4-scout-17b-16e-instruct"]

                                if "openai/gpt-4o" in llm_results:
                                    current_openai_gpt4o = llm_results["openai/gpt-4o"]

                                if "azure_openai/gpt-4o" in llm_results:
                                    current_azure_openai_gpt4o = llm_results["azure_openai/gpt-4o"]

                                # 累積テキストと複数画像の回答を結合して表示
                                current_llama_4_scout_text = accumulated_llama_4_scout_text + current_llama_4_scout
                                current_openai_gpt4o_text = accumulated_openai_gpt4o_text + current_openai_gpt4o
                                current_azure_openai_gpt4o_text = accumulated_azure_openai_gpt4o_text + current_azure_openai_gpt4o

                                # 更新されたVision 回答結果をyield
                                yield (
                                    gr.Markdown(value=current_llama_4_scout_text),
                                    gr.Markdown(value=current_openai_gpt4o_text),
                                    gr.Markdown(value=current_azure_openai_gpt4o_text)
                                )
                        elif single_image_processing == "ファイル単位で処理":
                            # ファイル単位で処理モード
                            print(f"ファイル単位処理開始: {len(base64_data_list)}枚の画像をファイル単位で処理中...")

                            # doc_idでグループ化
                            file_groups = {}
                            for base64_data, doc_id, img_id in base64_data_list:
                                if doc_id not in file_groups:
                                    file_groups[doc_id] = []
                                file_groups[doc_id].append((base64_data, doc_id, img_id))

                            # 各ファイルグループ内でimg_idによる昇順ソート
                            for doc_id in file_groups:
                                file_groups[doc_id].sort(key=lambda x: int(x[2]) if x[2] is not None else 0)
                                print(f"ファイル {doc_id}: img_idで昇順ソート完了 ({len(file_groups[doc_id])}枚)")

                            print(f"ファイルグループ数: {len(file_groups)}")

                            # 各ファイルグループを順次処理
                            for file_index, (doc_id, file_images) in enumerate(file_groups.items(), 1):
                                print(
                                    f"ファイル {file_index}/{len(file_groups)} (doc_id: {doc_id}) を処理中: {len(file_images)}枚の画像")

                                # 各モデルの現在のファイルに対する回答を保持
                                current_file_llama_4_scout = ""
                                current_file_openai_gpt4o = ""
                                current_file_azure_openai_gpt4o = ""

                                # 現在のファイルの画像を一括処理
                                async for llm_results in process_multiple_images_streaming(
                                        file_images,
                                        query_text,
                                        llm_answer_checkbox_group,
                                        target_models,
                                        custom_image_prompt
                                ):
                                    # 各LLMの結果を現在のファイルの回答として更新
                                    if "meta/llama-4-scout-17b-16e-instruct" in llm_results:
                                        current_file_llama_4_scout = llm_results["meta/llama-4-scout-17b-16e-instruct"]

                                    if "openai/gpt-4o" in llm_results:
                                        current_file_openai_gpt4o = llm_results["openai/gpt-4o"]

                                    if "azure_openai/gpt-4o" in llm_results:
                                        current_file_azure_openai_gpt4o = llm_results["azure_openai/gpt-4o"]

                                    # 累積テキストと現在のファイルの回答を結合して表示
                                    current_llama_4_scout_text = accumulated_llama_4_scout_text + current_file_llama_4_scout
                                    current_openai_gpt4o_text = accumulated_openai_gpt4o_text + current_file_openai_gpt4o
                                    current_azure_openai_gpt4o_text = accumulated_azure_openai_gpt4o_text + current_file_azure_openai_gpt4o

                                    # 更新されたVision 回答結果をyield
                                    yield (
                                        gr.Markdown(value=current_llama_4_scout_text),
                                        gr.Markdown(value=current_openai_gpt4o_text),
                                        gr.Markdown(value=current_azure_openai_gpt4o_text)
                                    )

                                # 現在のファイルの処理が完了したら、累積テキストに追加
                                accumulated_llama_4_scout_text += current_file_llama_4_scout
                                accumulated_openai_gpt4o_text += current_file_openai_gpt4o
                                accumulated_azure_openai_gpt4o_text += current_file_azure_openai_gpt4o

                        elif single_image_processing == "ファイル単位で処理+最初・最後":
                            # ファイル単位で処理+最初・最後モード
                            print(f"ファイル単位+最初・最後処理開始: {len(base64_data_list)}枚の画像を拡張処理中...")

                            # doc_idでグループ化
                            file_groups = {}
                            for base64_data, doc_id, img_id in base64_data_list:
                                if doc_id not in file_groups:
                                    file_groups[doc_id] = []
                                file_groups[doc_id].append((base64_data, doc_id, img_id))

                            # 各ファイルグループ内でimg_idによる昇順ソート
                            for doc_id in file_groups:
                                file_groups[doc_id].sort(key=lambda x: int(x[2]) if x[2] is not None else 0)
                                print(f"ファイル {doc_id}: img_idで昇順ソート完了 ({len(file_groups[doc_id])}枚)")

                            print(f"ファイルグループ数: {len(file_groups)}")

                            # 各ファイルグループを順次処理
                            for file_index, (doc_id, file_images) in enumerate(file_groups.items(), 1):
                                print(f"ファイル {file_index}/{len(file_groups)} (doc_id: {doc_id}) を拡張処理中...")

                                # 現在のファイルの検索された画像のimg_idを取得
                                searched_img_ids = [int(img_id) for _, _, img_id in file_images if img_id is not None]
                                print(f"検索された画像のimg_id: {searched_img_ids}")

                                # 追加で取得する画像を決定（最初と最後の画像のみ）
                                additional_images = []

                                # 最初と最後の画像を取得
                                try:
                                    with pool.acquire() as conn:
                                        with conn.cursor() as cursor:
                                            # 最初と最後の画像を取得
                                            first_last_sql = f"""
                                            SELECT base64_data, doc_id, img_id
                                            FROM {default_collection_name}_image
                                            WHERE doc_id = :doc_id
                                            AND base64_data IS NOT NULL
                                            AND (img_id = (SELECT MIN(img_id) FROM {default_collection_name}_image WHERE doc_id = :doc_id)
                                                 OR img_id = (SELECT MAX(img_id) FROM {default_collection_name}_image WHERE doc_id = :doc_id))
                                            ORDER BY img_id ASC
                                            """
                                            cursor.execute(first_last_sql, {'doc_id': doc_id})
                                            first_last_results = cursor.fetchall()

                                            for row in first_last_results:
                                                base64_data = row[0].read() if hasattr(row[0], 'read') else row[0]
                                                img_id = int(row[2]) if row[2] is not None else None
                                                if base64_data and img_id is not None and img_id not in searched_img_ids:
                                                    additional_images.append((base64_data, row[1], img_id))
                                                    print(f"最初/最後の画像を追加: img_id={img_id}")

                                except Exception as e:
                                    print(f"追加画像取得中にエラー: {e}")

                                # 検索された画像と追加画像を結合（重複を除去）
                                all_images = file_images.copy()
                                existing_img_ids = {img_id for _, _, img_id in all_images}

                                for additional_image in additional_images:
                                    if additional_image[2] not in existing_img_ids:
                                        all_images.append(additional_image)
                                        existing_img_ids.add(additional_image[2])

                                # img_idでソート
                                all_images.sort(key=lambda x: x[2])

                                print(
                                    f"処理対象画像数: {len(all_images)}枚 (検索: {len(file_images)}枚 + 追加: {len(additional_images)}枚)")

                                # 各モデルの現在のファイルに対する回答を保持
                                current_file_llama_4_scout = ""
                                current_file_openai_gpt4o = ""
                                current_file_azure_openai_gpt4o = ""

                                # 現在のファイルの全画像を一括処理
                                async for llm_results in process_multiple_images_streaming(
                                        all_images,
                                        query_text,
                                        llm_answer_checkbox_group,
                                        target_models,
                                        custom_image_prompt
                                ):
                                    # 各LLMの結果を現在のファイルの回答として更新
                                    if "meta/llama-4-scout-17b-16e-instruct" in llm_results:
                                        current_file_llama_4_scout = llm_results["meta/llama-4-scout-17b-16e-instruct"]

                                    if "openai/gpt-4o" in llm_results:
                                        current_file_openai_gpt4o = llm_results["openai/gpt-4o"]

                                    if "azure_openai/gpt-4o" in llm_results:
                                        current_file_azure_openai_gpt4o = llm_results["azure_openai/gpt-4o"]

                                    # 累積テキストと現在のファイルの回答を結合して表示
                                    current_llama_4_scout_text = accumulated_llama_4_scout_text + current_file_llama_4_scout
                                    current_openai_gpt4o_text = accumulated_openai_gpt4o_text + current_file_openai_gpt4o
                                    current_azure_openai_gpt4o_text = accumulated_azure_openai_gpt4o_text + current_file_azure_openai_gpt4o

                                    # 更新されたVision 回答結果をyield
                                    yield (
                                        gr.Markdown(value=current_llama_4_scout_text),
                                        gr.Markdown(value=current_openai_gpt4o_text),
                                        gr.Markdown(value=current_azure_openai_gpt4o_text)
                                    )

                                # 現在のファイルの処理が完了したら、累積テキストに追加
                                accumulated_llama_4_scout_text += current_file_llama_4_scout
                                accumulated_openai_gpt4o_text += current_file_openai_gpt4o
                                accumulated_azure_openai_gpt4o_text += current_file_azure_openai_gpt4o
                        else:
                            # ファイル単位で処理+最初・最後・前後画像モード
                            print(
                                f"ファイル単位+最初・最後・前後画像処理開始: {len(base64_data_list)}枚の画像を拡張処理中...")

                            # doc_idでグループ化
                            file_groups = {}
                            for base64_data, doc_id, img_id in base64_data_list:
                                if doc_id not in file_groups:
                                    file_groups[doc_id] = []
                                file_groups[doc_id].append((base64_data, doc_id, img_id))

                            # 各ファイルグループ内でimg_idによる昇順ソート
                            for doc_id in file_groups:
                                file_groups[doc_id].sort(key=lambda x: int(x[2]) if x[2] is not None else 0)
                                print(f"ファイル {doc_id}: img_idで昇順ソート完了 ({len(file_groups[doc_id])}枚)")

                            print(f"ファイルグループ数: {len(file_groups)}")

                            # 各ファイルグループを順次処理
                            for file_index, (doc_id, file_images) in enumerate(file_groups.items(), 1):
                                print(f"ファイル {file_index}/{len(file_groups)} (doc_id: {doc_id}) を拡張処理中...")

                                # 現在のファイルの検索された画像のimg_idを取得
                                searched_img_ids = [int(img_id) for _, _, img_id in file_images if img_id is not None]
                                print(f"検索された画像のimg_id: {searched_img_ids}")

                                # 追加で取得する画像を決定
                                additional_images = []

                                # 1. 最初と最後の画像を取得
                                try:
                                    with pool.acquire() as conn:
                                        with conn.cursor() as cursor:
                                            # 最初と最後の画像を取得
                                            first_last_sql = f"""
                                            SELECT base64_data, doc_id, img_id
                                            FROM {default_collection_name}_image
                                            WHERE doc_id = :doc_id
                                            AND base64_data IS NOT NULL
                                            AND (img_id = (SELECT MIN(img_id) FROM {default_collection_name}_image WHERE doc_id = :doc_id)
                                                 OR img_id = (SELECT MAX(img_id) FROM {default_collection_name}_image WHERE doc_id = :doc_id))
                                            ORDER BY img_id ASC
                                            """
                                            cursor.execute(first_last_sql, {'doc_id': doc_id})
                                            first_last_results = cursor.fetchall()

                                            for row in first_last_results:
                                                base64_data = row[0].read() if hasattr(row[0], 'read') else row[0]
                                                # img_idはNUMBER型なので、直接使用するか安全に変換
                                                img_id = row[2] if isinstance(row[2], (int, float)) and row[
                                                    2] is not None else (int(row[2]) if row[2] is not None else None)
                                                if base64_data and img_id is not None and img_id not in searched_img_ids:
                                                    additional_images.append((base64_data, row[1], img_id))
                                                    print(f"最初/最後の画像を追加: img_id={img_id}")

                                            # 2. 検索された画像の前後の画像を取得
                                            for searched_img_id in searched_img_ids:
                                                # img_idを整数に変換（NUMBER型対応）
                                                try:
                                                    searched_img_id_int = searched_img_id if isinstance(searched_img_id,
                                                                                                        (int,
                                                                                                         float)) else int(
                                                        searched_img_id)
                                                except (ValueError, TypeError):
                                                    print(f"img_idの変換に失敗: {searched_img_id}")
                                                    continue

                                                # 前の画像
                                                prev_sql = f"""
                                                SELECT base64_data, doc_id, img_id
                                                FROM {default_collection_name}_image
                                                WHERE doc_id = :doc_id
                                                AND img_id = :prev_img_id
                                                AND base64_data IS NOT NULL
                                                """
                                                cursor.execute(prev_sql, {'doc_id': doc_id,
                                                                          'prev_img_id': searched_img_id_int - 1})
                                                prev_result = cursor.fetchone()
                                                if prev_result:
                                                    base64_data = prev_result[0].read() if hasattr(prev_result[0],
                                                                                                   'read') else \
                                                    prev_result[0]
                                                    img_id = int(prev_result[2]) if prev_result[2] is not None else None
                                                    if base64_data and img_id is not None and img_id not in searched_img_ids:
                                                        additional_images.append((base64_data, prev_result[1], img_id))
                                                        print(f"前の画像を追加: img_id={img_id}")

                                                # 後の画像
                                                next_sql = f"""
                                                SELECT base64_data, doc_id, img_id
                                                FROM {default_collection_name}_image
                                                WHERE doc_id = :doc_id
                                                AND img_id = :next_img_id
                                                AND base64_data IS NOT NULL
                                                """
                                                cursor.execute(next_sql, {'doc_id': doc_id,
                                                                          'next_img_id': searched_img_id_int + 1})
                                                next_result = cursor.fetchone()
                                                if next_result:
                                                    base64_data = next_result[0].read() if hasattr(next_result[0],
                                                                                                   'read') else \
                                                    next_result[0]
                                                    # img_idはNUMBER型なので、直接使用するか安全に変換
                                                    img_id = next_result[2] if isinstance(next_result[2],
                                                                                          (int, float)) and next_result[
                                                                                   2] is not None else (
                                                        int(next_result[2]) if next_result[2] is not None else None)
                                                    if base64_data and img_id is not None and img_id not in searched_img_ids:
                                                        additional_images.append((base64_data, next_result[1], img_id))
                                                        print(f"後の画像を追加: img_id={img_id}")

                                except Exception as e:
                                    print(f"追加画像取得中にエラー: {e}")

                                # 検索された画像と追加画像を結合（重複を除去）
                                all_images = file_images.copy()
                                existing_img_ids = {img_id for _, _, img_id in all_images}

                                for additional_image in additional_images:
                                    if additional_image[2] not in existing_img_ids:
                                        all_images.append(additional_image)
                                        existing_img_ids.add(additional_image[2])

                                # img_idでソート
                                all_images.sort(key=lambda x: x[2])

                                print(
                                    f"処理対象画像数: {len(all_images)}枚 (検索: {len(file_images)}枚 + 追加: {len(additional_images)}枚)")

                                # 各モデルの現在のファイルに対する回答を保持
                                current_file_llama_4_scout = ""
                                current_file_openai_gpt4o = ""
                                current_file_azure_openai_gpt4o = ""

                                # 現在のファイルの全画像を一括処理
                                async for llm_results in process_multiple_images_streaming(
                                        all_images,
                                        query_text,
                                        llm_answer_checkbox_group,
                                        target_models,
                                        custom_image_prompt
                                ):
                                    # 各LLMの結果を現在のファイルの回答として更新
                                    if "meta/llama-4-scout-17b-16e-instruct" in llm_results:
                                        current_file_llama_4_scout = llm_results["meta/llama-4-scout-17b-16e-instruct"]

                                    if "openai/gpt-4o" in llm_results:
                                        current_file_openai_gpt4o = llm_results["openai/gpt-4o"]

                                    if "azure_openai/gpt-4o" in llm_results:
                                        current_file_azure_openai_gpt4o = llm_results["azure_openai/gpt-4o"]

                                    # 累積テキストと現在のファイルの回答を結合して表示
                                    current_llama_4_scout_text = accumulated_llama_4_scout_text + current_file_llama_4_scout
                                    current_openai_gpt4o_text = accumulated_openai_gpt4o_text + current_file_openai_gpt4o
                                    current_azure_openai_gpt4o_text = accumulated_azure_openai_gpt4o_text + current_file_azure_openai_gpt4o

                                    # 更新されたVision 回答結果をyield
                                    yield (
                                        gr.Markdown(value=current_llama_4_scout_text),
                                        gr.Markdown(value=current_openai_gpt4o_text),
                                        gr.Markdown(value=current_azure_openai_gpt4o_text)
                                    )

                                # 現在のファイルの処理が完了したら、累積テキストに追加
                                accumulated_llama_4_scout_text += current_file_llama_4_scout
                                accumulated_openai_gpt4o_text += current_file_openai_gpt4o
                                accumulated_azure_openai_gpt4o_text += current_file_azure_openai_gpt4o
        except Exception as db_e:
            print(f"データベース操作中にエラーが発生しました: {db_e}")
            # データベースエラー時も現在の状態をyield
            yield (
                gr.Markdown(value=llama_4_scout_image_answer_text),
                gr.Markdown(value=openai_gpt4o_image_answer_text),
                gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
            )
            return
    except Exception as e:
        print(f"base64_data取得中にエラーが発生しました: {e}")
        # エラー時も現在の状態をyield
        yield (
            gr.Markdown(value=llama_4_scout_image_answer_text),
            gr.Markdown(value=openai_gpt4o_image_answer_text),
            gr.Markdown(value=azure_openai_gpt4o_image_answer_text)
        )
    finally:
        # 画像処理完了後の軽量なリソースクリーンアップ
        try:
            await lightweight_cleanup()
        except Exception as cleanup_error:
            print(f"軽量リソースクリーンアップ中にエラー: {cleanup_error}")

    print("process_image_answers_streaming() 完了")


async def process_multiple_images_streaming(image_data_list, query_text, llm_answer_checkbox_group, target_models,
                                            custom_image_prompt=None):
    """
    複数の画像を一度にVLMに送信して処理し、ストリーミング形式で回答を返す

    Args:
        image_data_list: 画像データのリスト [(base64_data, doc_id, img_id), ...]
        query_text: クエリテキスト
        llm_answer_checkbox_group: 選択されたLLMモデルのリスト
        target_models: 対象モデルのリスト
        custom_image_prompt: カスタム画像プロンプトテンプレート

    Yields:
        dict: 各モデルの部分的な回答を含む辞書
    """
    if custom_image_prompt:
        custom_image_prompt = custom_image_prompt.replace('{{query_text}}', '{query_text}')

    region = get_region()

    # 画像URLリストを作成
    image_urls = []
    for base64_data, doc_id, img_id in image_data_list:
        image_url = f"data:image/png;base64,{base64_data}"
        image_urls.append(image_url)

    print(f"複数画像処理開始: {len(image_urls)}枚の画像を一括処理")

    # 各モデルのタスクジェネレーターを作成
    async def create_model_task(model):
        llm = None  # LLMインスタンスを初期化
        try:
            if model not in llm_answer_checkbox_group:
                # 選択されていないモデルは即座に完了を通知
                yield "TASK_DONE"
                return

            print(f"\n=== 複数画像 ({len(image_urls)}枚) - {model} での処理 ===")

            if model == "meta/llama-4-scout-17b-16e-instruct":
                llm = ChatOCIGenAI(
                    model_id="meta.llama-4-scout-17b-16e-instruct",
                    provider="meta",
                    service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                    compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
                    model_kwargs={"temperature": 0.0, "top_p": 0.75, "seed": 42, "max_tokens": 3600},
                )
            elif model == "openai/gpt-4o":
                load_dotenv(find_dotenv())
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    top_p=0.75,
                    seed=42,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ["OPENAI_BASE_URL"],
                )
            elif model == "azure_openai/gpt-4o":
                load_dotenv(find_dotenv())
                llm = AzureChatOpenAI(
                    deployment_name="gpt-4o",
                    temperature=0,
                    top_p=0.75,
                    seed=42,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_GPT_4O"],
                    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_GPT_4O"],
                )
            else:
                # 未対応のモデルは即座に完了を通知
                yield "TASK_DONE"
                return

            # メッセージを作成（複数画像対応）
            if custom_image_prompt:
                prompt_text = custom_image_prompt.format(query_text=query_text)
            else:
                prompt_text = get_image_qa_prompt(query_text)

            # メッセージコンテンツを構築
            message_content = [{"type": "text", "text": prompt_text}]

            # 各画像を追加
            for i, image_url in enumerate(image_urls):
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })

            human_message = HumanMessage(content=message_content)
            messages = [human_message]

            # LLMに送信して回答を取得
            start_time = time.time()

            # 表示用に画像を圧縮してヘッダー情報を作成
            compressed_images_text = ""
            for i, (image_url, (_, _, _)) in enumerate(zip(image_urls, image_data_list), 1):
                compressed_image_url = compress_image_for_display(image_url)
                compressed_images_text += f"\n\n![画像{i}]({compressed_image_url})\n"

            # ヘッダー情報を最初にyield
            header_text = f"\n\n---\n**{len(image_urls)}枚の画像による回答：**\n\n{compressed_images_text}\n\n"
            yield header_text

            # ストリーミングで回答を取得
            async for chunk in llm.astream(messages):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    yield chunk.content

            end_time = time.time()
            inference_time = end_time - start_time
            print(f"\n\n推論時間: {inference_time:.2f}秒")
            print(f"=== {model} での処理完了 ===\n")

            # 推論時間を追加
            yield f"\n\n推論時間: {inference_time:.2f}秒\n\n"
            yield "TASK_DONE"

        except Exception as e:
            print(f"エラーが発生しました ({model}): {e}")
            # 表示用に画像を圧縮してエラーメッセージを作成
            error_text = f"\n\nエラーが発生しました: {e}\n\n"
            yield error_text
            yield "TASK_DONE"
        finally:
            # リソースクリーンアップ：LLMクライアントの接続を適切に閉じる
            await cleanup_llm_client_async(llm)
            llm = None  # 参照をクリア

    # 各モデルのジェネレーターを作成
    llama_4_scout_gen = create_model_task("meta/llama-4-scout-17b-16e-instruct")
    openai_gpt4o_gen = create_model_task("openai/gpt-4o")
    azure_openai_gpt4o_gen = create_model_task("azure_openai/gpt-4o")

    # 各モデルの応答を蓄積
    llama_4_scout_response = ""
    openai_gpt4o_response = ""
    azure_openai_gpt4o_response = ""

    # 各モデルの状態を追跡
    responses_status = ["", "", ""]

    # タイムアウト設定（最大5分）
    timeout_seconds = 300
    start_time = time.time()

    try:
        while True:
            # タイムアウトチェック
            if time.time() - start_time > timeout_seconds:
                print(f"複数画像処理がタイムアウトしました（{timeout_seconds}秒）")
                break

            responses = ["", "", ""]
            generators = [llama_4_scout_gen, openai_gpt4o_gen, azure_openai_gpt4o_gen]

            for i, gen in enumerate(generators):
                if responses_status[i] == "TASK_DONE":
                    continue

                try:
                    # タイムアウト付きでanextを実行
                    response = await asyncio.wait_for(anext(gen), timeout=30.0)
                    if response:
                        if response == "TASK_DONE":
                            responses_status[i] = response
                        else:
                            responses[i] = response
                except StopAsyncIteration:
                    responses_status[i] = "TASK_DONE"
                except asyncio.TimeoutError:
                    print(f"モデル {i} の処理がタイムアウトしました")
                    responses_status[i] = "TASK_DONE"
                except Exception as e:
                    print(f"モデル {i} の処理中にエラーが発生しました: {e}")
                    responses_status[i] = "TASK_DONE"

            # 応答を蓄積
            llama_4_scout_response += responses[0]
            openai_gpt4o_response += responses[1]
            azure_openai_gpt4o_response += responses[2]

            # 現在の状態をyield
            yield {
                "meta/llama-4-scout-17b-16e-instruct": llama_4_scout_response,
                "openai/gpt-4o": openai_gpt4o_response,
                "azure_openai/gpt-4o": azure_openai_gpt4o_response
            }

            # すべてのタスクが完了したかチェック
            if all(response_status == "TASK_DONE" for response_status in responses_status):
                print("All multiple image processing tasks completed")
                break

    finally:
        # 最終的なリソースクリーンアップ：すべてのジェネレーターを適切に閉じる
        generators = [llama_4_scout_gen, openai_gpt4o_gen, azure_openai_gpt4o_gen]
        generator_names = ["llama_4_scout", "openai_gpt4o",
                           "azure_openai_gpt4o"]

        for i, gen in enumerate(generators):
            try:
                # 非同期ジェネレーターの場合
                if hasattr(gen, 'aclose'):
                    await gen.aclose()
                    print(f"複数画像処理: ジェネレーター {generator_names[i]} の非同期クローズが完了しました")
                elif hasattr(gen, 'close'):
                    gen.close()
                    print(f"複数画像処理: ジェネレーター {generator_names[i]} の同期クローズが完了しました")

                # ジェネレーター内のLLMクライアントも確実にクリーンアップ
                if hasattr(gen, 'gi_frame') and gen.gi_frame is not None:
                    # ジェネレーターがまだ実行中の場合、強制終了
                    try:
                        gen.close()
                        print(f"複数画像処理: 実行中のジェネレーター {generator_names[i]} を強制終了しました")
                    except GeneratorExit:
                        pass  # 正常な終了
                    except Exception as force_close_error:
                        print(
                            f"複数画像処理: ジェネレーター {generator_names[i]} の強制終了中にエラー: {force_close_error}")

            except Exception as cleanup_error:
                print(
                    f"複数画像処理: ジェネレーター {generator_names[i]} のクリーンアップ中にエラーが発生しました: {cleanup_error}")

        # 軽量なHTTP接続クリーンアップ
        await lightweight_cleanup()

        # ガベージコレクションを強制実行してリソースを解放
        import gc
        gc.collect()
        print("複数画像処理のリソースクリーンアップが完了しました")
