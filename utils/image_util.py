from PIL import Image
import base64
from io import BytesIO


# 支持的画像フォーマットの定義
SUPPORTED_FORMATS = {
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'gif': 'image/gif',
    'bmp': 'image/bmp',
    'webp': 'image/webp'
}


def get_image_media_type(image_path: str) -> str:
    """
    ファイル拡張子に基づいて画像のメディアタイプを決定する

    Args:
        image_path (str): 画像ファイルのパス

    Returns:
        str: 画像のメディアタイプ

    Raises:
        ValueError: サポートされていない画像フォーマットの場合
    """
    image_extension = image_path.lower().split('.')[-1]

    if image_extension in SUPPORTED_FORMATS:
        return SUPPORTED_FORMATS[image_extension]
    else:
        supported_list = ', '.join(SUPPORTED_FORMATS.keys())
        raise ValueError(f"サポートされていない画像フォーマットです。サポート対象: {supported_list}")


def _read_image_as_base64(image_path: str) -> str:
    """
    画像ファイルを読み込んでbase64エンコードする内部ヘルパー関数

    Args:
        image_path (str): 画像ファイルのパス

    Returns:
        str: base64エンコードされた画像データ
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image(image_path: str) -> str:
    """
    画像をbase64エンコードするヘルパー関数（後方互換性のため）

    Args:
        image_path (str): 画像ファイルのパス

    Returns:
        str: base64エンコードされた画像データ
    """
    return _read_image_as_base64(image_path)


def image_to_base64_url(image_path: str) -> str:
    """
    画像をデータURLとして返す

    Args:
        image_path (str): 画像ファイルのパス

    Returns:
        str: データURL形式の画像データ
    """
    base64_string = _read_image_as_base64(image_path)
    mime_type = get_image_media_type(image_path)
    return f"data:{mime_type};base64,{base64_string}"


def image_to_base64(image_path: str, output_format: str = 'JPEG', quality: int = 20,
                   include_data_url: bool = False) -> str:
    """
    画像を指定フォーマットで圧縮してbase64エンコードする

    Args:
        image_path (str): 画像ファイルのパス
        output_format (str): 出力フォーマット（JPEG, PNG等）
        quality (int): JPEG品質（1-100）
        include_data_url (bool): データURLとして返すかどうか

    Returns:
        str: base64エンコードされた画像データまたはデータURL
    """
    # 画像を開いてRGBに変換
    image = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format=output_format, quality=quality)

    # base64エンコード
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # データURLとして返すかどうか
    if include_data_url:
        mime_type = f"image/{output_format.lower()}"
        return f"data:{mime_type};base64,{base64_string}"
    else:
        return base64_string    


def combine_images(image1_path: str, image2_path: str, output_path: str, direction: str = 'vertical') -> None:
    """
    2つの画像を結合する

    Args:
        image1_path (str): 最初の画像のパス
        image2_path (str): 2番目の画像のパス
        output_path (str): 出力画像のパス
        direction (str): 結合方向（'vertical' または 'horizontal'）

    Raises:
        ValueError: サポートされていない方向が指定された場合
    """
    if direction not in ['vertical', 'horizontal']:
        raise ValueError("方向は 'vertical' または 'horizontal' である必要があります")

    # 2つの画像を開いてRGBに変換
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")

    # 画像サイズを取得
    image1_width, image1_height = image1.size
    image2_width, image2_height = image2.size

    # 結合方向に応じて新しい画像サイズと配置位置を計算
    if direction == 'horizontal':
        new_width = image1_width + image2_width
        new_height = max(image1_height, image2_height)
        image2_position = (image1_width, 0)
    else:  # vertical
        new_width = max(image1_width, image2_width)
        new_height = image1_height + image2_height
        image2_position = (0, image1_height)

    # 新しい空白画像を作成
    new_image = Image.new('RGB', (new_width, new_height))

    # 画像を配置
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, image2_position)

    # 結合画像を保存
    new_image.save(output_path)
    print(f"結合画像を保存しました: {output_path}")


def compress_image_for_display(image_url: str, quality: int = 85, max_width: int = 800, max_height: int = 1200) -> str:
    """
    画像URLを圧縮して表示用の新しいURLを生成する

    Args:
        image_url (str): 元の画像URL（data:image/...;base64,... 形式）
        quality (int): JPEG圧縮品質 (1-100)
        max_width (int): 最大幅
        max_height (int): 最大高さ

    Returns:
        str: 圧縮された画像のdata URL
    """
    compressed_url, _ = compress_image_for_display_with_info(image_url, quality, max_width, max_height)
    return compressed_url


def compress_image_for_display_with_info(image_url: str, quality: int = 85, max_width: int = 800, max_height: int = 1200) -> tuple[str, str]:
    """
    画像URLを圧縮して表示用の新しいURLと圧縮情報を生成する

    Args:
        image_url (str): 元の画像URL（data:image/...;base64,... 形式）
        quality (int): JPEG圧縮品質 (1-100)
        max_width (int): 最大幅
        max_height (int): 最大高さ

    Returns:
        tuple[str, str]: (圧縮された画像のdata URL, 圧縮情報テキスト)
    """
    output_buffer = None
    try:
        # data URLかどうかをチェック
        if not image_url.startswith('data:image/'):
            return image_url, ""

        # base64データ部分を取得（ヘッダー部分は使用しない）
        _, base64_data = image_url.split(',', 1)

        # base64をデコードして画像データを取得
        image_data = base64.b64decode(base64_data)

        # PILで画像を開く
        with Image.open(BytesIO(image_data)) as img:
            # RGBモードに変換（透明度やパレットモードの場合）
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # 元の画像サイズを取得
            original_width, original_height = img.size
            original_format = img.format if img.format else "Unknown"

            # リサイズが必要かチェック
            resized = False
            new_width, new_height = original_width, original_height
            if original_width > max_width or original_height > max_height:
                # アスペクト比を保持してリサイズ
                ratio = min(max_width / original_width, max_height / original_height)
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized = True
                print(f"画像サイズを圧縮: {original_width}x{original_height} -> {new_width}x{new_height}")

            # 圧縮された画像をバイトストリームに保存
            output_buffer = BytesIO()
            img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
            compressed_data = output_buffer.getvalue()

            # base64エンコード
            compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')

            # 新しいdata URLを生成
            compressed_url = f"data:image/jpeg;base64,{compressed_base64}"

            # 圧縮率を計算
            original_size = len(base64_data)
            compressed_size = len(compressed_base64)
            compression_ratio = (1 - compressed_size / original_size) * 100

            # 圧縮情報テキストを生成
            compression_info = f"元画像: {original_width}×{original_height}px ({original_format}), "
            if resized:
                compression_info += f"リサイズ後: {new_width}×{new_height}px, "
            compression_info += f"品質: {quality}%, データサイズ: {original_size:,} → {compressed_size:,} bytes ({compression_ratio:.1f}% 削減)"

            print(f"画像圧縮完了: {compression_info}")

            return compressed_url, compression_info

    except Exception as e:
        error_msg = f"画像圧縮中にエラーが発生しました: {e}"
        print(error_msg)
        return image_url, error_msg  # エラー時は元の画像URLとエラーメッセージを返す
    finally:
        # BytesIOバッファのクリーンアップ
        if output_buffer is not None:
            try:
                output_buffer.close()
            except Exception:
                pass  # クリーンアップエラーは無視
