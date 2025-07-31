import base64
import json
import logging
import os
import shutil
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image
from langchain_community.chat_models import ChatOCIGenAI
from pdf2image import convert_from_path

from utils.common_util import get_region

logger = logging.getLogger(__name__)


class _ImageOptimizationConfig:
    """画像最適化設定クラス"""

    def __init__(self):
        # DPI設定（PyMuPDFとの互換性）- 清晰度を重視して調整
        self.base_dpi = 200
        self.matrix_3x3_dpi_ratio = 0.85  # Matrix(3,3)相当 - 清晰度向上のため増加
        self.matrix_4x4_dpi_ratio = 0.9  # Matrix(4,4)相当 - 清晰度向上のため増加

        # 画像サイズ制限 - 清晰度を保つため制限を緩和
        self.max_dimension = 2400  # 最大寸法 - より高解像度を許可
        self.max_file_size_mb = 8  # 最大ファイルサイズ（MB）- 制限を緩和

        # 圧縮設定 - 清晰度とファイルサイズのバランス
        self.png_compress_level = 6  # PNG圧縮レベル（0-9）- 品質重視で軽減
        self.jpeg_quality = 88  # JPEG品質（1-100）- 品質向上

        # リサイズ設定
        self.enable_auto_resize = True
        self.resize_algorithm = Image.Resampling.LANCZOS

    def optimize_image(self, image: Image.Image) -> Image.Image:
        """
        画像を最適化してサイズを削減

        Args:
            image: 最適化する画像

        Returns:
            最適化された画像
        """
        if not self.enable_auto_resize:
            return image

        # サイズチェックとリサイズ
        if image.width > self.max_dimension or image.height > self.max_dimension:
            ratio = min(
                self.max_dimension / image.width,
                self.max_dimension / image.height
            )
            new_size = (
                int(image.width * ratio),
                int(image.height * ratio)
            )
            logger.debug(f"画像をリサイズ: {image.size} -> {new_size}")
            image = image.resize(new_size, self.resize_algorithm)

        return image

    def get_save_options(self, file_format: str) -> dict:
        """
        ファイル形式に応じた保存オプションを取得

        Args:
            file_format: ファイル形式（'PNG', 'JPEG'など）

        Returns:
            保存オプション辞書
        """
        file_format = file_format.upper()

        if file_format == 'PNG':
            return {
                'format': 'PNG',
                'optimize': True,
                'compress_level': self.png_compress_level
            }
        elif file_format in ['JPEG', 'JPG']:
            return {
                'format': 'JPEG',
                'optimize': True,
                'quality': self.jpeg_quality
            }
        else:
            return {'format': file_format}


def _save_optimized_image(image: Image.Image, filepath: str):
    """
    最適化設定で画像を保存

    Args:
        image: 保存する画像
        filepath: 保存先パス
    """
    config = _ImageOptimizationConfig()

    # ファイル拡張子から形式を判定
    file_ext = filepath.lower().split('.')[-1]
    format_map = {
        'png': 'PNG',
        'jpg': 'JPEG',
        'jpeg': 'JPEG'
    }

    file_format = format_map.get(file_ext, 'PNG')
    save_options = config.get_save_options(file_format)

    # 画像を最適化
    optimized_image = config.optimize_image(image)

    # 保存
    optimized_image.save(filepath, **save_options)
    logger.debug(f"最適化画像を保存: {filepath} (形式: {file_format})")


def convert_excel_to_text_document(file_path):
    """
    ExcelファイルをJSONライン形式のテキストドキュメントに変換する

    Args:
        file_path: アップロードされたファイルのパス

    Returns:
        tuple: (クリアされたファイル入力, 変換されたファイル)
    """
    has_error = False
    if not file_path:
        has_error = True
        gr.Warning("ファイルを選択してください")
    if has_error:
        return gr.File(value=None), gr.File()

    output_file_path = file_path.name + '.txt'
    df = pd.read_excel(file_path.name)
    data = df.to_dict('records')
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for row in data:
            # 各行を処理してTimestampを文字列に変換
            processed_row = {}
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    processed_row[key] = str(value)
                else:
                    processed_row[key] = value
            json_line = json.dumps(processed_row, ensure_ascii=False)
            f.write(json_line + ' <FIXED_DELIMITER>\n')
    return (
        gr.File(),
        gr.File(value=output_file_path)
    )


def convert_json_to_text_document(file_path):
    """
    JSONファイルをテキストドキュメントに変換する

    Args:
        file_path: アップロードされたファイルのパス

    Returns:
        tuple: (クリアされたファイル入力, 変換されたファイル)
    """
    has_error = False
    if not file_path:
        has_error = True
        gr.Warning("ファイルを選択してください")
    if has_error:
        return gr.File(value=None), gr.File()

    # 出力ファイルパスを生成（元のファイル名 + .txt）
    output_file_path = file_path.name + '.txt'

    try:
        # JSONファイルを読み込み
        with open(file_path.name, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # JSONデータが配列であることを確認
        if not isinstance(json_data, list):
            gr.Warning("JSONファイルは配列形式である必要があります: [{}、{}、{}]")
            return gr.File(value=None), gr.File()

        # テキストファイルに変換
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for item in json_data:
                if isinstance(item, dict):
                    # 辞書オブジェクトをJSON文字列に変換
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + ' <FIXED_DELIMITER>\n')
                else:
                    # 辞書以外のオブジェクトもJSON文字列として処理
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + ' <FIXED_DELIMITER>\n')

        return (
            gr.File(),
            gr.File(value=output_file_path)
        )

    except json.JSONDecodeError as e:
        gr.Warning(f"JSONファイルの解析に失敗しました: {str(e)}")
        return gr.File(value=None), gr.File()
    except Exception as e:
        gr.Warning(f"ファイル変換中にエラーが発生しました: {str(e)}")
        return gr.File(value=None), gr.File()


def convert_pdf_to_markdown(file_path):
    """
    PDFファイルをMarkdownドキュメントに変換する

    Args:
        file_path: アップロードされたファイルのパス

    Returns:
        tuple: (クリアされたファイル入力, 変換されたファイル)
    """
    has_error = False
    if not file_path:
        has_error = True
        gr.Warning("ファイルを選択してください")
    if has_error:
        return gr.File(value=None), gr.File()

    pdf_path = file_path.name
    pdf_filename = Path(pdf_path).stem
    output_dir = Path("output") / f"{pdf_filename}_md"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    images = convert_from_path(pdf_path)

    all_md_content = []
    for i, image in enumerate(images):
        page_num = i + 1

        relative_image_path = Path("images") / f"page_{page_num}.png"
        absolute_image_path = output_dir / relative_image_path

        _save_optimized_image(image, str(absolute_image_path))

        with open(absolute_image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        image_path_str = relative_image_path.as_posix()

        md_content = (
            f"<!-- image_begin -->\n\n"
            f"<!-- image_url_begin -->\n\n"
            f"<!-- ![画像]({image_path_str}) -->\n\n"
            f"<!-- image_url_end -->\n\n"
            f"<!-- image_base64_begin -->\n\n"
            f"<!-- {image_base64} -->\n\n"
            f"<!-- image_base64_end -->\n\n"
            f"<!-- image_end -->\n\n"
        )
        all_md_content.append(md_content)

    merged_md_path = output_dir / f"{pdf_filename}.md"
    with open(merged_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_md_content))

    return (
        gr.File(),
        gr.File(value=str(merged_md_path.resolve()))
    )
