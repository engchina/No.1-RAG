import base64
import json
import logging
import chardet
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path

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


def convert_xml_to_text_document(
        file_path,
        global_tag_text,
        fixed_tag_text,
        replace_tag_text,
        prefix_tag_text,
        main_tag_text,
        suffix_tag_text,
        merge_checkbox
):
    """
    XMLファイルをテキストドキュメントに変換する

    Args:
        file_path: アップロードされたファイルのパス
        global_tag_text: グローバルタグ
        fixed_tag_text: 固定タグ
        replace_tag_text: 置換タグ
        prefix_tag_text: 前置きタグ
        main_tag_text: 主タグ
        suffix_tag_text: 後付けタグ
        merge_checkbox: マージするかどうか

    Returns:
        tuple: (クリアされたファイル入力, 変換されたファイル)
    """
    has_error = False
    if not file_path:
        has_error = True
        gr.Warning("ファイルを選択してください")

    # 主タグの検証
    if not main_tag_text or not main_tag_text.strip():
        has_error = True
        gr.Warning("主タグは必須項目です。入力してください")
    elif main_tag_text and main_tag_text.strip():
        # 主タグにカンマが含まれていないかチェック
        if ',' in main_tag_text:
            has_error = True
            gr.Warning("主タグは単一の文字列である必要があります。カンマ区切りの複数値は使用できません")

    if has_error:
        return (
            gr.File(),  # tab_convert_document_convert_xml_to_text_file_text
            global_tag_text,  # tab_convert_document_convert_xml_global_tag_text
            fixed_tag_text,  # tab_convert_document_convert_xml_fixed_tag_text
            replace_tag_text,  # tab_convert_document_convert_xml_replace_tag_text
            prefix_tag_text,  # tab_convert_document_convert_xml_prefix_tag_text
            main_tag_text,  # tab_convert_document_convert_xml_main_tag_text
            suffix_tag_text,  # tab_convert_document_convert_xml_suffix_tag_text
            merge_checkbox,  # tab_convert_document_convert_xml_merge_checkbox
            gr.File()  # tab_load_document_file_text
        )

    try:
        # XMLファイルを解析
        tree = ET.parse(file_path.name)
        root = tree.getroot()

        # 出力ファイルパスを生成
        output_file_path = file_path.name + '.txt'

        # 主タグが指定されている場合、そのタグの要素を取得
        if main_tag_text:
            main_elements = root.findall(f".//{main_tag_text}")
        else:
            # 主タグが指定されていない場合、ルート要素の直接の子要素を使用
            main_elements = list(root)

        if not main_elements:
            gr.Warning(f"指定された主タグ '{main_tag_text}' が見つかりません")
            return (
                gr.File(),  # tab_convert_document_convert_xml_to_text_file_text
                global_tag_text,  # tab_convert_document_convert_xml_global_tag_text
                fixed_tag_text,  # tab_convert_document_convert_xml_fixed_tag_text
                replace_tag_text,  # tab_convert_document_convert_xml_replace_tag_text
                prefix_tag_text,  # tab_convert_document_convert_xml_prefix_tag_text
                main_tag_text,  # tab_convert_document_convert_xml_main_tag_text
                suffix_tag_text,  # tab_convert_document_convert_xml_suffix_tag_text
                merge_checkbox,  # tab_convert_document_convert_xml_merge_checkbox
                gr.File()  # tab_load_document_file_text
            )

        # テキストファイルに変換
        # 主タグ要素にIDを割り当て（元の順序を保持）
        element_id_map = {}
        for i, element in enumerate(main_elements):
            element_id_map[element] = str(i + 1)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            if merge_checkbox:
                # マージモード：主タグを親要素ごとにグループ化してから処理
                parent_groups = _group_main_elements_by_parent(main_elements, root)

                # 全ての合併グループを収集して、源文档の順序でソート
                all_merged_groups = []

                for parent_element, parent_main_elements in parent_groups.items():
                    merged_groups = _merge_adjacent_main_tags(parent_main_elements, root, main_tag_text,
                                                              global_tag_text, fixed_tag_text,
                                                              prefix_tag_text, suffix_tag_text, replace_tag_text,
                                                              element_id_map)
                    all_merged_groups.extend(merged_groups)

                # 各グループの最初の主要素の元の位置でソート
                def get_group_position(group):
                    first_element = group.get('_first_main_element')
                    if first_element and first_element in element_id_map:
                        return int(element_id_map[first_element])
                    return float('inf')

                all_merged_groups.sort(key=get_group_position)

                # マージモード用のIDカウンター（源文档の順序で1から開始）
                for i, group in enumerate(all_merged_groups):
                    # 内部使用の_first_main_elementを削除
                    if '_first_main_element' in group:
                        del group['_first_main_element']

                    # IDを追加
                    group_with_id = {"id": str(i + 1)}
                    group_with_id.update(group)

                    json_line = json.dumps(group_with_id, ensure_ascii=False)
                    f.write(json_line + ' <FIXED_DELIMITER>\n')
            else:
                # 非マージモード：主タグを親要素ごとにグループ化してから個別処理
                parent_groups = _group_main_elements_by_parent(main_elements, root)

                for parent_element, parent_main_elements in parent_groups.items():
                    for i, element in enumerate(parent_main_elements):
                        element_dict = _xml_element_to_dict(element, root, global_tag_text, fixed_tag_text,
                                                            prefix_tag_text, suffix_tag_text, replace_tag_text,
                                                            parent_main_elements, i, element_id_map)

                        json_line = json.dumps(element_dict, ensure_ascii=False)
                        f.write(json_line + ' <FIXED_DELIMITER>\n')

        return (
            gr.File(),  # tab_convert_document_convert_xml_to_text_file_text
            global_tag_text,  # tab_convert_document_convert_xml_global_tag_text
            fixed_tag_text,  # tab_convert_document_convert_xml_fixed_tag_text
            replace_tag_text,  # tab_convert_document_convert_xml_replace_tag_text
            prefix_tag_text,  # tab_convert_document_convert_xml_prefix_tag_text
            main_tag_text,  # tab_convert_document_convert_xml_main_tag_text
            suffix_tag_text,  # tab_convert_document_convert_xml_suffix_tag_text
            merge_checkbox,  # tab_convert_document_convert_xml_merge_checkbox
            gr.File(value=output_file_path)  # tab_load_document_file_text
        )

    except ET.ParseError as e:
        gr.Warning(f"XMLファイルの解析に失敗しました: {str(e)}")
        return (
            gr.File(),  # tab_convert_document_convert_xml_to_text_file_text
            global_tag_text,  # tab_convert_document_convert_xml_global_tag_text
            fixed_tag_text,  # tab_convert_document_convert_xml_fixed_tag_text
            replace_tag_text,  # tab_convert_document_convert_xml_replace_tag_text
            prefix_tag_text,  # tab_convert_document_convert_xml_prefix_tag_text
            main_tag_text,  # tab_convert_document_convert_xml_main_tag_text
            suffix_tag_text,  # tab_convert_document_convert_xml_suffix_tag_text
            merge_checkbox,  # tab_convert_document_convert_xml_merge_checkbox
            gr.File()  # tab_load_document_file_text
        )
    except Exception as e:
        gr.Warning(f"ファイル変換中にエラーが発生しました: {str(e)}")
        return (
            gr.File(),  # tab_convert_document_convert_xml_to_text_file_text
            global_tag_text,  # tab_convert_document_convert_xml_global_tag_text
            fixed_tag_text,  # tab_convert_document_convert_xml_fixed_tag_text
            replace_tag_text,  # tab_convert_document_convert_xml_replace_tag_text
            prefix_tag_text,  # tab_convert_document_convert_xml_prefix_tag_text
            main_tag_text,  # tab_convert_document_convert_xml_main_tag_text
            suffix_tag_text,  # tab_convert_document_convert_xml_suffix_tag_text
            merge_checkbox,  # tab_convert_document_convert_xml_merge_checkbox
            gr.File()  # tab_load_document_file_text
        )


def _group_main_elements_by_parent(main_elements, root_element):
    """
    主タグ要素を親要素ごとにグループ化する

    Args:
        main_elements: 主タグ要素のリスト
        root_element: ルートXML要素

    Returns:
        dict: {親要素: [その親要素下の主タグ要素のリスト], ...}
    """
    parent_groups = {}

    for main_element in main_elements:
        # 主タグの親要素を見つける
        parent = None
        for elem in root_element.iter():
            if main_element in list(elem):
                parent = elem
                break

        if parent is not None:
            if parent not in parent_groups:
                parent_groups[parent] = []
            parent_groups[parent].append(main_element)

    return parent_groups


def _xml_element_to_dict(element, root_element, global_tag=None, fixed_tag=None, prefix_tag=None,
                         suffix_tag=None, replace_tag=None, main_elements=None, current_index=None,
                         element_id_map=None):
    """
    XML要素を辞書に変換する

    Args:
        element: XML要素
        root_element: ルートXML要素
        global_tag: グローバルタグ（カンマ区切り）
        fixed_tag: 固定タグ
        prefix_tag: 前置きタグ（カンマ区切り）
        suffix_tag: 後付けタグ（カンマ区切り）
        replace_tag: 置換タグ
        main_elements: 主要素のリスト
        current_index: 現在の要素のインデックス
        element_id_map: 要素IDマッピング

    Returns:
        dict: 変換された辞書
    """
    # 順序付きの結果辞書を構築: id，グローバルタグ，固定タグ，前置きタグ，主タグ，後付けタグ
    result = {}

    # 1. IDを最前面に追加
    if element_id_map and element in element_id_map:
        result["id"] = element_id_map[element]

    # 2. グローバルタグを追加
    global_data = {}
    if global_tag and global_tag.strip():
        # グローバルタグの形式: "tag1,tag2,tag3,..."
        # 主タグと同じ親要素内から指定されたタグを検索
        for global_tag_name in global_tag.split(','):
            global_tag_name = global_tag_name.strip()
            if global_tag_name:
                # 主タグの親要素を見つける
                parent = None
                for elem in root_element.iter():
                    if element in list(elem):
                        parent = elem
                        break

                if parent is not None:
                    # 親要素内から指定されたタグを検索
                    global_elements = parent.findall(f".//{global_tag_name}")
                    for global_elem in global_elements:
                        # グローバル要素のテキスト内容を直接追加
                        if global_elem.text and global_elem.text.strip():
                            global_data[f"{global_tag_name}"] = global_elem.text.strip()

                        # グローバル要素の属性も追加
                        if global_elem.attrib:
                            for attr_name, attr_value in global_elem.attrib.items():
                                global_data[f"{global_tag_name}@{attr_name}"] = attr_value

    result.update(global_data)

    # 3. 固定タグを追加
    fixed_data = {}
    if fixed_tag and fixed_tag.strip():
        for fixed_item in fixed_tag.split(','):
            if '=' in fixed_item:
                key, value = fixed_item.strip().split('=', 1)
                fixed_data[key.strip()] = value.strip()

    result.update(fixed_data)

    # 4. 前置きタグを追加
    prefix_data = {}
    if prefix_tag and prefix_tag.strip() and main_elements is not None and current_index is not None:
        prefix_tags = [tag.strip() for tag in prefix_tag.split(',') if tag.strip()]
        if prefix_tags:
            # 現在の要素より前の要素から前置きタグを検索
            # 親要素を見つけるためにルート要素から検索
            parent = None
            for elem in root_element.iter():
                if element in list(elem):
                    parent = elem
                    break

            if parent is not None:
                all_children = list(parent)

                # 現在の要素の位置を見つける
                current_pos = None
                for i, child in enumerate(all_children):
                    if child == element:
                        current_pos = i
                        break

                if current_pos is not None:
                    # 前の主要素の位置を見つける
                    prev_main_pos = None
                    for i in range(current_index):
                        if i < len(main_elements):
                            prev_main_element = main_elements[i]
                            for j, child in enumerate(all_children):
                                if child == prev_main_element:
                                    prev_main_pos = j
                                    break

                    start_pos = prev_main_pos + 1 if prev_main_pos is not None else 0

                    # start_pos から current_pos の間で前置きタグを検索
                    for i in range(start_pos, current_pos):
                        child = all_children[i]
                        if child.tag in prefix_tags:
                            if child.text and child.text.strip():
                                key = f"{child.tag}"
                                if key in prefix_data:
                                    # 既存の値と新しい値を\nで連結
                                    prefix_data[key] = prefix_data[key] + "\n" + child.text.strip()
                                else:
                                    prefix_data[key] = child.text.strip()
                            if child.attrib:
                                for attr_name, attr_value in child.attrib.items():
                                    attr_key = f"{child.tag}@{attr_name}"
                                    if attr_key in prefix_data:
                                        # 既存の値と新しい値を\nで連結
                                        prefix_data[attr_key] = prefix_data[attr_key] + "\n" + attr_value
                                    else:
                                        prefix_data[attr_key] = attr_value

    result.update(prefix_data)

    # 5. 主タグ（要素のタグ名とテキスト内容）を追加
    tag_name = element.tag

    # 要素の属性を追加
    if element.attrib:
        for attr_name, attr_value in element.attrib.items():
            result[f"{tag_name}@{attr_name}"] = attr_value

    # 要素のテキスト内容を追加
    if element.text and element.text.strip():
        result[tag_name] = element.text.strip()

    # 子要素を再帰的に処理
    for child in element:
        child_dict = _xml_element_to_dict(child, root_element, global_tag, fixed_tag, prefix_tag,
                                          suffix_tag, replace_tag, None, None)
        result.update(child_dict)

    # 6. 後付けタグを追加
    suffix_data = {}
    if suffix_tag and suffix_tag.strip() and main_elements is not None and current_index is not None:
        suffix_tags = [tag.strip() for tag in suffix_tag.split(',') if tag.strip()]
        if suffix_tags and current_index < len(main_elements) - 1:
            # 現在の要素より後の要素から後付けタグを検索
            # 親要素を見つけるためにルート要素から検索
            parent = None
            for elem in root_element.iter():
                if element in list(elem):
                    parent = elem
                    break

            if parent is not None:
                # 親要素の子要素を順番に取得
                all_children = list(parent)
                current_pos = None
                for i, child in enumerate(all_children):
                    if child == element:
                        current_pos = i
                        break

                if current_pos is not None:
                    # 現在の要素より後の要素から後付けタグを検索
                    next_main_pos = None
                    if current_index < len(main_elements) - 1:
                        # 次の主要素の位置を見つける
                        next_main_element = main_elements[current_index + 1]
                        for i, child in enumerate(all_children):
                            if child == next_main_element:
                                next_main_pos = i
                                break

                    end_pos = next_main_pos if next_main_pos is not None else len(all_children)

                    # current_pos + 1 から end_pos の間で後付けタグを検索
                    for i in range(current_pos + 1, end_pos):
                        child = all_children[i]
                        if child.tag in suffix_tags:
                            if child.text and child.text.strip():
                                key = f"{child.tag}"
                                if key in suffix_data:
                                    # 既存の値と新しい値を\nで連結
                                    suffix_data[key] = suffix_data[key] + "\n" + child.text.strip()
                                else:
                                    suffix_data[key] = child.text.strip()
                            if child.attrib:
                                for attr_name, attr_value in child.attrib.items():
                                    attr_key = f"{child.tag}@{attr_name}"
                                    if attr_key in suffix_data:
                                        # 既存の値と新しい値を\nで連結
                                        suffix_data[attr_key] = suffix_data[attr_key] + "\n" + attr_value
                                    else:
                                        suffix_data[attr_key] = attr_value

    result.update(suffix_data)

    # 固定タグが指定されている場合、固定値を追加
    if fixed_tag and fixed_tag.strip():
        # 固定タグの形式: "key1=value1,key2=value2,..."
        for fixed_item in fixed_tag.split(','):
            if '=' in fixed_item:
                key, value = fixed_item.strip().split('=', 1)
                result[key.strip()] = value.strip()

    # 置換タグが指定されている場合、JSONのキーを置換
    if replace_tag and replace_tag.strip():
        # 置換タグの形式: "old_key=new_key,old_key2=new_key2,..."
        replacements = {}
        for replacement in replace_tag.split(','):
            if '=' in replacement:
                old_key, new_key = replacement.strip().split('=', 1)
                replacements[old_key.strip()] = new_key.strip()

        # 辞書のキーを置換
        if replacements:
            new_result = {}
            for key, value in result.items():
                new_key = replacements.get(key, key)  # 置換対象でなければ元のキーを使用
                new_result[new_key] = value
            result = new_result

    return result


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
        # 入力ファイルのエンコーディングを推定
        with open(file_path.name, 'rb') as f:
            raw_data = f.read(4096)
            result = chardet.detect(raw_data)
            print(f"{result=}")
            detected_encoding = result.get("encoding")

        # 検出されたエンコーディングを検証し、失敗した場合はフォールバック
        encoding = None
        
        # まず検出されたエンコーディングを試す（置信度が0.5以上の場合）
        if detected_encoding and result.get("confidence", 0.0) >= 0.5:
            try:
                with open(file_path.name, 'r', encoding=detected_encoding) as test_file:
                    test_file.read(1000)  # 最初の1000文字を読んで検証
                encoding = detected_encoding
                print(f"Validated detected encoding: {encoding}")
            except (UnicodeDecodeError, UnicodeError, LookupError):
                print(f"Detected encoding {detected_encoding} failed validation, trying fallbacks...")

        # 検出されたエンコーディングが使えない場合、フォールバックを試す
        if encoding is None:
            # 中国語、日本語、一般的なエンコーディングを含む包括的なリスト
            fallback_encodings = [
                'utf-8', 'utf-8-sig',  # UTF-8系（最も一般的）
                'gbk', 'gb18030', 'gb2312',  # 中国語エンコーディング
                'cp932', 'shift_jis', 'euc-jp', 'iso-2022-jp',  # 日本語エンコーディング
                'latin1', 'cp1252',  # 西欧系
                'big5'  # 繁体字中国語
            ]

            for test_encoding in fallback_encodings:
                try:
                    with open(file_path.name, 'r', encoding=test_encoding) as test_file:
                        test_file.read(1000)
                    encoding = test_encoding
                    print(f"Fallback encoding detected: {encoding}")
                    break
                except (UnicodeDecodeError, UnicodeError, LookupError):
                    continue

            if encoding is None:
                encoding = 'utf-8'
                print(f"All encodings failed, using default: {encoding}")

        # JSONファイルを読み込み（検証済みエンコーディングを使用）
        with open(file_path.name, 'r', encoding=encoding) as f:
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


def _merge_adjacent_main_tags(main_elements, root_element, main_tag=None, global_tag=None, fixed_tag=None,
                              prefix_tag=None, suffix_tag=None, replace_tag=None, element_id_map=None):
    """
    相邻的主タグ内容をマージしてグループ化する

    Args:
        main_elements: 主要素のリスト
        root_element: ルートXML要素
        main_tag: 主タグ名
        global_tag: グローバルタグ
        fixed_tag: 固定タグ
        prefix_tag: 前置きタグ
        suffix_tag: 後付けタグ
        replace_tag: 置換タグ
        element_id_map: 要素IDマッピング

    Returns:
        list: マージされたグループのリスト
    """
    if not main_elements:
        return []

    # 親要素を見つける
    parent = None
    for elem in root_element.iter():
        if main_elements[0] in list(elem):
            parent = elem
            break

    if parent is None:
        return []

    # 親要素の全ての子要素を取得
    all_children = list(parent)

    # 主要素の位置をマップ
    main_element_positions = {}
    for i, main_elem in enumerate(main_elements):
        for j, child in enumerate(all_children):
            if child == main_elem:
                main_element_positions[j] = i
                break

    groups = []
    current_group_data = {
        'global': {},
        'fixed': {},
        'prefix': {},
        'main_texts': [],
        'suffix': {},
        'first_main_element': None  # 各グループの最初の主要素を追跡
    }

    # グローバルタグを処理
    if global_tag and global_tag.strip():
        for global_tag_name in global_tag.split(','):
            global_tag_name = global_tag_name.strip()
            if global_tag_name:
                # 主タグと同じ親要素内から指定されたタグを検索
                if parent is not None:
                    global_elements = parent.findall(f".//{global_tag_name}")
                    for global_elem in global_elements:
                        if global_elem.text and global_elem.text.strip():
                            current_group_data['global'][global_tag_name] = global_elem.text.strip()
                        if global_elem.attrib:
                            for attr_name, attr_value in global_elem.attrib.items():
                                current_group_data['global'][f"{global_tag_name}@{attr_name}"] = attr_value

    # 固定タグを処理
    if fixed_tag and fixed_tag.strip():
        for fixed_item in fixed_tag.split(','):
            if '=' in fixed_item:
                key, value = fixed_item.strip().split('=', 1)
                current_group_data['fixed'][key.strip()] = value.strip()

    # 前置きタグのリストを準備
    prefix_tags = []
    if prefix_tag and prefix_tag.strip():
        prefix_tags = [tag.strip() for tag in prefix_tag.split(',') if tag.strip()]

    # 後付けタグのリストを準備
    suffix_tags = []
    if suffix_tag and suffix_tag.strip():
        suffix_tags = [tag.strip() for tag in suffix_tag.split(',') if tag.strip()]

    i = 0
    while i < len(all_children):
        child = all_children[i]

        if i in main_element_positions:
            # これは主要素
            main_element = main_elements[main_element_positions[i]]
            # 最初の主要素を記録
            if current_group_data['first_main_element'] is None:
                current_group_data['first_main_element'] = main_element
            if main_element.text and main_element.text.strip():
                current_group_data['main_texts'].append(main_element.text.strip())
        else:
            # これは主要素ではない - 前置きタグまたは後付けタグとして指定されている場合のみ処理
            if child.text and child.text.strip():
                # 前置きタグとして指定されている場合
                if prefix_tags and child.tag in prefix_tags:
                    # 現在のグループにメインテキストがある場合、グループを完成させる
                    if current_group_data['main_texts']:
                        group = _build_ordered_group(current_group_data, main_tag, replace_tag, element_id_map)
                        groups.append(group)

                        # 新しいグループを開始（グローバルと固定タグは保持）
                        current_group_data = {
                            'global': current_group_data['global'].copy(),
                            'fixed': current_group_data['fixed'].copy(),
                            'prefix': {},
                            'main_texts': [],
                            'suffix': {},
                            'first_main_element': None
                        }

                    # 新しいグループのセクションタグを設定（連続する場合は\nで連結）
                    if child.tag in current_group_data['prefix']:
                        # 既存の値と新しい値を\nで連結
                        current_group_data['prefix'][child.tag] = current_group_data['prefix'][
                                                                      child.tag] + "\n" + child.text.strip()
                    else:
                        current_group_data['prefix'][child.tag] = child.text.strip()

                    # 属性も追加（連続する場合は\nで連結）
                    if child.attrib:
                        for attr_name, attr_value in child.attrib.items():
                            attr_key = f"{child.tag}@{attr_name}"
                            if attr_key in current_group_data['prefix']:
                                # 既存の値と新しい値を\nで連結
                                current_group_data['prefix'][attr_key] = current_group_data['prefix'][
                                                                             attr_key] + "\n" + attr_value
                            else:
                                current_group_data['prefix'][attr_key] = attr_value

                # 後付けタグとして指定されている場合
                elif suffix_tags and child.tag in suffix_tags:
                    # 後付けタグを現在のグループに追加（連続する場合は\nで連結）
                    if child.tag in current_group_data['suffix']:
                        # 既存の値と新しい値を\nで連結
                        current_group_data['suffix'][child.tag] = current_group_data['suffix'][
                                                                      child.tag] + "\n" + child.text.strip()
                    else:
                        current_group_data['suffix'][child.tag] = child.text.strip()

                    # 属性も追加（連続する場合は\nで連結）
                    if child.attrib:
                        for attr_name, attr_value in child.attrib.items():
                            attr_key = f"{child.tag}@{attr_name}"
                            if attr_key in current_group_data['suffix']:
                                # 既存の値と新しい値を\nで連結
                                current_group_data['suffix'][attr_key] = current_group_data['suffix'][
                                                                             attr_key] + "\n" + attr_value
                            else:
                                current_group_data['suffix'][attr_key] = attr_value

        i += 1

    # 最後のグループを処理
    if current_group_data['main_texts']:
        group = _build_ordered_group(current_group_data, main_tag, replace_tag, element_id_map)
        groups.append(group)

    return groups


def _build_ordered_group(group_data, main_tag=None, replace_tag=None, element_id_map=None):
    """
    グループデータから順序付きの辞書を構築する
    順序: id，グローバルタグ，固定タグ，前置きタグ，主タグ，後付けタグ
    """
    result = {}

    # 最初の主要素を保存（ソート用）
    if group_data.get('first_main_element'):
        result['_first_main_element'] = group_data['first_main_element']

    # 1. グローバルタグ
    result.update(group_data['global'])

    # 2. 固定タグ
    result.update(group_data['fixed'])

    # 3. 前置きタグ
    result.update(group_data['prefix'])

    # 4. 主タグ（テキストを\nで連結）
    if group_data['main_texts']:
        merged_text = "\n".join(group_data['main_texts'])
        # 主タグ名が指定されている場合はそれを使用、そうでなければ"text"をデフォルトとして使用
        main_tag_key = main_tag if main_tag and main_tag.strip() else "text"
        result[main_tag_key] = merged_text

    # 5. 後付けタグ
    result.update(group_data['suffix'])

    # 置換タグの処理
    if replace_tag and replace_tag.strip():
        replacements = {}
        for replacement in replace_tag.split(','):
            if '=' in replacement:
                old_key, new_key = replacement.strip().split('=', 1)
                replacements[old_key.strip()] = new_key.strip()

        if replacements:
            new_result = {}
            for key, value in result.items():
                new_key = replacements.get(key, key)
                new_result[new_key] = value
            result = new_result

    return result
