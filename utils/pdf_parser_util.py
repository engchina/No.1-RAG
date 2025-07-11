import os

import anthropic
import fitz  # PyMuPDF
from dotenv import load_dotenv, find_dotenv
from langchain.docstore.document import Document

from utils.image_util import image_to_base64, get_image_media_type, combine_images

# Load environment variables from .env file
load_dotenv(find_dotenv())


def __analyze_image(image_path, image_base64, language="English", model="claude-3-5-sonnet-20240620", max_tokens=1024):
    """
    Analyze an image using Anthropic's API.

    Args:
        image_path (str): The path of the image to analyze.
        model (str): The Anthropic model to use for analysis. Default is "claude-3-5-sonnet-20240620".
        max_tokens (int): The maximum number of tokens for the response. Default is 1024.

    Returns:
        str: The response from Anthropic's API.
    """
    # Determine the media type based on the image extension
    image_media_type = get_image_media_type(image_path)

    # Create the message using Anthropic's API
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=f"You are an expert in describing the responses of linearized tables into plain {language} text for LLMs to use. \
        Explain the following linearized table.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_base64,
                        },
                    }
                ],
            }
        ],
    )

    return message.content[0].text


def __convert_image_to_markdown(image_path, image_base64, model="claude-3-5-sonnet-20240620", max_tokens=4096):
    prompt = f"""You are an AI assistant tasked with analyzing one or more images of tables and generating markdown-formatted content based on the images. Follow these instructions carefully:

1. You will be provided with one or more images of tables. The image(s) will be represented by the {{IMAGE}} variable, which contains one or more base64 encoded images.

2. Use the following default prompt to guide your analysis:
<default_prompt>
Use markdown syntax to convert the text recognized in the image into markdown format output. You must:
1. Output in the same language as the recognized text in the image. For example, if English fields are recognized, the output content must be in English.
2. Do not explain or output irrelevant text, directly output the content in the image. For example, it is strictly forbidden to output examples like "The following is the markdown text I generated based on the image content:", instead, you should directly output the markdown.
3. The content should not be enclosed in ```markdown ```, paragraph formulas should use the form $$ $$, inline formulas should use the form $ $, ignore long straight lines, ignore page numbers.
Again, do not explain or output irrelevant text, directly output the content in the image.
</default_prompt>

3. Analyze the provided image(s) in {{IMAGE}} according to these steps:
   a. If there is only one image, proceed to analyze it directly.
   b. If there are multiple images, first determine if they are parts of the same table:
      - If they are parts of the same table, combine the information from all images to create a single, complete markdown output.
      - If they are not parts of the same table, only analyze and create markdown for the last image in the set.

4. When generating the markdown-formatted content based on your analysis, ensure that you:
   - Use appropriate markdown syntax for tables, headers, and text formatting
   - Use LaTeX syntax for any mathematical formulas or equations
   - Include any areas marked with red boxes, if present
   - Maintain the original language of the text in the image
   - Do not add any explanatory text or comments outside of the actual content from the image(s)

5. Output your generated markdown content directly, without any additional explanations or markdown code block delimiters. Use the following format:

[Your generated markdown content here, starting immediately without any preamble]

Remember to analyze the structure of the table(s), the text content, and any specially marked areas in the image(s). Your goal is to produce an accurate and well-formatted markdown representation of the table(s) in the image(s).
    """

    image_media_type = get_image_media_type(image_path)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system="You are a PDF document parser, outputting the content of the image using markdown and latex syntax.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_base64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    return message.content[0].text


def extract_surrounding_text(text_blocks, bbox, page_height, threshold_percentage=0.1):
    """
    Extract text before and after a given bounding box on a page.

    Args:
        text_blocks (list): List of text blocks on the page.
        bbox (fitz.Rect): Bounding box of the item.
        page_height (float): Height of the page.
        threshold_percentage (float): Percentage threshold for determining proximity.

    Returns:
        tuple: Text before and after the bounding box.
    """
    before_text, after_text = "", ""
    vertical_threshold = page_height * threshold_percentage
    horizontal_threshold = bbox.width * threshold_percentage

    for block in text_blocks:
        block_bbox = fitz.Rect(block[:4])
        vertical_distance = min(abs(block_bbox.y1 - bbox.y0), abs(block_bbox.y0 - bbox.y1))
        horizontal_overlap = max(0, min(block_bbox.x1, bbox.x1) - max(block_bbox.x0, bbox.x0))

        if vertical_distance <= vertical_threshold and horizontal_overlap >= -horizontal_threshold:
            if block_bbox.y1 < bbox.y0 and not before_text:
                before_text = block[4]
            elif block_bbox.y0 > bbox.y1 and not after_text:
                after_text = block[4]
                break

    return before_text, after_text


def parse_tables_on_page(filepath, page, page_number, text_blocks, prev_page_last_table_doc):
    """
    Parse all tables on a given page and return table documents and bounding boxes.

    Args:
        filepath (str): Path to the PDF file.
        page (fitz.Page): The current page of the PDF.
        page_number (int): The page number.
        text_blocks (list): List of text blocks on the page.

    Returns:
        tuple: List of table documents, list of bounding boxes, list of tables.
    """
    output_dir = './cropped_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    table_docs = []
    table_bboxes = []
    tables = []

    try:
        tables = page.find_tables(horizontal_strategy="lines_strict", vertical_strategy="lines_strict")
    except Exception as e:
        print(f"Error during table extraction on page {page_number}: {e}")
        return table_docs, table_bboxes, tables

    for idx, table in enumerate(tables, start=1):
        prev_table_doc = None
        prev_table = None
        if idx == 1 and prev_page_last_table_doc:
            prev_table_doc = prev_page_last_table_doc
            prev_table = prev_page_last_table_doc.metadata["table"]
        bbox = fitz.Rect(table.bbox)
        table_bboxes.append(bbox)

        before_text, after_text = extract_surrounding_text(text_blocks, bbox, page.rect.height)
        if prev_table_doc and prev_table:
            try:
                if prev_table.to_pandas().shape[1] != table.to_pandas().shape[1]:
                    prev_table_doc = None
                if len(prev_table_doc.metadata["after_text"].strip()) > 0 or len(before_text.strip()) > 0:
                    prev_table_doc = None
            except (IndexError, AttributeError):
                print("here-3")
                prev_table_doc = None

        # page.draw_rect(bbox, color=(1, 0, 0), width=2)
        table_id = f"page_{page_number + 1}_table_{idx}"
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=bbox)
        image_path = os.path.join(output_dir, f"{table_id}.jpg")
        pixmap.save(image_path)
        image_base64 = image_to_base64(image_path)
        combined_image_path = image_path
        combined_image_base64 = image_base64
        if prev_table_doc:
            combined_image_path = os.path.join(output_dir, f"page_{page_number + 1}_table_{idx}_combined.jpg")
            combine_images(prev_table_doc.metadata["combined_image_path"], image_path, combined_image_path)
            combined_image_base64 = image_to_base64(combined_image_path)
            description = __analyze_image(combined_image_path, image_base64=combined_image_base64, language="Japanese")
            # description = "okay"
            print(f"{description=}")
            caption = prev_table_doc.metadata["before_text"].replace("\n", " ") + description + after_text.replace("\n",
                                                                                                                   " ")
        else:
            description = __analyze_image(combined_image_path, image_base64=combined_image_base64, language="Japanese")
            # description = "okay"
            print(f"{description=}")
            caption = before_text.replace("\n", " ") + description + after_text.replace("\n", " ")
        if not caption:
            caption = " ".join(table.header.names)

        table_to_markdown = __convert_image_to_markdown(combined_image_path, combined_image_base64)
        # table_to_markdown = table.to_markdown()
        print(f"{table_to_markdown=}")

        table_metadata = {
            "type": "table",
            "header": table.header.names,
            "caption": caption,
            "page_num": page_number,
            "before_text": before_text,
            "after_text": after_text,
            "image_path": image_path,
            "image_base64": image_base64,
            "combined_image_path": combined_image_path,
            "combined_image_base64": combined_image_base64,
            "table_id": table_id,
            "table": table,
            "table_to_markdown": table_to_markdown,
            "source": f"{filepath[:-4]}-page{page_number}-table{idx}",
        }

        doc = Document(
            page_content=f"This is a table with the caption: {caption}\n",
            metadata=table_metadata)
        table_docs.append(doc)

        idx += 1

    return table_docs, table_bboxes, tables


def parser_pdf(filepath, output_dir='./cropped_images') -> dict:
    """
    Parser a PDF file, mark tables, and extract table information.

    Args:
        filepath (str): Path to the PDF file.
        output_dir (str): Directory to save marked images.

    Returns:
        list: List of extracted table documents.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        pdf_document = fitz.open(filepath)
    except Exception as e:
        print(f"Error opening PDF file {filepath}: {e}")
        return []

    all_table_documents = []

    prev_page_last_table_doc = None
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]

        # Filter out header and footer text blocks
        page_height = page.rect.height
        header_threshold, footer_threshold = page_height * 0.1, page_height * 0.9
        text_blocks = [block for block in page.get_text("blocks", sort=True)
                       if block[-1] == 0 and not (block[1] < header_threshold or block[3] > footer_threshold)]

        table_docs, table_bboxes, tables = parse_tables_on_page(filepath, page, page_number, text_blocks,
                                                                prev_page_last_table_doc)
        prev_page_last_table_doc = table_docs[-1] if table_docs else None
        all_table_documents.extend(table_docs)

    page_table_documents = {}
    for table_document in all_table_documents:
        page_table_documents[table_document.metadata["table_id"]] = {"page_content": table_document.page_content,
                                                                     "table_to_markdown": table_document.metadata[
                                                                         "table_to_markdown"]}

    pdf_document.close()
    return page_table_documents
