from PIL import Image
import base64
from io import BytesIO


def get_image_media_type(image_path):
    """
    Determine the media type of an image based on its file extension.

    Args:
    image_path (str): The path to the image file.

    Returns:
    str: The media type of the image.

    Raises:
    ValueError: If the image format is not supported (only JPEG and PNG are supported).
    """
    # Determine the media type based on the image extension
    image_extension = image_path.lower().split('.')[-1]
    if image_extension in ['jpg', 'jpeg']:
        return "image/jpeg"
    elif image_extension == 'png':
        return "image/png"
    else:
        raise ValueError("Unsupported image format. Only JPEG and PNG are supported.")


def image_to_base64(image_path, output_format='JPEG', quality=20, include_data_url=False):
    # Open and convert the image
    image = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format=output_format, quality=quality)

    # Encode the image to base64
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Optionally include the data URL
    if include_data_url:
        mime_type = f"image/{output_format.lower()}"
        return f"data:{mime_type};base64,{base64_string}"
    else:
        return base64_string


def image_to_base64_url(image_path):
    with open(image_path, 'rb') as file:
        binary_data = file.read()
        base64_encoded_data = base64.b64encode(binary_data)
        base64_string = base64_encoded_data.decode('utf-8')

        # Determine the image format from the file extension
        file_extension = image_path.split('.')[-1].lower()
        mime_type = f"image/{file_extension}"

        # Create the data URL
        base64_url = f"data:{mime_type};base64,{base64_string}"
    return base64_url


def combine_images(image1_path, image2_path, output_path, direction='vertical'):
    # 打开两张图片
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")

    # 获取图片的尺寸
    image1_width, image1_height = image1.size
    image2_width, image2_height = image2.size

    # 确定新图片的尺寸
    if direction == 'horizontal':
        new_width = image1_width + image2_width
        new_height = max(image1_height, image2_height)
    elif direction == 'vertical':
        new_width = max(image1_width, image2_width)
        new_height = image1_height + image2_height
    else:
        raise ValueError("Direction should be either 'horizontal' or 'vertical'")

    # 创建一个新的空白图片
    new_image = Image.new('RGB', (new_width, new_height))

    # 将两张图片粘贴到新图片上
    if direction == 'vertical':
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (0, image1_height))
    elif direction == 'horizontal':
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1_width, 0))

    # 保存新图片
    new_image.save(output_path)
    print(f"Combined image saved to {output_path}")
