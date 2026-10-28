"""Persistent product photos for display only; never modify CLIP features."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tempfile

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000


def image_path(data: Path, asin: str) -> Path:
    # The original filename and ASIN never become a filesystem path component.
    return Path(data) / 'demo_images' / (sha256(asin.encode('utf-8')).hexdigest() + '.jpg')


def save_image(catalog, asin: str, payload: bytes) -> Path:
    if asin not in catalog.lookup:
        raise ValueError('Sản phẩm không có trong catalog.')
    if not 0 < len(payload) <= MAX_IMAGE_BYTES:
        raise ValueError('Chọn ảnh có dung lượng tối đa 5 MB.')
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(BytesIO(payload)) as original:
            if original.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise ValueError('Chỉ hỗ trợ ảnh JPG, PNG hoặc WebP.')
            if original.width * original.height > MAX_IMAGE_PIXELS:
                raise ValueError('Ảnh quá lớn; hãy chọn ảnh không quá 12 megapixel.')
            photo = ImageOps.exif_transpose(original).convert('RGBA')
            photo.thumbnail((1600, 1600))
            background = Image.new('RGB', photo.size, 'white')
            background.paste(photo, mask=photo.getchannel('A'))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('File không phải ảnh hợp lệ hoặc ảnh bị hỏng.') from error

    target = image_path(catalog.data, asin)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tmp', delete=False) as out:
            temporary = Path(out.name)
            background.save(out, format='JPEG', quality=90)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
