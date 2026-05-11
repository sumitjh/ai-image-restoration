import io
import numpy as np
from PIL import Image


MAX_IMAGE_SIZE_MB = 10
MAX_DIMENSION = 2000


def validate_image(image_bytes: bytes) -> tuple[bool, str]:
    if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        return False, f"Image exceeds {MAX_IMAGE_SIZE_MB}MB limit"
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if max(img.size) > MAX_DIMENSION:
            return False, f"Image dimensions exceed {MAX_DIMENSION}px limit"
    except Exception:
        return False, "Invalid or corrupted image file"
    return True, ""


def load_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def image_to_bytes(image: np.ndarray, format: str = "PNG") -> bytes:
    pil_img = Image.fromarray(image.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


def compute_psnr(lr_image: np.ndarray, sr_image: np.ndarray, scale: int) -> float:
    # Upscale LR to HR domain via bicubic — fair comparison in same domain
    h, w = lr_image.shape[:2]
    bicubic = np.array(
        Image.fromarray(lr_image).resize((w * scale, h * scale), Image.BICUBIC)
    )
    mse = np.mean((bicubic.astype(float) - sr_image.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))
