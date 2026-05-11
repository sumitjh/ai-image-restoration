import numpy as np
from gfpgan import GFPGANer


_face_instance = None


def get_face_model() -> GFPGANer:
    global _face_instance
    if _face_instance is None:
        _face_instance = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=2,
            arch="clean",
            channel_multiplier=2,
        )
    return _face_instance


def enhance_face(image: np.ndarray) -> np.ndarray:
    restorer = get_face_model()
    _, _, enhanced = restorer.enhance(
        image,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
    )
    return enhanced
