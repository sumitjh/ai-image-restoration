import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


_sr_instances: dict = {}


def get_sr_model(scale: int = 4) -> RealESRGANer:
    if scale not in _sr_instances:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32,
            scale=scale
        )
        model_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            if scale == 4 else
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        )
        _sr_instances[scale] = RealESRGANer(
            scale=scale,
            model_path=model_url,
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
        )
    return _sr_instances[scale]


def enhance_sr(image: np.ndarray, scale: int = 4) -> np.ndarray:
    if scale not in (2, 4):
        raise ValueError("Scale must be 2 or 4")
    upsampler = get_sr_model(scale)
    enhanced, _ = upsampler.enhance(image, outscale=scale)
    return enhanced
