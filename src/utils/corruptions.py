from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def _to_uint8_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")

def gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
    """
    severity: 1-5
    """
    img = _to_uint8_rgb(img)
    x = np.asarray(img).astype(np.float32) / 255.0
    #sigma increases with severity
    sigmas = [0.02, 0.04, 0.06, 0.08, 0.10]
    sigma = sigmas[severity - 1]
    noise = np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)
    y = np.clip(x + noise, 0.0, 1.0)
    y = (y * 255.0).astype(np.uint8)
    return Image.fromarray(y)

def gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
    """
    severity: 1-5
    """
    img = _to_uint8_rgb(img)
    radii = [0.5, 1.0, 1.5, 2.0, 2.5]
    r = radii[severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=r))

def brightness(img: Image.Image, severity: int) -> Image.Image:
    """
    severity: 1-5, reduce brightness progressively
    """
    img = _to_uint8_rgb(img)
    factors = [0.9, 0.75, 0.6, 0.45, 0.3]
    f = factors[severity - 1]
    return ImageEnhance.Brightness(img).enhance(f)

def jpeg_compression(img: Image.Image, severity: int) -> Image.Image:
    """
    severity: 1-5, lower JPEG quality progressively
    """
    img = _to_uint8_rgb(img)
    qualities = [70, 50, 35, 20, 10]
    q = qualities[severity - 1]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

CORRUPTIONS: Dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "gaussian_noise": gaussian_noise,
    "blur": gaussian_blur,
    "brightness": brightness,
    "jpeg": jpeg_compression,
}