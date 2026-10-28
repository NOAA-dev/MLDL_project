import numpy as np
from PIL import Image

AUGMENT_DEFAULTS = {
    "max_rotation": 10.0,         # degrees
    "max_shift": 0.10,            # fraction of image size
    "zoom_range": (0.9, 1.1),
    "contrast_range": (0.8, 1.2),
    "max_brightness": 0.10,
}


def random_affine(image, rng, max_rotation, max_shift, zoom_range):
    h, w = image.shape
    angle = np.deg2rad(rng.uniform(-max_rotation, max_rotation))
    tx = rng.uniform(-max_shift, max_shift) * w
    ty = rng.uniform(-max_shift, max_shift) * h
    zoom = rng.uniform(*zoom_range)

    # PIL maps each output pixel back to an input pixel, so we use the inverse transform
    s = 1.0 / zoom
    cos, sin = np.cos(angle), np.sin(angle)
    cx, cy = w / 2.0, h / 2.0
    a, b = s * cos, s * sin
    d, e = -s * sin, s * cos
    c = cx - (a * (cx + tx) + b * (cy + ty))
    f = cy - (d * (cx + tx) + e * (cy + ty))

    pil_image = Image.fromarray(image, mode="F")
    out = pil_image.transform((w, h), Image.AFFINE, (a, b, c, d, e, f),
                              resample=Image.BILINEAR, fillcolor=float(image.mean()))
    return np.array(out, dtype=np.float32)


def random_brightness_contrast(image, rng, contrast_range, max_brightness):
    contrast = rng.uniform(*contrast_range)
    brightness = rng.uniform(-max_brightness, max_brightness)
    out = (image - 0.5) * contrast + 0.5 + brightness
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def augment_batch(X, rng, **overrides):
    if X.ndim != 4 or X.shape[1] != 1:
        raise ValueError(f"Expected X of shape (N, 1, H, W), got {X.shape}")
    params = {**AUGMENT_DEFAULTS, **overrides}

    out = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        image = X[i, 0].astype(np.float32)
        image = random_affine(image, rng, params["max_rotation"], params["max_shift"], params["zoom_range"])
        image = random_brightness_contrast(image, rng, params["contrast_range"], params["max_brightness"])
        out[i, 0] = image
    return out
