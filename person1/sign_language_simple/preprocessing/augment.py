"""
Person 1: simple data augmentation.

Takes a float32 image in [0,1] and returns a slightly changed version.
"""

import numpy as np
from PIL import Image


def random_rotate(image, rng, max_angle=10):
    """Rotate image by a small random angle."""
    angle = rng.uniform(-max_angle, max_angle)
    pil_img = Image.fromarray((image * 255).astype(np.uint8))
    pil_img = pil_img.rotate(angle)
    return np.array(pil_img, dtype=np.float32) / 255.0


def random_brightness(image, rng, max_change=0.1):
    """Make the image a little brighter or darker."""
    change = rng.uniform(-max_change, max_change)
    new_image = image + change
    return np.clip(new_image, 0.0, 1.0)


def augment_batch(X, rng):
    """Augment a batch of images, shape (N,1,H,W)."""
    new_X = np.empty_like(X)
    for i in range(len(X)):
        image = X[i, 0]
        image = random_rotate(image, rng)
        image = random_brightness(image, rng)
        new_X[i, 0] = image
    return new_X
