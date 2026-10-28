import json
from pathlib import Path
import numpy as np


def save_checkpoint(model, path, epoch=None, history=None):
    pairs = model.parameters()
    arrays = {}
    shapes = []
    for i, (parameter, _gradient) in enumerate(pairs):
        arrays[f"param_{i}"] = np.asarray(parameter)
        shapes.append(list(parameter.shape))
    metadata = {"epoch": epoch, "parameter_shapes": shapes, "history": history or {}}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # File handle prevents NumPy silently appending .npz to the requested path.
    with target.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    return target


def load_checkpoint(model, path):
    pairs = model.parameters()
    with np.load(Path(path), allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"].item()))
        expected_shapes = [list(parameter.shape) for parameter, _ in pairs]
        if metadata.get("parameter_shapes") != expected_shapes:
            raise ValueError("Checkpoint parameter shapes do not match this model architecture.")
        saved_keys = [f"param_{i}" for i in range(len(pairs))]
        if any(key not in checkpoint for key in saved_keys):
            raise ValueError("Checkpoint is missing one or more parameter arrays.")
        saved = [checkpoint[key] for key in saved_keys]
        for (parameter, _), value in zip(pairs, saved):
            if parameter.shape != value.shape:
                raise ValueError("Checkpoint parameter shape mismatch.")
            if not np.all(np.isfinite(value)):
                raise ValueError("Checkpoint contains non-finite parameters.")
        for (parameter, _), value in zip(pairs, saved):
            parameter[...] = value
    return metadata
