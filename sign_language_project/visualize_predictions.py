import argparse

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import json

from PIL import Image

from cnn.cnn_model import CNN, CNNConfig
from evaluation.metrics import predict_all
from training.checkpoints import load_checkpoint
from preprocessing.dataset import load_class_info, load_split

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="experiments/final/model.npz")
parser.add_argument("--data_dir", default="data/processed")
parser.add_argument("--num", type=int, default=10)
parser.add_argument("--cols", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="prediction_samples.png")
parser.add_argument("--no_show", action="store_true")
args = parser.parse_args()

if args.no_show:
    matplotlib.use("Agg")

class_info = load_class_info(args.data_dir)
class_names = class_info["class_names"]

def build_model(checkpoint_path, class_info):
    with np.load(checkpoint_path, allow_pickle=False) as ckpt:
        shapes = json.loads(str(ckpt["metadata_json"].item()))["parameter_shapes"]
    conv_channels = tuple(s[3] for s in shapes if len(s) == 4)
    dense_inputs = [s for s in shapes if len(s) == 2][0][0]
    side = int(round(np.sqrt(dense_inputs / conv_channels[-1]))) * 2 ** len(conv_channels)
    config = CNNConfig(input_shape=(side, side, 1), num_classes=class_info["num_classes"],
                       conv_channels=conv_channels)
    model = CNN(config)
    load_checkpoint(model, checkpoint_path)
    return model, side


def resize_batch(X, side):
    if X.shape[-1] == side:
        return X
    resized = [np.asarray(Image.fromarray((img[0] * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR),
                          dtype=np.float32) / 255 for img in X]
    return np.stack(resized)[:, None]


model, model_size = build_model(args.checkpoint, class_info)
X_test, y_test = load_split("test", args.data_dir)

rng = np.random.default_rng(args.seed)
idx = rng.choice(len(X_test), size=args.num, replace=False)
X, y = X_test[idx], y_test[idx]

y_pred, probs = predict_all(model, resize_batch(X, model_size))
confidence = probs.max(axis=1)

cols = args.cols
rows = int(np.ceil(args.num / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 9), constrained_layout=True)
for i, ax in enumerate(axes.flat):
    if i >= args.num:
        ax.axis("off")
        continue
    correct = y_pred[i] == y[i]
    mark = "\u2713" if correct else "\u2717"
    ax.imshow(X[i, 0], cmap="gray")
    ax.set_title(f"Actual: {class_names[y[i]]}\nPred: {class_names[y_pred[i]]} ({confidence[i] * 100:.0f}%) {mark}",
                 color="black", fontsize=10, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("tab:blue" if correct else "tab:red")
        spine.set_linewidth(3)

accuracy = (y_pred == y).mean()
fig.suptitle(f"Actual vs predicted ({accuracy * 100:.0f}% correct in this sample)", fontsize=14)
plt.savefig(args.out, dpi=120)
print(f"Saved {args.out}")
if not args.no_show:
    plt.show()
