"""Person 6: figures for the report. Every function saves a PNG and does not open a window."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.confusion_matrix import normalize_rows


def plot_history(history, path):
    """Training and validation loss and accuracy per epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    if "val_loss" in history:
        ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_accuracy"], label="train")
    if "val_accuracy" in history:
        ax_acc.plot(epochs, history["val_accuracy"], label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, path, normalize=True):
    """Heat map of the confusion matrix. Numbers are written in the cells only for small matrices."""
    values = normalize_rows(cm) if normalize else cm
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    if n <= 12:
        for i in range(n):
            for j in range(n):
                text = f"{values[i, j]:.2f}" if normalize else str(values[i, j])
                color = "white" if values[i, j] > values.max() / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_class_f1(per_class, path):
    """Horizontal bars of F1 per class, weakest class at the bottom."""
    ordered = sorted(per_class, key=lambda row: row["f1"], reverse=True)
    names = [row["class"] for row in ordered]
    scores = [row["f1"] for row in ordered]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(names))))
    ax.barh(names, scores)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score")
    ax.set_title("F1-score per class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(X, y_true, y_pred, class_names, path, max_images=12):
    """Grid of wrongly classified test images. X is (N, 1, H, W). Returns False if there are no mistakes."""
    wrong = np.where(np.asarray(y_true) != np.asarray(y_pred))[0][:max_images]
    if len(wrong) == 0:
        return False
    columns = 6
    rows = (len(wrong) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, wrong):
        ax.imshow(X[index, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"true {class_names[y_true[index]]}\npred {class_names[y_pred[index]]}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True