"""Evaluate the trained CNN on the test set and save all results for the report.

Run from the project root, after run_training.py:
    python evaluate_model.py
"""

import argparse
import csv
import json
from pathlib import Path

from evaluation.confusion_matrix import compute_confusion_matrix, top_confusions
from evaluation.metrics import accuracy_score, class_report, load_trained_model, measure_inference_time, predict_all
from evaluation.plots import plot_class_f1, plot_confusion_matrix, plot_history, plot_misclassified
from preprocessing.dataset import load_class_info, load_split

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="experiments/model.npz")
parser.add_argument("--history", default="experiments/history.json")
parser.add_argument("--data_dir", default="data/processed")
parser.add_argument("--out_dir", default="experiments/results")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ---------- 1. model and test data ----------
class_info = load_class_info(args.data_dir)
class_names = class_info["class_names"]
model = load_trained_model(args.checkpoint, class_info)
X_test, y_test = load_split("test", args.data_dir)

# ---------- 2. predictions and metrics ----------
y_pred, probabilities = predict_all(model, X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = compute_confusion_matrix(y_test, y_pred, class_info["num_classes"])
per_class, macro, weighted = class_report(cm, class_names)
confusions = top_confusions(cm, class_names, top_n=10)
timing = measure_inference_time(model, X_test)

# ---------- 3. save results ----------
with open(out_dir / "class_report.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
    writer.writeheader()
    writer.writerows(per_class)

results = {"test_accuracy": accuracy, "macro_average": macro, "weighted_average": weighted,
           "inference_time": timing, "top_confusions": confusions, "test_images": int(len(y_test))}
with open(out_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
plot_class_f1(per_class, out_dir / "f1_per_class.png")
had_mistakes = plot_misclassified(X_test, y_test, y_pred, class_names, out_dir / "misclassified_examples.png")
if Path(args.history).exists():
    with open(args.history) as f:
        plot_history(json.load(f)["history"], out_dir / "training_curves.png")

# ---------- 4. print summary ----------
print(f"Test images: {len(y_test)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro precision / recall / F1: {macro['precision']:.4f} / {macro['recall']:.4f} / {macro['f1']:.4f}")
print(f"Weighted F1: {weighted['f1']:.4f}")
print("\nWeakest classes by F1:")
for row in sorted(per_class, key=lambda r: r["f1"])[:5]:
    print(f"  {row['class']:8s} precision {row['precision']:.2f}  recall {row['recall']:.2f}  f1 {row['f1']:.2f}  ({row['support']} images)")
print("\nMost common mistakes:")
for row in confusions[:5]:
    print(f"  true {row['true']:8s} -> predicted {row['predicted']:8s} {row['count']} times ({row['share_of_true_class'] * 100:.1f}% of that class)")
print(f"\nInference time: {timing['single_image_ms']:.2f} ms per single image, "
      f"{timing['batch_ms_per_image']:.2f} ms per image in a batch of {timing['batch_size']}")
print(f"\nResults saved to {out_dir}/" + ("" if had_mistakes else " (no misclassified images to plot)"))