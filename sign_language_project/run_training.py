import argparse
import json
from pathlib import Path

import numpy as np

from cnn.cnn_model import CNN, CNNConfig
from preprocessing.dataset import batch_generator, load_class_info, load_split
from training.optimizer import SGD, SGDMomentum
from training.train import evaluate, train

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--momentum", type=float, default=0.0, help="0 = plain SGD, 0.9 is a good value")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--data_dir", default="data/processed")
parser.add_argument("--out_dir", default="experiments")
args = parser.parse_args()


info = load_class_info(args.data_dir)
img_size = info["img_size"]
X_train, y_train = load_split("train", args.data_dir)
X_val, y_val = load_split("val", args.data_dir)
X_test, y_test = load_split("test", args.data_dir)
print(f"train {X_train.shape}  val {X_val.shape}  test {X_test.shape}  classes {info['num_classes']}")

epoch_counter = {"epoch": 0}


def train_batches():
    epoch_counter["epoch"] += 1
    return batch_generator(X_train, y_train, args.batch_size, augment=True,
                           seed=epoch_counter["epoch"], channels_last=True)


def val_batches():
    return batch_generator(X_val, y_val, 64, shuffle=False, channels_last=True)


def test_batches():
    return batch_generator(X_test, y_test, 64, shuffle=False, channels_last=True)


# ---------- 2. model ----------
np.random.seed(0)
config = CNNConfig(input_shape=(img_size, img_size, 1), num_classes=info["num_classes"])
model = CNN(config)

# ---------- 3. training ----------
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
optimizer = SGDMomentum(args.lr, args.momentum) if args.momentum > 0 else SGD(args.lr)
history = train(model, train_batches, val_batches, epochs=args.epochs,
                optimizer=optimizer, checkpoint_path=out_dir / "model.npz")

# ---------- 4. test ----------
test_result = evaluate(model, test_batches)
print(f"\nTest loss: {test_result['loss']:.4f}   Test accuracy: {test_result['accuracy']:.4f}")

with open(out_dir / "history.json", "w") as f:
    json.dump({"history": history, "test": test_result, "epochs": args.epochs, "learning_rate": args.lr, "momentum": args.momentum}, f, indent=2)
print(f"Saved model and history to {out_dir}/")