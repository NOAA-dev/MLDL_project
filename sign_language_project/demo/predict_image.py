"""Person 6: predict the sign in one image.

Run from the project root:
    python -m demo.predict_image path/to/image.jpg
"""

import argparse

import numpy as np

from evaluation.metrics import load_trained_model
from preprocessing.dataset import load_class_info, to_channels_last
from preprocessing.preprocess_images import preprocess_for_inference


def predict_image(model, image, class_names, top_k=3):
    """image: file path, PIL image or numpy array. Returns [(class_name, probability), ...] best first."""
    img_size = model.cfg.input_shape[0]
    x = to_channels_last(preprocess_for_inference(image, img_size))
    probabilities = model.forward(x)[0]
    best = np.argsort(probabilities)[::-1][:top_k]
    return [(class_names[i], float(probabilities[i])) for i in best]


def main():
    parser = argparse.ArgumentParser(description="Predict the sign in one image")
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--checkpoint", default="experiments/model.npz")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    class_info = load_class_info(args.data_dir)
    model = load_trained_model(args.checkpoint, class_info)
    for name, probability in predict_image(model, args.image, class_info["class_names"], args.top):
        print(f"{name:10s} {probability * 100:5.1f}%")


if __name__ == "__main__":
    main()