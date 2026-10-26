import matplotlib.pyplot as plt
from preprocessing.dataset import load_split, load_class_info

info = load_class_info()
X, y = load_split("train")

fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for ax, i in zip(axes.flat, range(0, len(X), len(X) // 30)):
    ax.imshow(X[i, 0], cmap="gray")
    ax.set_title(info["class_names"][y[i]])
    ax.axis("off")
plt.savefig("sample_images.png")
plt.show()