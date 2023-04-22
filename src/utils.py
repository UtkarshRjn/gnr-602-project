import matplotlib.pyplot as plt
import numpy as np
import os


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def save_fig(img, name, dir="", keep_axis=True, num_labels=0, isLabel=False):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    if isLabel:
        cmap = plt.get_cmap("Set2", num_labels)
    else:
        cmap = plt.get_cmap("gray")

    ax.imshow(img, interpolation="nearest")
    # ax.imshow(img, interpolation="nearest", cmap=cmap)
    if not keep_axis:
        ax.set_axis_off()


    if not os.path.exists(dir):
        os.makedirs(dir)

    plt.savefig(
        os.path.join(dir, name),
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
        edgecolor="none",
        facecolor="none",
    )

    return os.path.join(dir, name)
