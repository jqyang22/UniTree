import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

def display_images(img, titles=None, cmap=None, norm=None, interpolation=None, savefig=None, savename=None):
    cols = img.shape[-1]
    rows = img.shape[0]
    titles = titles if titles is not None else [""] * (rows * cols)
    plt.figure(figsize=(14, 14 * rows // cols))
    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows, cols, (i * cols) + j + 1)
            plt.axis('off')
            plt.imshow(img[i, ..., j], cmap=cmap, norm=norm, interpolation=interpolation)
            plt.title(titles[j])
    if savefig and savename:
        plt.savefig(savename)
