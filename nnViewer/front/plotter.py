from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from io import BytesIO

from matplotlib import patches


def plot_fully_connected():
    layer1_y = [1, 2, 3, 4, 5]  # 4 neurones en haut
    layer2_y = [1.7, 3, 4.3]  # 3 neurones en bas

    layer1_x = [1] * len(layer1_y)
    layer2_x = [3] * len(layer2_y)
    fig, ax = plt.subplots()

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.scatter(layer1_x, layer1_y, s=20, c='black', marker='o')

    ax.scatter(layer2_x, layer2_y, s=20, c='black', marker='o')

    for y1 in layer1_y:
        for y2 in layer2_y:
            ax.plot([1, 3], [y1, y2], 'k-', lw=0.7)

    plt.xlim(0.7, 3.3)
    plt.ylim(0.7, 5.3)

    ax.set_aspect('equal', adjustable='box')

    ax.axis('off')

    fig.set_size_inches(0.5, 0.7)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    buf.close()

    return QGraphicsPixmapItem(pixmap)

def plot_transpose():
    width = 3
    height = 5
    eps1 = 0.1
    eps2 = 0.3

    fig, ax = plt.subplots()

    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    rect1 = patches.Rectangle((0, width + eps1), width, height, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect1)

    rect2 = patches.Rectangle((width + eps1, 0), height, width, linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect2)

    arrow_start = (width + eps2, width + height * 0.85 + eps2)
    arrow_end = (width + height * 0.85 , width + eps2)

    # Dessin de la flèche avec un arc de cercle
    arrow = patches.FancyArrowPatch(arrow_start,
                                    arrow_end,
                                    arrowstyle='-|>',
                                    mutation_scale=5,
                                    connectionstyle='arc3,rad=-0.3',
                                    color='black', linewidth=1)
    ax.add_patch(arrow)

    ax.set_aspect('equal')
    plt.xlim(-0.1, width + height + 0.3)
    plt.ylim(-0.1, width + height + 0.3)
    ax.axis('off')

    fig.set_size_inches(0.3, 0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buf.getvalue())
    buf.close()

    pixmap_item = QGraphicsPixmapItem(pixmap)
    return pixmap_item

if __name__ == '__main__':
    plot_transpose()

