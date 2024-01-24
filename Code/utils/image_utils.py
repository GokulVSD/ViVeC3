from torchvision.transforms import transforms
from matplotlib import pyplot as plt
from math import ceil


def convert_to_rgb(pil_img):
    """
    Convert image that isn't RGB to RGB.
    """
    if pil_img.mode != 'RGB':
        return pil_img.convert('RGB')
    return pil_img


def partition_to_grid(pil_img, num_rows, num_cols, hor_pixels, ver_pixels):
    """
    Partition input image into patches. It is expected that the image should
    be partitionable with the provided parameters.
    """

    img_tensor = transforms.Compose([transforms.PILToTensor()])(pil_img)

    grid = []

    for i in range(num_rows):
        grid.append([])
        for j in range(num_cols):
            grid[-1].append(
                img_tensor[:, ver_pixels*i : ver_pixels*(i+1), hor_pixels*j : hor_pixels*(j+1)]
            )

    return grid


def display_images(img, title_text=""):
    # pil_img = convert_to_pil_image(img)
    plt.imshow(img)
    if len(title_text) > 0:
        plt.title(title_text)


def save_image(img, path, title="", show=True):
    if show:
        display_images(img, title)
    plt.savefig(path)



def plot_2d_scatter_with_color(X, Y, labels, title):
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(x=X, y=Y, c=labels, s=[200]*len(labels), cmap='Spectral', linewidth=1, edgecolor='k')
    leg = ax.legend(*scatter.legend_elements(), loc="best", title="Cluster")
    ax.add_artist(leg)
    ax.set_title(title)
    plt.show()


def plot_image_thumbnails(title, images, max_cols=10):
    ncols = min(max_cols, len(images))
    nrows = ceil(len(images)/ncols)
    fig, axes = plt.subplots(figsize=(ncols*2, nrows*2), ncols=ncols, nrows=nrows, sharex=True, sharey=True)
    if nrows == 1 and ncols == 1:
         axes = [axes]
    else:
         axes = axes.flatten()
    for i, ax in enumerate(axes):
            if i < len(images):
                ax.axis('off')
                ax.imshow(images[i])
            else:
                ax.set_visible(False)

    fig.suptitle(title)
    plt.show()