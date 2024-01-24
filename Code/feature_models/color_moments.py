from torch import tensor
from utils.image_utils import convert_to_rgb, partition_to_grid


class ColorMomentsExtractor:
    """
    Color moments capture the distribution of intensities of each channel (R, G, B).
    In essense, we will be calculating the mean, standard deviation, and skewness of
    each 10x30 (HxW) grid cell of which there are 10x10, for each channel, of which there are 3.
    This results in a feature descriptor of length 10x10x3x3 = 900.

    Each cell vector's shape is 3 x 10 x 30 (C x H x W)
    https://en.wikipedia.org/wiki/Color_moments
    """

    def __init__(self, image):
        """
        Preprocess PIL image for color moments extraction.
        """
        # Convert image that isn't RGB to RGB.
        image = convert_to_rgb(image)

        # Resize image to 300x100.
        image300x100 = image.resize((300, 100))

        # Partition image into 10x10 grid.
        self.grid = partition_to_grid(image300x100, num_rows=10, num_cols=10, hor_pixels=30, ver_pixels=10)


    def get_color_vector(self):
        """
        Return color moments represented as a 900-dim torch vector.
        The sequential vector representation is as follows:

        Row-major order of cells (10x10 cells), with each cell having 3 channel (RGB in that order),
        with each channel having 3 values (mean, standard deviation, skewness in that order).

        vec: [<cell_1 <red <mean, stdev, skew>>, <green <...>>, <blue <...>>>, <cell_2 ...>...]
        """

        color_descriptor = []

        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):

                cell = self.grid[i][j]

                cell_descriptor = []

                for channel in cell:

                    channel_descriptor = []

                    mean = channel.float().mean()
                    channel_descriptor.append(mean)

                    diffs = (channel - mean)

                    # As per the source, standard deviation is using population standard deviation
                    # as opposed to sample standard deviation (N in the denominator instead
                    # of N-1). In other words, it does not use Bessel's correction:
                    # https://en.wikipedia.org/wiki/Bessel%27s_correction
                    std = diffs.pow(2).mean().pow(1/2)
                    channel_descriptor.append(std)

                    # This is not Fisher-Pearson Skewness, it is what is expressed on the Wiki
                    # page as well as the lecture notes. The reason why I'm doing cube root
                    # this way is due to NaNs when taking cube roots of negatives in PyTorch
                    # tensors: https://github.com/pytorch/pytorch/issues/25766
                    cube_diffs_mean = diffs.pow(3).mean()
                    skew = cube_diffs_mean.sign() * cube_diffs_mean.abs().pow(1/3)
                    channel_descriptor.append(skew)

                    cell_descriptor.extend(channel_descriptor)

                color_descriptor.extend(cell_descriptor)

        return tensor(color_descriptor)