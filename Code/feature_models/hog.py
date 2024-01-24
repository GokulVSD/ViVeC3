from numpy import arctan2, array, ceil, correlate, floor, power, rad2deg, sqrt, transpose
from torch import tensor
from torchvision.transforms import Grayscale
from utils.image_utils import convert_to_rgb, partition_to_grid


class HOGExtractor:
    """
    9 bin (signed) magnitude-weighted gradient histograms.
    Signed implies we consider 360 degrees, hence 9 bins => 40 degree bins.
    https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    https://medium.com/@skillcate/histogram-of-oriented-gradients-hog-simplest-intuition-2392995f8010
    """

    def __init__(self, image):
        """
        Preprocess PIL image for HOG extraction.
        """
        # Convert image that isn't RGB to RGB. We anyway convert to grayscale later,
        # but this step is meant for consistency, as the Grayscale transformer might
        # process non-RGB differently.
        image = convert_to_rgb(image)

        # Resize image to 300x100.
        image300x100 = image.resize((300, 100))

        gs_image300x100 = Grayscale()(image300x100)

        # Partition image into 10x10 grid.
        self.grid = partition_to_grid(gs_image300x100, num_rows=10, num_cols=10, hor_pixels=30, ver_pixels=10)


    def get_hog_vector(self):
        """
        Represent as a 900-dim torch vector.
        The sequential vector representation is as follows:

        Row-major order of cells (10x10 cells), with each cell having 9 values, one each
        for binned degrees: 0, 40, 80, 120, 160, 200, 240, 280, 320.
        0 is the same as 360.
        The magnitude of the gradient is distributed according to its direction,
        with the adjacent bins getting the magnitude weighted according to how
        close the direction is to the bin.

        vec: [<cell_1 <0, 40, 80, 120 ... 320>>, <cell_2 ...>...]
        """
        self.__generate_gradients()

        self.__generate_gradient_bins()

        hog_descriptor = []

        for i in range(len(self.gradient_bins)):
            for j in range(len(self.gradient_bins[0])):

                bins = self.gradient_bins[i][j]

                hog_descriptor.extend(bins)

        return tensor(hog_descriptor)


    def __generate_gradient_bins(self):
        """
        We accomplish magnitude-weighted binning via the following strategy:

        pixel(x, y) has magnitude 72, direction 50. Since the adjacent bins
        are at 40 and 80 degrees, 72 * |80 - 50|/40 = 54 is added to bin 40,
        and 72 * |40 - 50|/40 = 18 is added to bin 80.
        (direction 50 is closer to 40 than to 80, so gets a larger share)

        Doing this in essense increases the smoothness of the histogram.
        """

        self.gradient_bins = []

        for i in range(len(self.g_mags)):

            self.gradient_bins.append([])

            for j in range(len(self.g_mags[0])):

                g_mag = self.g_mags[i][j]
                g_dir = self.g_dirs[i][j]

                bins = [0] * 9
                # index 0 => 0 deg (and also 360 deg), index 8 => 320

                for row in range(len(g_mag)):
                    for col in range(len(g_mag[0])):

                        low_bin_idx = int(floor(g_dir[row][col] / 40))
                        high_bin_idx = int(ceil(g_dir[row][col] / 40))

                        low_bin_deg = low_bin_idx * 40
                        high_bin_deg = high_bin_idx * 40

                        # Handle case when degree is > 320.
                        high_bin_idx = 0 if high_bin_idx == 9 else high_bin_idx

                        bins[low_bin_idx] += g_mag[row][col] * (abs(high_bin_deg - g_dir[row][col]) / 40)
                        bins[high_bin_idx] += g_mag[row][col] * (abs(low_bin_deg - g_dir[row][col]) / 40)

                self.gradient_bins[-1].append(bins)


    def __generate_gradients(self):
        """
        Generate gradient magnitudes and directions.
        """

        g_mask = array([-1, 0, 1])

        self.g_mags = []
        self.g_dirs = []


        for i in range(len(self.grid)):

            self.g_mags.append([])
            self.g_dirs.append([])

            for j in range(len(self.grid[0])):

                cell = self.grid[i][j][0]

                g_x = []
                g_y = []

                # Prior to calculating dI/dx and dI/dy, we 0 pad the patch so as to maintain
                # resolution. The assumption here is that the contributions to the bins by each
                # of these edge pixels is negligible relative to the rest of the pixels.
                # This is accomplished by using np.correlate with mode 'same'.
                #
                # NOTE: np.correlate is not np.convolve. np.correlate merely moves the kernel
                # over the image (which gives us the gradient).

                for row in cell:
                    g_x.append(correlate(row, g_mask, mode='same'))

                for column in transpose(cell): # Transpose to compute vertical gradient horizontally.
                    g_y.append(correlate(column, g_mask, mode='same'))

                g_x = array(g_x)
                g_y = transpose(array(g_y)) # Transpose back so that it is vertical.

                # Compute magnitudes: sqrt(g_x^2 + g_y^2)
                g_x2 = power(g_x, 2)
                g_y2 = power(g_y, 2)
                g_mag = sqrt(g_x2 + g_y2)

                # Compute directions: atan(g_y/g_x)
                # We use np.arctan2 to correctly select quadrant.
                g_dir = arctan2(g_y, g_x)
                # We do modulo 360 to so as to make -45 deg => 315 deg
                g_dir = rad2deg(g_dir) % 360

                self.g_mags[-1].append(g_mag)
                self.g_dirs[-1].append(g_dir)