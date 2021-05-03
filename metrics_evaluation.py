import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from bitstring import BitStream


class Distortion_Evaluation():

    def __init__(self):
        ##### Attribute for saving computed results
        self.results = []
        return


    def display_comparison(self, original, reconstructed, bitstring, quantizer_id):
        ##### Compute PSNR between image versions
        psnr = self.psnr(original.astype(np.int32), reconstructed.astype(np.int32))
        ##### Compute bpp
        bpp = self.bpp(original, bitstring)

        ##### Plot comparison
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f'{quantizer_id}: {psnr:.2f}dB', fontsize=16)
        cmap = "gray" if len(original.shape) == 2 else 'viridis'
        axs[0].imshow(original, cmap=cmap)
        axs[0].set_title('Original Image.')
        axs[1].imshow(reconstructed, cmap=cmap)
        axs[1].set_title(f'Quantized with {bpp:.2f}bpp.')

        plt.show()
        return

    
    def get_img_pairs_and_bitstring(self, original, reconstructed, bitstring, N, M):
        ##### Compute PSNR between image versions
        mse = self.mse(original.astype(np.int32), reconstructed.astype(np.int32))
        ##### Compute bpp
        bpp = self.bpp(original, bitstring)
        ##### Save results
        self.results.append([mse, bpp, N, M])


    def plot_rd_curve(self, quantizer_id, img_name):
        ##### Choose lower convex hull.
        # Sort results by bpp.
        self.results = np.array(self.results)
        sorted_results = self.results[np.argsort(self.results[:, 1])]
        # Get values in the hull.
        convex_hull_indexes = [True] + list(map(lambda idx: np.all(sorted_results[idx, 0] < sorted_results[:idx, 0]), range(1, len(sorted_results))))
        ##### Plot scatter plot cloud.
        colors_dict = {**mcolors.BASE_COLORS, **mcolors.TABLEAU_COLORS}
        colors_dict.pop('w')
        used_colors = [random.choice(list(colors_dict.items()))[0]]
        scatters = []
        for mse, bpp in sorted_results[:, :2]:
            # Add point to scatter plot
            scatters.append(plt.scatter(bpp, mse, color=used_colors[-1]))
            # Chose next color
            chosen_color = random.choice(list(mcolors.CSS4_COLORS.items()))[0]
            while chosen_color in used_colors:
                chosen_color = random.choice(list(mcolors.CSS4_COLORS.items()))[0]
            used_colors.append(chosen_color)
        ##### Define title and axes labels
        plt.title(f"{quantizer_id} Scatter Plot for {img_name}.")
        plt.xlabel("BPP")
        plt.ylabel("MSE")
        ##### Define legend
        legends = list(map(lambda parameters: f"N={int(parameters[0])}; M={int(parameters[1])}", sorted_results[:, 2:]))
        plt.legend(scatters, legends, ncol=4, fontsize=8)
        ##### Plot lower convex hull
        # plt.plot(sorted_results[convex_hull_indexes, 1], sorted_results[convex_hull_indexes, 0], color='r')
        ##### Plot Image
        plt.show()
        return


    def psnr(self, original, reconstructed):
        mse = self.mse(original, reconstructed)
        psnr = 10 * math.log10(255**2/mse)
        return psnr


    def mse(self, original, reconstructed):
        return np.mean(np.square(np.subtract(original, reconstructed)))


    def bpp(self, original, bitstring):
        ##### Compute pixel amount
        pixel_amount = np.prod(original.shape[:2])
        ##### Compute achieved rate
        bpp = len(bitstring.bin.__str__()) / pixel_amount
        return bpp