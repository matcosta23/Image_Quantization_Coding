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


    def display_comparison(self, original, reconstructed_versions, bitstring, quantizer_id, rec_titles=None):
        ##### Create titles
        if rec_titles is None:
            rec_titles = [""] * len(reconstructed_versions)
        ##### Compute PSNR between image versions
        psnr = list(map(lambda rec_version: self.psnr(original.astype(np.int32), rec_version.astype(np.int32)), reconstructed_versions))
        ##### Add PSNR to titles
        titles = list(map(lambda old_title, psnr_value: old_title + f"PSNR: {psnr_value:.2f}dB", rec_titles, psnr))
        ##### Compute bpp
        if bitstring is not None:
            bpp = list(map(lambda bs: self.bpp(original, bs), bitstring))
            titles = list(map(lambda old_title, bpp_value: old_title + f"; BPP: {bpp_value:.2f}", titles, bpp))
        ##### Update axes title font size.
        parameters = {'axes.titlesize': 10}
        plt.rcParams.update(parameters)
        ##### Plot comparison
        fig, axs = plt.subplots(1, len(reconstructed_versions) + 1)
        fig.suptitle(f'{quantizer_id}', fontsize=16)
        cmap = "gray" if len(original.shape) == 2 else 'viridis'
        axs[0].imshow(original, cmap=cmap)
        axs[0].set_title('Original Image.')
        axs[0].set_yticklabels([])
        axs[0].set_xticklabels([])
        for idx in range(len(reconstructed_versions)):
            axs[idx + 1].imshow(reconstructed_versions[idx] , cmap=cmap)
            axs[idx + 1].set_title(titles[idx])
            axs[idx + 1].set_yticklabels([])
            axs[idx + 1].set_xticklabels([])
        plt.tight_layout()
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
        plt.legend(scatters, legends, ncol=3, fontsize=7)
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