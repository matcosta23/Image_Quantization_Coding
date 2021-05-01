import itertools
import numpy as np
from PIL import Image


class Dithering_Quantizer():

    def __init__(self, image_file_path, levels=8, rec_image_folders=None):
        ##### Save input arguments
        self.image_file_path = image_file_path
        self.rec_image_folders = rec_image_folders
        self.levels = levels
        ##### Read input image
        self.image = np.asarray(Image.open(image_file_path)).astype(np.int32)
        return

    
    def evaluate_filtering(self):
        self.uniform_quantization()
        self.floyd_steinberg_filtering()
        # self.display_comparison()
        # self._save_both_versions()
        return


    def uniform_quantization(self):
        ##### Verify if image has one or three channels
        if len(self.image.shape) == 2:
            grayscale = True
            self.image = np.expand_dims(self.image, axis=2)
        else: 
            grayscale = False
        ##### Create quantization levels
        if grayscale:
            # Grayscale images have 'self.levels' options of equally spaced gray scales.
            self.quantization_levels = np.expand_dims(np.linspace(0, 255, num=self.levels, dtype=np.uint8), axis=1)
        else:
            # NOTE: For colorful images, only 8 and 16 levels are available.
            pixel_values = np.linspace(0, 255, num=self.levels//4, dtype=np.uint8)
            self.quantization_levels = self._get_possible_combinations(3, pixel_values[0], pixel_values[-1])
            if self.levels == 16:
                self.quantization_levels = np.vstack([self.quantization_levels, self._get_possible_combinations(3, *pixel_values[1:3])])
        ##### Expand flattened image dimensions
        exp_image = np.repeat(np.expand_dims(self.image, axis=2), self.levels, axis=2)
        ##### Compute Euclidean distance
        euclidean_dist = np.linalg.norm(np.subtract(exp_image, self.quantization_levels), axis=3)
        ##### Get quantization indexes
        indexes = np.argsort(euclidean_dist)[:, :, 0]
        ##### Get quantized image
        self.quantized_image = np.squeeze(self.quantization_levels[indexes])
        return


    def floyd_steinberg_filtering(self):
        ##### Get image dimensions
        height, width = self.image.shape[:2]
        ##### Create filtered image
        self.filtered = self.image.astype(np.int32)
        ##### Iterate over all every pixel and spread noise.
        for v in range(height):
            for h in range(width):
                ##### Quantize pixel
                previous_pixel = self.filtered[v, h].copy()
                self.filtered[v, h] = self.quantization_levels[np.argsort(np.linalg.norm(previous_pixel - self.quantization_levels, axis=1))[0]]
                ##### Compute error
                quantization_error = previous_pixel - self.filtered[v, h]
                ##### Spread error
                try: self.filtered[v, h + 1]     += np.around(quantization_error * 7/16).astype(np.int32)
                except IndexError: pass
                try: self.filtered[v + 1, h - 1] += np.around(quantization_error * 3/16 * (h - 1 >= 0)).astype(np.int32)
                except IndexError: pass
                try: self.filtered[v + 1, h]     += np.around(quantization_error * 5/16).astype(np.int32)
                except IndexError: pass
                try: self.filtered[v + 1, h + 1] += np.around(quantization_error * 1/16).astype(np.int32)
                except IndexError: pass
        ##### Return image to original shape and cast bytes to uint8.
        self.filtered = np.squeeze(self.filtered).astype(np.uint8)
        return

    
    def _get_possible_combinations(self, lists_lengths, v1, v2):
        combinations = []

        for v2_amounts in range(lists_lengths + 1):
            for combination in itertools.combinations(range(lists_lengths), v2_amounts):
                comb_list = np.full(lists_lengths, v1)
                comb_list[list(combination)] = v2
                combinations.append(comb_list)

        return np.array(combinations)



if __name__ == "__main__":
    file_name = "Image_Database/kodim03.png"
    output_name = "kodim03.bin"
    model = Dithering_Quantizer(file_name, levels=8)
    model.evaluate_filtering()