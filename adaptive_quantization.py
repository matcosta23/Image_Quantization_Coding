import math
import numpy as np
from PIL import Image 
from bitstring import BitStream

import common_classes


class CIQA(common_classes.LossyCompression):

    def encode_image(self):
        ##### Read input image and separate into coding blocks.
        self._separate_blocks()
        ##### Quantize image.
        self._compute_coefficients()
        ##### Instantiate and write bitstring
        self._write_bitstring()
        return


    def decode_image(self):
        ##### Obtain bitstring
        self._get_bitstring()
        ##### Read header and decode bitstring.
        self._read_coefficientes()


    def _compute_coefficients(self):
        ##### Get minimum and maximum values by patch
        min_values = np.expand_dims(np.amin(self.patches, axis=2), axis=2)
        max_values = np.expand_dims(np.amax(self.patches, axis=2), axis=2)

        ##### Compute deltas
        deltas = (max_values - min_values) / self.M

        ##### Quantize patches
        normalized_patches = self.patches - min_values
        quantized_patches = np.clip(np.floor(normalized_patches / deltas), 0, 7)

        ##### Stack information to be sent to the decoder and reshape them.
        info_to_be_sent = np.concatenate((min_values, max_values, quantized_patches), axis=2)
        info_to_be_sent = np.reshape(info_to_be_sent, (int(np.prod(info_to_be_sent.shape[:2])), info_to_be_sent.shape[-1]))
        self.info_to_be_sent = info_to_be_sent.astype(np.uint8)

        return


    def _write_bitstring(self):
        ##### Write information about patch size and quantization levels.
        # The first is the difference between width and height: 
        # NOTE: The maximum patch dimension is 32, while the maximum quantization level is 16.
        self.bitstring = BitStream(f'int:7={self.patches.shape[1] - self.patches.shape[0]}, '
                                   f'uint:5={self.N - 1}, uint:4={self.M - 1}')
        
        ##### Write patch coefficients into the bitstring.
        bits_for_quantized_values = int(np.ceil(math.log2(self.M)))
        for patch_information in self.info_to_be_sent:
            ##### Write minimum and maximum values.
            self.bitstring.append(f'uint:8={patch_information[0]}, uint:8={patch_information[1]}')
            ##### Write coefficients.
            list(map(lambda quantized: self.bitstring.append(f'uint:{bits_for_quantized_values}={quantized}'), patch_information[2:]))

        return


    def _read_coefficientes(self):
        ##### Read header
        patches_diff, self.N, self.M = self.bitstring.readlist(f'int:7, uint:5, uint:4')
        self.N += 1
        self.M += 1
        ##### Obtain image dimensions
        bits_for_quantized_values = int(np.ceil(math.log2(self.M)))
        bits_per_patch = 2 * 8 + np.ceil(math.log2(self.M)) * self.N**2
        patches_amount = len(self.bitstring.bin.__str__()[16:]) / bits_per_patch
        second_degree_coeff = [1, np.abs(patches_diff), -patches_amount]
        vertical_patches = int(np.around(np.roots(second_degree_coeff).max(), 0))
        horizontal_patches = vertical_patches + patches_diff
        image_shape = np.array([vertical_patches, horizontal_patches]) * self.N
        ##### Decode patches
        self.quantized_image = np.zeros(image_shape)
        for v in range(vertical_patches):
            for h in range(horizontal_patches):
                ##### Get index mapping within a patch
                min_value, max_value = self.bitstring.readlist(f'uint:8, uint:8')
                index_mapping = np.linspace(min_value, max_value, num=self.M)
                ##### Read index
                indexes = np.reshape(list(\
                    map(lambda pixel_idx: self.bitstring.read(f'uint:{bits_for_quantized_values}'), range(self.N**2))), \
                        (self.N, self.N))
                ##### Decode and store patch.
                reconstructed_patch = index_mapping[indexes]
                self.quantized_image[v * self.N: (v + 1) * self.N, h * self.N: (h + 1) * self.N] = reconstructed_patch
        ##### Cast patch to uint8.
        self.quantized_image = self.quantized_image.astype(np.uint8)
        return



file_name = "Image_Database/lena.bmp"
output_name = "lena.bin"
adaptive = CIQA(file_name, output_name, N=8, M=8)
adaptive.encode_image()
adaptive.decode_image()