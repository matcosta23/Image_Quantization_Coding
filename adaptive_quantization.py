import math
import numpy as np
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
        # NOTE: The maximum patch dimension is 32, while the maximum quantization level is 16.
        self.bitstring = BitStream(f'uint:5={self.N - 1}, uint:4={self.M - 1}')
        
        ##### Write patch coefficients into the bitstring.
        bits_for_quantized_values = int(np.ceil(math.log2(self.M)))
        for patch_information in self.info_to_be_sent:
            ##### Write minimum and maximum values.
            self.bitstring.append(f'uint:8={patch_information[0]}, uint:8={patch_information[1]}')
            ##### Write coefficients.
            list(map(lambda quantized: self.bitstring.append(f'uint:{bits_for_quantized_values}={quantized}'), patch_information[2:]))

        return


file_name = "Image_Database/lena.bmp"
output_name = "lena.bin"
encoder = CIQA(file_name, output_name, N=8, M=8)
encoder.encode_image()