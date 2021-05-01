import math
import numpy as np
from bitstring import BitStream
from sklearn.cluster import KMeans

import common_classes


class CIVQ(common_classes.LossyCompression):

    def encode_image(self):
        ##### Read input image and separate into coding blocks.
        self._separate_blocks()
        ##### Quantize and write coefficients
        self._generate_codebook_and_quantize()
        self._write_bitstring()
        return


    # TODO: Add method to decoding class
    # def decode_binary(self):
    #     self._read_bitstring_header()
    #     self._decode_patches()
    #     return


    def _separate_blocks(self):
        super()._separate_blocks()
        ##### Reshape patches: flatten the patches (pixels inside patches are already flattened).
        self.patches = np.reshape(self.patches, (int(np.prod(self.patches.shape[:2])), self.patches.shape[-1]))


    def _generate_codebook_and_quantize(self):
        ##### Instantiate and run clustering algorithm.
        clustering = KMeans(n_clusters=self.M).fit(self.patches)
        self.codebook = clustering.cluster_centers_
        self.indexes = clustering.predict(self.patches)
        return


    def _write_bitstring(self):
        ##### Instantiate bitstring and write hyper-parameters.
        self._instantiate_bitstring()
        ##### Write codebook on bitstring
        flatten_codevectors = np.round(self.codebook.flatten()).astype(np.uint8)
        for coefficient in flatten_codevectors:
            self.bitstring.append(f'uint:8={coefficient}')
        ##### Write indexes in the bit string.
        index_bits_amount = int(math.log2(self.M))
        for index in self.indexes:
            self.bitstring.append(f'uint:{index_bits_amount}={index}')
        return


    def _instantiate_bitstring(self):
        ##### Write information about patch size and quantization levels.
        # NOTE: The available patch dimensions are 2x2, 4x4, 8x8 and 16x16
        # NOTE: The available codebook sizes are 32, 64, 128 and 256
        block_size_flag = int(math.log2(self.N) - 1)
        codebook_size_flag = int(math.log2(self.M) - 5)
        self.bitstring = BitStream(f'uint:2={block_size_flag}, uint:2={codebook_size_flag}')
        return


    # TODO: Add method to decoding class
    # def _read_bitstring_header(self):
    #     ##### Read Information about patch dimensions and codebook lenght.
    #     block_size = self.bitstring.read('uint:2')
    #     codebook_size = self.bitstring.read('uint:2')
    #     self.N = 2**(block_size + 1)
    #     self.M = 2**(codebook_size + 5)

    #     ##### Read codebook
    #     coefficients = []
    #     for i in range(self.M * self.N**2):
    #         coefficients.append(self.bitstring('uint:8'))
        
    #     self.codebook = np.reshape(np.array(coefficients), (self.M, self.N**2))

    #     return



if __name__ == "__main__":
    file_name = "Image_Database/lena.bmp"
    output_name = "lena.bin"
    encoder = CIVQ(file_name, output_name, N=8, M=32)
    encoder.encode_image()