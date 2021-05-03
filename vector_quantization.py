import os
import math
import numpy as np
from PIL import Image
from bitstring import BitStream
from sklearn.cluster import KMeans

import common_classes
#from metrics_evaluation import Distortion_Evaluation


class CIVQ(common_classes.LossyCompression):

    def encode_image(self):
        ##### Read input image and separate into coding blocks.
        self._separate_blocks()
        ##### Quantize and write coefficients
        self._generate_codebook_and_quantize()
        self._write_bitstring()
        return


    def decode_binary(self):
        ##### Obtain bitstring
        self._get_bitstring()
        ##### Read codevectors and codebook from bitstring.
        self._read_codebook_and_indexes()
        ##### Map indexes and recovery image.
        self._reconstruct_image()
        return


    def _separate_blocks(self):
        super()._separate_blocks()
        ##### Get dimensions difference
        self.patch_dims_diff = -np.subtract(self.patches.shape[0], self.patches.shape[1])
        ##### Reshape patches: flatten the patches (pixels inside patches are already flattened).
        self.patches = np.reshape(self.patches, (int(np.prod(self.patches.shape[:2])), self.patches.shape[-1]))
        return


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


    def _read_codebook_and_indexes(self):
        ##### Read Header
        self._read_bitstring_header()
        ##### Read codebook
        coefficients = []
        for i in range(self.M * self.block_length):
            coefficients.append(self.bitstring.read('uint:8'))
        self.codebook = np.reshape(np.array(coefficients), (self.M, self.block_length))
        ##### Read indexes
        index_bits_amount = int(math.log2(self.M))
        bits_before_indexes = self.bits_in_header + self.M * self.block_length * 8
        indexes_amount = int(len(self.bitstring.bin.__str__()[bits_before_indexes:]) / index_bits_amount)
        self.indexes = np.array(list(map(lambda idx: self.bitstring.read(f'uint:{index_bits_amount}'), range(indexes_amount))))
        return


    def _instantiate_bitstring(self):
        ##### Write information about patch size and quantization levels.
        # Dimensions difference will be encoded with 7 bits.
        # NOTE: The available patch dimensions are 2x2, 4x4, 8x8 and 16x16
        # NOTE: The available codebook sizes are 32, 64, 128 and 256
        block_size_flag = int(math.log2(self.N) - 1)
        codebook_size_flag = int(math.log2(self.M) - 5)
        self.bitstring = BitStream(f'int:7={self.patch_dims_diff}, '
                                   f'uint:2={block_size_flag}, uint:2={codebook_size_flag}')
        return


    def _read_bitstring_header(self):
        ##### Read info about image dimensions and codebook lenght.
        self.patch_dims_diff = self.bitstring.read('int:7')
        block_size = self.bitstring.read('uint:2')
        codebook_size = self.bitstring.read('uint:2')
        self.N = 2**(block_size + 1)
        self.M = 2**(codebook_size + 5)
        self.block_length = self.N**2
        self.bits_in_header = 7 + 2 + 2
        return


    def _reconstruct_image(self):
        ##### Obtain dimensions
        self.patch_dims = self._compute_dimensions(self.patch_dims_diff, len(self.indexes))
        self.img_dims = self.patch_dims * self.N
        ##### Arange index in the associated patch position.
        self.indexes = np.reshape(self.indexes, self.patch_dims)
        ##### Map index from codebook
        self.flattened_patches = self.codebook[self.indexes]
        ##### Reshape and cast
        self._build_image_from_patches()
        return


    def _build_image_from_patches(self):
        patches = np.reshape(self.flattened_patches, (*self.patch_dims, self.N, self.N))
        self.quantized_image = np.zeros(self.img_dims).astype(np.uint8)
        for v in range(self.patch_dims[0]):
            for h in range(self.patch_dims[1]):
                self.quantized_image[v * self.N: (v + 1) * self.N, h * self.N: (h + 1) * self.N] = patches[v, h]
        return



if __name__ == "__main__":
    ##### Read arguments from command line
    args = common_classes.read_arguments()
    ##### Create directories
    args.binaries_folder, args.quantized_folder = common_classes.create_folders(args.binaries_folder, args.quantized_folder, "Vector", args.save_results)
    ##### Define possible parameters
    N_values = np.array([2, 4, 8, 16])
    M_values = np.array([32, 64, 128, 256])
    ##### Verify if multiple or single points should be executed
    if args.global_evaluation is False:
        # Verify if hyper-parameters are ok
        if args.N not in N_values:
            raise ValueError(f"The block dimension 'N' must be one of these values: {list(N_values)}")
        if args.M not in M_values:
            raise ValueError(f"The codebook length 'M' must be one of these values: {list(M_values)}")
        # Evaluate point
        common_classes.evaluate_one_point(args, CIVQ, "Vector Quantizer")
    else:
        ##### Global evaluation.
        common_classes.global_evaluation(args, N_values, M_values, CIVQ, "Vector Quantizer")