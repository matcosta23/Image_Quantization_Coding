import math
import numpy as np
from PIL import Image
from bitstring import BitStream

import vector_quantization


class CIMap(vector_quantization.CIVQ):

    def __init__(self, image_file_path=None, binary_file_path=None, rec_image_file_path=None, M=None):
        ##### Call parent constructor without 'N' parameter.
        super().__init__(image_file_path, binary_file_path, rec_image_file_path, None, M)
        return


    def _separate_blocks(self):
        ##### Read Image and use colors as blocks. Therefore, pixels are the patches here
        pil_image = Image.open(self.image_file_path)
        self.patches = np.asarray(pil_image)
        ##### Get dimensions difference
        self.patch_dims_diff = -np.subtract(self.patches.shape[0], self.patches.shape[1])
        ##### Reshape patches: flatten the patches (pixels inside patches are already flattened).
        self.patches = np.reshape(self.patches, (int(np.prod(self.patches.shape[:2])), self.patches.shape[-1]))
        return


    def _instantiate_bitstring(self):
        ##### Write information about codebook length.
        # NOTE: The available codebook sizes are 16, 32, 64, 128 and 256.
        codebook_size_flag = int(math.log2(self.M) - 4)
        self.bitstring = BitStream(f'int:10={self.patch_dims_diff}, uint:3={codebook_size_flag}')
        return


    def _read_bitstring_header(self):
        ##### Read info about image dimensions and codebook lenght.
        self.patch_dims_diff = self.bitstring.read('int:10')
        codebook_size_flag = self.bitstring.read('uint:3')
        self.M = 2**(codebook_size_flag + 4)
        self.N = 1
        self.block_length = 3
        self.bits_in_header = 10 + 3
        return


    def _build_image_from_patches(self):
        self.quantized_image = self.flattened_patches.astype(np.uint8)
        return


file_name = "Image_Database/kodim03.png"
output_name = "kodim03.bin"
color_map = CIMap(file_name, output_name, M=16)
color_map.encode_image()
color_map.decode_binary()