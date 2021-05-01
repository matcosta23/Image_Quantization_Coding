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
        ##### Read Image and use colors as blocks
        pil_image = Image.open(self.image_file_path)
        self.patches = np.asarray(pil_image)
        ##### Reshape patches: flatten the patches (pixels inside patches are already flattened).
        self.patches = np.reshape(self.patches, (int(np.prod(self.patches.shape[:2])), self.patches.shape[-1]))
        return


    def _instantiate_bitstring(self):
        ##### Write information about codebook length.
        # NOTE: The available codebook sizes are 16, 32, 64, 128 and 256.
        codebook_size_flag = int(math.log2(self.M) - 4)
        self.bitstring = BitStream(f'uint:3={codebook_size_flag}')
        return


file_name = "Image_Database/kodim03.png"
output_name = "kodim03.bin"
encoder = CIMap(file_name, output_name, M=16)
encoder.encode_image()