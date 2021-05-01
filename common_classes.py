import os
import numpy as np

from abc import ABC, abstractmethod
from PIL import Image
from bitstring import BitStream



class LossyCompression(ABC):

    def __init__(self, image_file_path=None, binary_file_path=None, 
                       rec_image_file_path=None, N=None, M=None):
        ##### Save paths.
        self.image_file_path = image_file_path
        self.binary_file_path = binary_file_path
        self.rec_image_file_path = rec_image_file_path
        ##### Save quantization parameters
        self.N = N
        self.M = M
        return


    @abstractmethod
    def encode_image(self):
        pass

    
    # TODO: Add method to decoding class
    # @abstractmethod
    # def decode_binary(self):
    #     pass


    def save_binary_file(self):
        with open(self.binary_file_path, "wb") as bin_file:
            bin_file.write(self.bitstring.bin.encode())
            bin_file.close()


    ########## Private Methods ##########

    def _separate_blocks(self):
        ##### Read Image
        pil_image = Image.open(self.image_file_path)
        image = np.asarray(pil_image)

        ##### Verify shapes and pad image
        bottom_padding = 0 if image.shape[0] % self.N == 0 else self.N - image.shape[0] % self.N
        right_padding = 0 if image.shape[1] % self.N == 0 else self.N - image.shape[1] % self.N
        padded_image = np.pad(image, ((0, bottom_padding), (0, right_padding)), 'edge')
        
        ##### Split array into patches
        rows_of_patches = np.vsplit(padded_image, padded_image.shape[0] // self.N)
        patches = np.array(list(map(lambda row: np.hsplit(row, padded_image.shape[1] // self.N), rows_of_patches)))
        # Flatten patches
        self.patches = np.reshape(patches, [patches.shape[0], patches.shape[1], self.N ** 2])

        return


    # TODO: Add method to decoding class
    # def _read_binary_file(self, input_file_path):
    #     ##### Read binary file and write bitstring.
    #     with open(input_file_path) as bin_file:
    #         bitstring = bin_file.read()
    #         self.bitstring = BitStream(f'0b{bitstring}')


    @abstractmethod
    def _write_bitstring(self):
        pass


    # TODO: Add method to decoding class
    # @abstractmethod
    # def _read_bitstring(self):
    #     pass
    
    
    ########## Auxiliary Methods ##########




########## Main Code
if __name__ == "__main__":
    file_name = "Image_Database/kodim03.png"
    output_name = "kodim03.bin"
    # # encoder = CIMap_Encoder(file_name, output_name, 16)
    # # encoder.encode_image()
    # model  = Dithering_Quantizer(file_name, 8)
    # model.evaluate_filtering()