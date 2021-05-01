import os
import math
import itertools
import numpy as np

from abc import ABC, abstractmethod
from PIL import Image
from bitstring import BitStream
from sklearn.cluster import KMeans



class LossyCompression(ABC):

    def __init__(self, image_file_path, binary_file_path, N=None, M=None):
        ##### Save parameters
        self.N = N
        self.M = M
        self.binary_file_path = binary_file_path
        ##### Read input image and separate into coding blocks.
        self._separate_blocks(image_file_path)
        return


    @abstractmethod
    def encode_image(self):
        pass

    
    # TODO: Add method to decoding class
    # @abstractmethod
    # def decode_binary(self):
    #     pass


    ########## Private Methods ##########

    def _separate_blocks(self, image_file_path):
        ##### Read Image
        pil_image = Image.open(image_file_path)
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


    def _save_binary_file(self):
        with open(self.binary_file_path, "wb") as bin_file:
            bin_file.write(self.bitstring.bin.encode())
            bin_file.close()


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



class CIQA_Encoder(LossyCompression):

    def encode_image(self):
        self._compute_coefficients()
        self._write_bitstring()
        self._save_binary_file()
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



class CIVQ_Encoder(LossyCompression):

    def encode_image(self):
        ##### Reshape patches: flatten the patches (pixels inside patches are already flattened).
        self.patches = np.reshape(self.patches, (int(np.prod(self.patches.shape[:2])), self.patches.shape[-1]))

        ##### Quantize and save coefficients
        self._generate_codebook_and_quantize()
        self._write_bitstring()
        self._save_binary_file()

        return


    # TODO: Add method to decoding class
    # def decode_binary(self):
    #     self._read_bitstring_header()
    #     self._decode_patches()
    #     return


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



class CIMap_Encoder(CIVQ_Encoder):

    def __init__(self, image_file_path, binary_file_path, M):
        ##### Call parent constructor without 'N' parameter.
        super().__init__(image_file_path, binary_file_path, None, M)
        return


    def _separate_blocks(self, image_file_path):
        ##### Read Image and use colors as blocks
        pil_image = Image.open(image_file_path)
        self.patches = np.asarray(pil_image)
        return


    def _instantiate_bitstring(self):
        ##### Write information about codebook length.
        # NOTE: The available codebook sizes are 16, 32, 64, 128 and 256.
        codebook_size_flag = int(math.log2(self.M) - 4)
        self.bitstring = BitStream(f'uint:3={codebook_size_flag}')
        return



class Dithering_Quantizer():

    def __init__(self, image_file_path, levels=8):
        ##### Save quantization precision
        self.levels = levels
        ##### Read input image
        self.image = np.asarray(Image.open(image_file_path)).astype(np.int32)
        return

    
    def evaluate_filtering(self):
        self._quantize_image()
        self._floyd_steinberg_filtering()
        self._display_PSNR()
        self._save_both_versions()


    def _quantize_image(self):
        ##### Verify if image has one or three channels
        if len(self.image.shape) == 2:
            greenscale = True
            self.image = np.expand_dims(self.image, axis=2)
        else: 
            greenscale = False

        ##### Create quantization levels
        if greenscale:
            # Greenscale images have self.levels options of equally spaced grey scales.
            self.quantization_levels = np.expand_dims(np.linspace(0, 255, num=self.levels, dtype=np.uint8), axis=1)
        else:
            # NOTE: For colorful images, only 8 and 16 levels are available.
            pixel_values = np.linspace(0, 255, num=self.levels//4, dtype=np.uint8)
            self.quantization_levels = self._get_possible_combinations(3, pixel_values[0], pixel_values[-1])
            if self.levels == 16:
                self.quantization_levels = np.vstack([self.quantization_levels, self._get_possible_combinations(3, *pixel_values[1:3])])

        ##### Flatten input image.
        #flattened_image = np.reshape(self.image, (np.prod(self.image.shape[:2]), self.image.shape[-1]))
        ##### Expand flattened image dimensions
        exp_image = np.repeat(np.expand_dims(self.image, axis=2), self.levels, axis=2)
        ##### Compute Euclidean distance
        euclidean_dist = np.linalg.norm(np.subtract(exp_image, self.quantization_levels), axis=3)
        ##### Get quantization indexes
        indexes = np.argsort(euclidean_dist)[:, :, 0]
        ##### Get quantized image
        self.quantized_image = np.squeeze(self.quantization_levels[indexes])

        return


    def _floyd_steinberg_filtering(self):
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

        return

    
    def _get_possible_combinations(self, lists_lengths, v1, v2):
        combinations = []

        for v2_amounts in range(lists_lengths + 1):
            for combination in itertools.combinations(range(lists_lengths), v2_amounts):
                comb_list = np.full(lists_lengths, v1)
                comb_list[list(combination)] = v2
                combinations.append(comb_list)

        return np.array(combinations)


########## Main Code

file_name = "Image_Database/kodim03.png"
output_name = "kodim03.bin"
# encoder = CIMap_Encoder(file_name, output_name, 16)
# encoder.encode_image()
model  = Dithering_Quantizer(file_name, 8)
model.evaluate_filtering()