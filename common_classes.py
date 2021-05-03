import os
import sys
import argparse
import numpy as np

from abc import ABC, abstractmethod
from PIL import Image
from bitstring import BitStream

from metrics_evaluation import Distortion_Evaluation


""" Parent Class """

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


    @abstractmethod
    def decode_binary(self):
        pass


    def save_binary_file(self):
        with open(self.binary_file_path, "wb") as bin_file:
            bin_file.write(self.bitstring.bin.encode())
            bin_file.close()
        return


    def save_quantized_image(self):
        pil_img = Image.fromarray(self.quantized_image)
        pil_img.save(self.rec_image_file_path)
        return


    ########## Private Methods ##########

    def _separate_blocks(self):
        ##### Read Image
        pil_image = Image.open(self.image_file_path)
        self.image = np.asarray(pil_image)

        ##### Verify shapes and pad image
        bottom_padding = 0 if self.image.shape[0] % self.N == 0 else self.N - self.image.shape[0] % self.N
        right_padding = 0 if self.image.shape[1] % self.N == 0 else self.N - self.image.shape[1] % self.N
        padded_image = np.pad(self.image, ((0, bottom_padding), (0, right_padding)), 'edge')
        
        ##### Split array into patches
        rows_of_patches = np.vsplit(padded_image, padded_image.shape[0] // self.N)
        patches = np.array(list(map(lambda row: np.hsplit(row, padded_image.shape[1] // self.N), rows_of_patches)))
        # Flatten patches
        self.patches = np.reshape(patches, [patches.shape[0], patches.shape[1], self.N ** 2])

        return


    @abstractmethod
    def _write_bitstring(self):
        pass


    def _get_bitstring(self):
        ##### Verify if bitstring is already instantiated.
        try:
            bitstring_exists = isinstance(self.bitstring, BitStream)
        except AttributeError:
            bitstring_exists = False 
        ##### Read binary file and write bitstring.
        if bitstring_exists is False:
            with open(self.binary_file_path) as bin_file:
                bitstring = bin_file.read()
                self.bitstring = BitStream(f'0b{bitstring}')


    def _compute_dimensions(self, dims_diff, n_elements):
        second_degree_coeff = [1, np.abs(dims_diff), -n_elements]
        vertical = int(np.around(np.roots(second_degree_coeff).max(), 0))
        horizontal = vertical + dims_diff
        return np.array([vertical, horizontal])

        
""" Auxiliary Functions """

def read_arguments():
    ##### Define parser
    parser = argparse.ArgumentParser(description="Receives arguments for quantization.")
    ##### Define arguments.
    parser.add_argument('--image_to_quantize', required=True, help='Path to original image.')
    parser.add_argument('-g', '--global_evaluation', action='store_true', help='If set, all hyper-parameter combinations are compared.')
    parser.add_argument('-s', '--save_results', action='store_true', help='If set, results are saved on output paths.')
    parser.add_argument('--N', required=False, type=int, help="Value for N hyper-parameter.")
    parser.add_argument('--M', required=True, type=int, help="Value for M hyper-parameter.")
    parser.add_argument('--binaries_folder', required=False, help='Folder to save binaries. '
                                                                  "Only used if '-s' flag is set.")
    parser.add_argument('--quantized_folder', required=False, help='Folder to save quantized images. '
                                                                   "Only used if '-s' flag is set.")
    ##### Return namespace.
    return parser.parse_args(sys.argv[1:])



def create_folders(binaries_folder, quantized_folder, quantizer_id, save_results):
    ##### Verify if user has provided destiny folders.
    if binaries_folder is None: 
        binaries_folder = "Binaries_" + quantizer_id
    if quantized_folder is None:
        quantized_folder = "Quantized_" + quantizer_id
    ##### Create folders
    if save_results:
        if not binaries_folder.exists():
            binaries_folder.mkdir(parents=True)
        if not quantized_folder.exists():
            quantized_folder.mkdir(parents=True)
    return binaries_folder, quantized_folder



def evaluate_one_point(args, ModelClass, model_id):
    # Define output paths
    binary_file_path    = os.path.join(args.binaries_folder, os.path.splitext(os.path.basename(args.image_to_quantize))[0] + '.bin')
    rec_image_file_path = os.path.join(args.binaries_folder, os.path.splitext(os.path.basename(args.image_to_quantize))[0] + '_rec.png')
    # Instantiate Model
    quantizer = ModelClass(args.image_to_quantize, binary_file_path, rec_image_file_path, args.N, args.M)
    # Quantize image
    quantizer.encode_image()
    quantizer.decode_binary()
    # Save files, if required
    if args.save_results:
        quantizer.save_binary_file()
        quantizer.save_quantized_image()
    # Display image comparison
    distortion_meter = Distortion_Evaluation()
    distortion_meter.display_comparison(quantizer.image, quantizer.quantized_image, quantizer.bitstring, model_id)
    return



def global_evaluation(args, N_values, M_values, ModelClass, quantizer_id):
    # Instantiate distortion meter
    distortion_meter = Distortion_Evaluation()
    # Quantize and compute metrics from multiple combinations
    for M in M_values:
        for N in N_values:
            # Quantize image.
            binary_file_path    = os.path.join(args.binaries_folder, os.path.splitext(os.path.basename(args.image_to_quantize))[0] + f'_N={N}_M={M}.bin')
            rec_image_file_path = os.path.join(args.binaries_folder, os.path.splitext(os.path.basename(args.image_to_quantize))[0] + f'_N={N}_M={M}_rec.png')
            quantizer = ModelClass(args.image_to_quantize, binary_file_path, rec_image_file_path, N, M)
            quantizer.encode_image()
            quantizer.decode_binary()
            if args.save_results:
                quantizer.save_binary_file()
                quantizer.save_quantized_image()
            # Include results to the PSNR meter
            distortion_meter.get_img_pairs_and_bitstring(quantizer.image, quantizer.quantized_image, quantizer.bitstring, N, M)
            # Signalize end of quantization.
            print(f"--------------------\n End of quantization with parameters 'N'={N} and 'M'={M}.\n--------------------\n")
    # Plot points and RD curve.               
    distortion_meter.plot_rd_curve(quantizer_id, os.path.basename(args.image_to_quantize))