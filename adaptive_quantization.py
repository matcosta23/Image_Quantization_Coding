import os
import math
import numpy as np
from PIL import Image 
from bitstring import BitStream

import common_classes
from metrics_evaluation import PSNR_Evaluation


class CIQA(common_classes.LossyCompression):

    def encode_image(self):
        ##### Read input image and separate into coding blocks.
        self._separate_blocks()
        ##### Quantize image.
        self._compute_coefficients()
        ##### Instantiate and write bitstring
        self._write_bitstring()
        return


    def decode_binary(self):
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
        quantized_patches = np.clip(np.floor(normalized_patches / deltas), 0, self.M - 1)

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
        patches_shape = self._compute_dimensions(patches_diff, patches_amount)
        image_shape = patches_shape * self.N
        ##### Decode patches
        self.quantized_image = np.zeros(image_shape)
        for v in range(patches_shape[0]):
            for h in range(patches_shape[1]):
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



if __name__ == "__main__":
    ##### Read arguments from command line
    args = common_classes.read_arguments()
    ##### Create directories
    args.binaries_folder, args.quantized_folder = common_classes.create_folders(args.binaries_folder, args.quantized_folder, "Adaptive", args.save_results)
    ##### Verify hyper-parameters values
    if args.M > 16:
        raise ValueError("Quantization supports only up to 16 levels (M parameter).")
    if args.N > 32:
        raise ValueError("The maximum dimension for the square block is 32 pixels (N parameter).")
    ##### Verify if only one or all points should be runned.
    if args.global_evaluation is False:
        # Define output paths
        binary_file_path    = os.path.join(args.binaries_folder, os.path.splitext(os.path.basename(args.image_to_quantize))[0] + '.bin')
        rec_image_file_path = os.path.join(args.binaries_folder, os.path.splitext(os.path.basename(args.image_to_quantize))[0] + '_rec.png')
        # Instantiate Model
        adaptive_quantizer = CIQA(args.image_to_quantize, binary_file_path, rec_image_file_path, args.N, args.M)
        # Quantize image
        adaptive_quantizer.encode_image()
        adaptive_quantizer.decode_binary()
        # Save files, if required
        if args.save_results:
            adaptive_quantizer.save_binary_file()
            adaptive_quantizer.save_quantized_image()
        # Display image comparison
        psnr_meter = PSNR_Evaluation()
        psnr_meter.display_comparison(adaptive_quantizer.image, adaptive_quantizer.quantized_image, adaptive_quantizer.bitstring, "Adaptive Quantizer")
    ##### Perform global evaluation    
    else:
        # Define parameters values
        M_values = np.array([2, 4, 8, 16])
        N_values = np.array([4, 8, 16, 32])
        # Innstantiate PSNR meter
        psnr_meter = PSNR_Evaluation()
        # Quantize and compute metrics from multiple combinations
        for M in M_values:
            for N in N_values:
                # Quantize image.
                binary_file_path    = os.path.join(args.binaries_folder, os.path.splitext(os.path.basename(args.image_to_quantize))[0] + f'_N={N}_M={M}.bin')
                rec_image_file_path = os.path.join(args.binaries_folder, os.path.splitext(os.path.basename(args.image_to_quantize))[0] + f'_N={N}_M={M}_rec.png')
                adaptive_quantizer = CIQA(args.image_to_quantize, binary_file_path, rec_image_file_path, N, M)
                adaptive_quantizer.encode_image()
                adaptive_quantizer.decode_binary()
                if args.save_results:
                    adaptive_quantizer.save_binary_file()
                    adaptive_quantizer.save_quantized_image()
                # Include results to the PSNR meter
                psnr_meter.get_img_pairs_and_bitstring(adaptive_quantizer.image, adaptive_quantizer.quantized_image, adaptive_quantizer.bitstring, N, M)
        # Plot points and RD curve.               
        psnr_meter.plot_rd_curve("Adaptive Quantizer", os.path.basename(args.image_to_quantize))