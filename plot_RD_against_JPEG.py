import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from glob import glob
from pathlib import Path

from metrics_evaluation import Distortion_Evaluation


if __name__ == "__main__":
    ##### Define arguments
    parser = argparse.ArgumentParser(description="Receives arguments for plotting RD curve.")
    parser.add_argument('--original_image', required=True, help='Path to original image.')
    parser.add_argument('--binaries_glob', required=True, help="Glob for binaries of interest.")
    parser.add_argument('--jpg_folder', default="JPEG_Images", help="Folder for saving jpeg versions.")
    parser.add_argument('--csv_path', required=True, help='Path for CSV file with metrics.')
    parser.add_argument('--csv_indexes', nargs='+', type=int, help="Indexes of combinations of interest.")
    args = parser.parse_args(sys.argv[1:])

    ##### Convert original Image to JPEG.
    pil_image = Image.open(args.original_image)
    original_image = np.asarray(pil_image)
    pixels_amount = np.prod(original_image.shape[:2])
    # Create JPEG directory
    if not Path(args.jpg_folder).exists():
            Path(args.jpg_folder).mkdir(parents=True)
    # Convert original image with Image Magick
    image_name = os.path.splitext(os.path.basename(args.original_image))[0]
    jpeg_file_path = os.path.join(args.jpg_folder, image_name + '.jpg')
    os.system(f"convert {args.original_image} {jpeg_file_path}")
    
    ##### Read csv and choose indexes.
    metrics_df = pd.read_csv(args.csv_path, index_col=0)
    if args.csv_indexes:
        metrics_df = metrics_df.iloc[args.csv_indexes]
    ##### Lists to save metrics
    model_psnr = []
    model_bpp = []
    jpeg_psnr = []
    jpeg_bpp = []
    ##### Iterate over binaries
    for binary_path in glob(args.binaries_glob):
        # Read PSNR and BPP from csv.
        binary_file_name = os.path.splitext(os.path.basename(binary_path))[0]
        image_name, N, M = binary_file_name.split('_')
        N = int(N[1:])
        M = int(M[1:])
        try:
            row = metrics_df.loc[(metrics_df['N'] == N) & (metrics_df['M'] == M)].iloc[0]
            model_psnr.append(row.PSNR)
            model_bpp.append(row.BPP)
            # Get binary size
            binary_size = os.stat(binary_path).st_size / 1000
            # Create jpeg with same size with 'jpegoptim' software
            resized_img_path = os.path.join(args.jpg_folder, image_name + f"_{int(binary_size)}k.jpg")
            os.system(f"cp {jpeg_file_path} {resized_img_path}")
            os.system(f"jpegoptim --size={int(binary_size)}k {resized_img_path} --overwrite")
            # Read JPEG version
            pil_image = Image.open(resized_img_path)
            resized_image = np.asarray(pil_image)
            # Compute PSNR
            meter = Distortion_Evaluation()
            psnr = meter.psnr(original_image.astype(np.int32), resized_image.astype(np.int32))
            jpeg_psnr.append(psnr)
            # Compute bpp
            jpeg_bpp.append(os.stat(resized_img_path).st_size / pixels_amount)
        except IndexError:
            pass 

    ##### Plot RD curve
    plt.plot(model_bpp, model_psnr, color='darkred', marker='*', label="Proposed Quantizer")
    plt.plot(jpeg_bpp, jpeg_psnr, color='royalblue', marker='^', label="JPEG")
    plt.title('RD Curve')
    plt.xlabel('BPP'), plt.ylabel('PSNR')
    plt.grid(), plt.legend(), plt.tight_layout()
    plt.show()
    ##### Save results inside destiny folder.
    plt.savefig(os.path.join(args.jpg_folder, image_name + '_RD_plot.pdf'), target='pdf')