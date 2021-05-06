# Project 3 - Image Quantization

### Author:
- Name: Matheus Costa de Oliveira
- Registration: 17/0019039
### Course Information.
- Subject: Fundamentals of Signal Compression
- Department of Electrical Engineering
- University of Brasilia

___

## Overview of Available Quantizations.

### Adaptive Quantization.

- The provided image is divided into N x N patches, for which their pixels are quantized uniformly at M levels.

### Vector Quantization.

- Similarly, the image is subdivided into N x N patches. However, the block consisting of the N^2 pixels is interpreted as a single vector. Thus, it is possible to find the M vectors that best represent the image from the [LBG Algorithm](https://en.wikipedia.org/wiki/Linde%E2%80%93Buzo%E2%80%93Gray_algorithm).

### Color Quantization.

- Color quantization is completely analogous to vector quantization. However, instead of considering blocks, quantization is performed over three-dimensional vectors that represent the RGB colors of the pixels. In the same way, the user chooses a number M of colors to represent the entire image.

### Floyd-Steinberg Dithering.

- The [Floyd-Steinberg dithering algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering) aims to insert noise into quantized images in order to obtain subjectively more pleasing results.
- For comparison, non-adaptive uniform quantization is applied to the images. Using the same coefficients, the results obtained by the dithering technique are compared objectively and subjectively.

## Script Descriptions.

### Adaptive, Vector and Color Quantizers.

- From the brief description of the first three quantizers, it can be concluded that the implementations and the arguments received have several similarities.
- In this context, the following commands can be generalized to the scripts [adaptive_quantization](adaptive_quantization.py), [vector_quantization](vector_quantization.py) and [color_map_quantization](color_map_quantization.py).

#### Evaluate Only One Operation Point.

Running the models for only one set of hyper-parameters is achieved by the following command:

```bash
<python version> <quantizer script> \
    --image_to_quantize <Path to image to be quantized> \
    --N <N parameter> \
    --M <M parameter> \
    -s \
    --binaries_folder <Folder to save binary file> \
    --quantized_folder <Folder to save quantized file>
```

- The N and M parameters have the purposes explained above. Note that for color quantization the dimension of the vectors is already defined, so the N parameter does not apply.

- The adaptive and vector quantizers ***only deal with grayscale images***. The color quantizer only handles RGB images.

- The available values for each of the quantizers are:

    - **Adaptive**: N less than or equal to 16; M less than or equal to 32;
    - **Vector**: N in [2, 4, 8, 16]; M in [32, 64, 128, 256];
    - **Color**: M in [16, 32, 64, 128, 256].

- The '-s' flag is intended to signal that the binary file and the quantized image should be saved on disk. When set, the parameters *binaries_folder* and *quantized_folder* can be used to determine the destination directories. However, they are not mandatory. There are default folders.

#### Evaluate Quantizer Globally.

In order to evaluate the quantizer at multiple operating points, the '-g' flag can be used. 

Additionally, the parameter "--metrics_folder" is intended to receive a custom path to the directory where a csv with MSE, PSNR and BPP rate for each of the points and a scatter plot comparing MSE and BPP will be saved. This parameter is not mandatory either.

```bash
<python version> <quantizer script> \
    --image_to_quantize <Path to image to be quantized> \
    -g \
    -s \
    --binaries_folder <Folder to save binaries> \
    --quantized_folder <Folder to save quantized files> \
    --metrics_folder <Folder to save quantizer metrics>
```

### Dithering Quantization

The command for evaluating the Floyd-Steinberg algorithm is illustrated below.

```bash
<python version> dithering_quantization.py \
    --image_to_quantize <Path to image to be quantized> \
    -s \
    --levels <8 or 16> \
    --uniform_folder <Folder to save uniformly quantized file> \
    --dithering_folder <Folder to save dithering images>
```

- For the case of grayscale images, the pixels are quantized into "--levels" values within the range [0, 255]. The number of levels available are 8 and 16.

- Colored images also undergo uniform quantization. With 8 levels, the used RGB colors are obtained from three-element combinations of the values 0 and 255. If number of levels is 16, the other 8 combinations are obtained from analogous combinations for values 85 and 170.

- The parameters '-s', '--uniform_folder' and '--dithering_folder' work in an equivalent way to that shown for the other quantizers.

### Plot RD Curve.

In order to evaluate the performance of a given quantizer on an image, you can use the [plot_RD_against_JPEG](plot_RD_against_JPEG.py) file to generate the rate-distortion curves of the quantizer compared to JPEG.

The script reads a 'csv' file that is created when the quantizer is evaluated globally ('-g'), generates JPEG versions with the same size as the obtained bitstrings and compares the algorithms by means of PSNR.

The conversion of the original files into 'jpg' and the quality control are performed by the programs [ImageMagick](https://techpiezo.com/linux/install-imagemagick-in-ubuntu-20-04-lts/) and [Jpegoptim](https://www.omgubuntu.co.uk/2016/03/how-to-optimize-jpeg-command-line-linux), respectively. **These programs must be preinstalled**.

The command line for executing the file is illustrated below:

```bash
<python version> plot_RD_against_JPEG.py \
    --original_image <Path to original image> \
    --binaries_glob <Glob with binaries created by global evaluation>\
    --csv_path <Path to csv file with metrics generated by global evaluation> \
    --csv_indexes <Csv indices to be considered to plot the curve>
```

- The '--binaries_glob' parameter should be given a glob that leads to the binary files generated by the quantizer of interest after having its multiple points evaluated. Similarly, the '--csv_path' should receive the path to the csv with the metrics.

- The '--csv_indexes' argument is not required. It can be given multiple values. These numbers are the points of interest for the RD curve. Thus, it is not necessary to use all the points in the csv, or to filter the csv beforehand.