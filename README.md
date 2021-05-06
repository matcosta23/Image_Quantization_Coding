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
<python version> <quantizer script> --image_to_quantize <Path to image to be quantized> --N <N parameter> --M <M parameter> -s --binaries_folder <Folder to save binary file> --quantized_folder <Folder to save quantized file>
```

1. The N and M parameters have the purposes explained above. Note that for color quantization the dimension of the vectors is already defined, so the N parameter does not apply.

2. The adaptive and vector quantizers ***only deal with grayscale images***. The color quantizer only handles RGB images.

3. The available values for each of the quantizers are:

    - **Adaptive**: N less than or equal to 16; M less than or equal to 32;
    - **Vector**: N in [2, 4, 8, 16]; M in [32, 64, 128, 256];
    - **Color**: M in [16, 32, 64, 128, 256].

4. The '-s' flag is intended to signal that the binary file and the quantized image should be saved on disk. When set, the parameters *binaries_folder* and *quantized_folder* can be used to determine the destination directories. However, they are not mandatory. There are default folders.

#### Evaluate Quantizer Globally.

In order to evaluate the quantizer at multiple operating points, the '-g' flag can be used. 

Additionally, the parameter "--metrics_folder" is intended to receive a custom path to the directory where a csv with MSE, PSNR and BPP rate for each of the points and a scatter plot comparing MSE and BPP will be saved. This parameter is not mandatory either.

```bash
<python version> <quantizer script> --image_to_quantize <Path to image to be quantized> -g -s --binaries_folder <Folder to save binaries> --quantized_folder <Folder to save quantized files> --metrics_folder <Folder to save quantizer metrics>
```

### Dithering Quantization

The command for evaluating the Floyd-Steinberg algorithm is illustrated below.

```bash
<python version> dithering_quantization.py --image_to_quantize <Path to image to be quantized> -s --levels <8 or 16> --uniform_folder <Folder to save uniformly quantized file> --dithering_folder <Folder to save dithering images>
```

- For the case of grayscale images, the pixels are quantized into "--levels" values within the range [0, 255]. The number of levels available are 8 and 16.

- Colored images also undergo uniform quantization. With 8 levels, the used RGB colors are obtained from three-element combinations of the values 0 and 255. If number of levels is 16, the other 8 combinations are obtained from analogous combinations for values 85 and 170.

- The parameters '-s', '--uniform_folder' and '--dithering_folder' work in an equivalent way to that shown for the other quantizers.