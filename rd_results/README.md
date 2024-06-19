# Quantitative Results

Rate-distortion-realism results of our pre-trained model using `17 quality values: {0.00, 0.25, ..., 3.75, 4.00}` and `2 beta values: {0.00, 3.84}`.

`CLIC_reproduce.csv` compiles the results of our reproduced training, not those reported in the paper.

### Datasets:
- [CLIC test dataset](https://www.tensorflow.org/datasets/catalog/clic) (428 images)
- [DIV2K HR Validation dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (100 images)
- [Kodak dataset](https://r0k.us/graphics/kodak/) (24 images)

### Metrics:
- bpp (bits per pixel)
- PSNR
- FID: Followed protocol used in HiFiC (Mentzer+, NeurIPS2020).
- LPIPS
- DISTS