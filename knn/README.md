# Results

#### Calculation of Memory usage

Image size:
  
  (Width * Height * Channels * Computer number format * Total images) / 1000000 = Number of Mega Bytes
  
  (32 * 32 * 3 * 8 * 1875) / 1000000 = 46.08 [MB]
  
  (64 * 64 * 3 * 8 * 1875) / 1000000 = 184.32 [MB]
  
  (128 * 128 * 3 * 8 * 1875) / 1000000 = 737.28 [MB]
 
Bin size:
  
  (Width * Height * Depth * Computer number format * Total images) / 1000000 = Number of Mega Bytes
  
  (8 * 8 * 8 * 8 * 1875) / 1000000 = 7.68 [MB]
  
  (16 * 16 * 16 * 8 * 1875) / 1000000 = 61.44 [MB]
  
  (32 * 32 * 32 * 8 * 1875) / 1000000 = 491.62 [MB]

### Pixel feature on full images

| Image type | Image size | Memory usage | Prediction time | Accuracy|
|:----------:|:----------:|:------------:|:---------------:|:-------:|
| Normal | 32x32 | 46.08 [MB] | 0.0605 [s] | 63.1 % |
| Normal | 64x64 | 184.32 [MB] | 0.241 [s] | 64.6 % |
| Normal | 128x128 | 737.28 [MB] | 0.928 [s] | 65.0 % |
| Smoothed | 32x32 | 46.08 [MB] | 0.0578 [s] | 65.7 % |
| Smoothed | 64x64 | 184.32 [MB] | 0.207 [s] | 66.3 % |
| Smoothed | 128x128 | 737.28 [MB] | 0.784 [s] | 65.7 % |
| Filtered | 32x32 | 46.08 [MB] | 0.0616 [s] | 44.9 % |
| Filtered | 64x64 | 184.32 [MB] | 0.235 [s] | 45.7 % |
| Filtered | 128x128 | 737.28 [MB] | 0.859 [s] | 47.7 % |
| Smoothed and filtered | 32x32 | 46.08 [MB] | 0.0774 [s] | 51.3 % |
| Smoothed and filtered | 64x64 | 184.32 [MB] | 0.211 [s] | 49.4 % |
| Smoothed and filtered | 128x128 | 737.28 [MB] | 0.859 [s] | 50.6 % |

### Pixel feature on cropped images

| Image type | Image size | Memory usage | Prediction time | Accuracy|
|:----------:|:----------:|:------------:|:---------------:|:-------:|
| Normal | 32x32 | 46.08 [MB] | 0.0588 [s] | 76.7 % |
| Normal | 64x64 | 184.32 [MB] | 0.234 [s] | 73.3 % |
| Normal | 128x128 | 737.28 [MB] | 0.935 [s] | 74.2 % |
| Smoothed | 32x32 | 46.08 [MB] | 0.0661 [s] | 77.3 % |
| Smoothed | 64x64 | 184.32 [MB] | 0.210 [s] | 77.1 % |
| Smoothed | 128x128 | 737.28 [MB] | 0.855 [s] | 77.1 % |
| Filtered | 32x32 | 46.08 [MB] | 0.0647 [s] | 64.4 % |
| Filtered | 64x64 | 184.32 [MB] | 0.230 [s] | 68.9 % |
| Filtered | 128x128 | 737.28 [MB] | 0.826 [s] | 65.5 % |
| Smoothed and filtered | 32x32 | 46.08 [MB] | 0.0586 [s] | 70.8 % |
| Smoothed and filtered | 64x64 | 184.32 [MB] | 0.212 [s] | 72.2 % |
| Smoothed and filtered | 128x128 | 737.28 [MB] | 0.817 [s] | 73.1 % |

### Histogram feature on full images

| Image type | Bin size | Memory usage | Prediction time | Accuracy|
|:----------:|:--------:|:------------:|:---------------:|:-------:|
| Normal | (8, 8, 8) | 7.68 [MB] | 0.0207 [s] | 85.2 % |
| Normal | (16, 16, 16) | 61.44 [MB] | 0.0954 [s] | 87.7 % |
| Normal | (32, 32, 32) | 491.52 [MB] | 0.569 [s] | 90.0 % |
| Smoothed | (8, 8, 8) | 7.68 [MB] | 0.0141 [s] | 84.1 % |
| Smoothed | (16, 16, 16) | 61.44 [MB] | 0.0769 [s] | 87.0 % |
| Smoothed | (32, 32, 32) | 491.52 [MB] | 0.520 [s] | 89.0 % |
| Filtered | (8, 8, 8) | 7.68 [MB] | 0.0238 [s] | 95.6 % |
| Filtered | (16, 16, 16) | 61.44 [MB] | 0.113 [s] | 95.6 % |
| Filtered | (32, 32, 32) | 491.52 [MB] | 0.823 [s] | 93.6 % |
| Smoothed and filtered | (8, 8, 8) | 7.68 [MB] | 0.0165 [s] | 94.1 % |
| Smoothed and filtered | (16, 16, 16) | 61.44 [MB] | 0.0817 [s] | 94.9 % |
| Smoothed and filtered | (32, 32, 32) | 491.52 [MB] | 0.524 [s] | 95.3 % |

### Histogram feature on cropped images

| Image type | Bin size | Memory usage | Prediction time | Accuracy|
|:----------:|:--------:|:------------:|:---------------:|:-------:|
| Normal | (8, 8, 8) | 7.68 [MB] | 0.0148 [s] | 73.7 % |
| Normal | (16, 16, 16) | 61.44 [MB] | 0.0822 [s] | 70.8 % |
| Normal | (32, 32, 32) | 491.52 [MB] | 0.572 [s] | 71 % |
| Smoothed | (8, 8, 8) | 7.68 [MB] | 0.0147 [s] | 68.6 % |
| Smoothed | (16, 16, 16) | 61.44 [MB] | 0.0784 [s] | 67.3 % |
| Smoothed | (32, 32, 32) | 491.52 [MB] | 0.563 [s] | 69.5 % |
| Filtered | (8, 8, 8) | 7.68 [MB] | 0.0244 [s] | 95.6 % |
| Filtered | (16, 16, 16) | 61.44 [MB] | 0.0804 [s] | 94.5 % |
| Filtered | (32, 32, 32) | 491.52 [MB] | 0.572 [s] | 94.7 % |
| Smoothed and filtered | (8, 8, 8) | 7.68 [MB] | 0.0151 [s] | 91.7 % |
| Smoothed and filtered | (16, 16, 16) | 61.44 [MB] | 0.0784 [s] | 92.6 % |
| Smoothed and filtered | (32, 32, 32) | 491.52 [MB] | 0.546 [s] | 93.9 % |
