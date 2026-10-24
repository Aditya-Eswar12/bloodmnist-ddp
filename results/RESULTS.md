# Results (BloodMNIST 64×64 test set: 3,421 images, unless stated)

## Main comparison

| Model | Seeds | Accuracy | Macro-F1 | Monocyte recall | Mono→IG (of 284) |
|---|---|---|---|---|---|
| VGG-style CNN, supervised | 3 | 0.979 ± 0.001 | 0.979 ± 0.002 | 0.973 ± 0.002 | 3 ± 1 |
| ResNet-18, supervised | 1 | 0.980 | 0.980 | 0.958 | 5 |
| ViT, supervised from scratch | 1 | 0.956 | 0.954 | 0.933 | 10 |
| VGG-style CNN, 19 epochs (step-matched control for the cGAN row) | 3 | 0.980 ± 0.005 | 0.980 ± 0.006 | 0.975 ± 0.009 | 3 ± 3 |
| VGG-style CNN + cGAN top-up of 3 smallest classes | 3 | 0.982 ± 0.001 | 0.982 ± 0.001 | 0.966 ± 0.005 | 5 ± 2 |
| VGG-style CNN + scale and colour augmentation | 3 | 0.935 ± 0.003 | 0.922 ± 0.003 | 0.897 ± 0.070 | 28 ± 19 |
| MAE, random masking + linear probe | 3 | 0.931 ± 0.006 | 0.924 ± 0.008 | 0.891 ± 0.009 | 21 ± 2 |
| MAE, A: hide cell patches 3x + linear probe | 3 | 0.928 ± 0.001 | 0.920 ± 0.001 | 0.883 ± 0.011 | 19 ± 3 |
| MAE, B: cross-compartment views + linear probe | 3 | 0.932 ± 0.003 | 0.925 ± 0.005 | 0.898 ± 0.011 | 20 ± 3 |
| MAE, C: B + confusion feedback + linear probe | 3 | 0.928 ± 0.003 | 0.921 ± 0.006 | 0.887 ± 0.015 | 21 ± 4 |

## Label efficiency (test macro-F1)

| Method | 1% labels (119 img) | 5% (598) | 10% (1,196) | 100% |
|---|---|---|---|---|
| CNN, trained end to end | 0.863 ± 0.017 | 0.954 ± 0.004 | 0.968 ± 0.003 | 0.979 ± 0.002 |
| ViT from scratch, end to end | 0.707 ± 0.038 | 0.922 | 0.922 | 0.954 |
| ViT fine-tuned from MAE (random masking) | 0.820 ± 0.014 | 0.927 | 0.948 | – |
| ViT fine-tuned from MAE (policy B) | 0.816 ± 0.012 | 0.924 | 0.947 | – |
| Linear probe on MAE (random masking) | 0.717 ± 0.012 | 0.842 ± 0.010 | 0.874 ± 0.007 | 0.924 ± 0.008 |
| Linear probe on MAE (policy A) | 0.738 ± 0.012 | 0.844 ± 0.009 | 0.875 ± 0.006 | 0.920 ± 0.001 |
| Linear probe on MAE (policy B) | 0.743 ± 0.017 | 0.848 ± 0.009 | 0.876 ± 0.007 | 0.925 ± 0.005 |
| Linear probe on MAE (policy C) | 0.740 ± 0.017 | 0.838 ± 0.008 | 0.872 ± 0.007 | 0.921 ± 0.006 |

## External images (accuracy; 5 white-cell classes; never used for training)

D1: 300 images, a different stain. D2: 100 CellaVision images from other labs. raw = resized as is; scale = cell cropped to the BloodMNIST size in frame; stain = Reinhard colour normalisation.

| Model | D1 raw | D1 scale | D1 stain | D1 scale+stain | D2 raw | D2 scale | D2 stain | D2 scale+stain |
|---|---|---|---|---|---|---|---|---|
| baseline_cnn | 0.076 ± 0.010 | 0.032 ± 0.005 | 0.212 ± 0.064 | 0.154 ± 0.038 | 0.240 ± 0.026 | 0.483 ± 0.025 | 0.193 ± 0.061 | 0.370 ± 0.035 |
| baseline_cnn_strongaug | 0.149 ± 0.027 | 0.088 ± 0.008 | 0.279 ± 0.039 | 0.203 ± 0.027 | 0.420 ± 0.085 | 0.467 ± 0.040 | 0.350 ± 0.075 | 0.387 ± 0.045 |
| baseline_resnet18 | 0.057 | 0.013 | 0.133 | 0.170 | 0.240 | 0.400 | 0.190 | 0.330 |
| baseline_vit | 0.083 | 0.057 | 0.233 | 0.103 | 0.210 | 0.430 | 0.170 | 0.360 |
| mae_A | 0.094 ± 0.039 | 0.071 ± 0.004 | 0.224 ± 0.100 | 0.173 ± 0.081 | 0.180 ± 0.030 | 0.327 ± 0.025 | 0.200 ± 0.017 | 0.327 ± 0.006 |
| mae_B | 0.084 ± 0.010 | 0.071 ± 0.002 | 0.318 ± 0.129 | 0.139 ± 0.083 | 0.117 ± 0.021 | 0.253 ± 0.029 | 0.123 ± 0.046 | 0.290 ± 0.035 |
| mae_C | 0.073 ± 0.000 | 0.071 ± 0.002 | 0.292 ± 0.162 | 0.111 ± 0.056 | 0.133 ± 0.049 | 0.253 ± 0.035 | 0.190 ± 0.075 | 0.283 ± 0.067 |
| mae_random | 0.054 ± 0.012 | 0.073 ± 0.000 | 0.277 ± 0.192 | 0.111 ± 0.013 | 0.183 ± 0.064 | 0.310 ± 0.053 | 0.217 ± 0.040 | 0.310 ± 0.010 |
