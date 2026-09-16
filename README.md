# COMP3710 Demonstration 2 - Part 4

This project implements the three hard-route recognition tasks using the
preprocessed OASIS brain MRI dataset:

1. Variational Autoencoder (VAE)
2. UNet categorical segmentation
3. Generative Adversarial Network (GAN)

## Dataset

The course dataset is provided on Rangpur at `/home/groups/comp3710/OASIS/`.
It contains 256 x 256 grayscale MRI slices stored as PNG files ending in
`.nii.png`, with supplied training, validation and test splits. The shared
dataset is read directly from the course directory and is not included in
this repository.

Segmentation classes are identified only as classes 0-3; their anatomical
meanings have not been established.

## Software

- Python
- PyTorch
- Training performed on the UQ Rangpur GPU cluster

Run the commands below from `~/COMP3710/lab2/part4` in a suitable GPU session
on Rangpur.

## Task 1 - VAE

`vae.py` manually implements a convolutional VAE in PyTorch for 256 x 256
grayscale MRI images. The encoder produces `mu` and `logvar`, and an explicit
reparameterisation step samples the latent vector. A convolutional decoder
reconstructs the MRI. Training combines reconstruction loss (summed pixel
MSE) with KL divergence.

The latent dimension is 2, allowing direct visualisation of the learned
manifold by decoding a regular grid of latent coordinates.

```bash
python -B vae.py
```

Outputs:

- [Reconstruction examples](results/vae_reconstructions.png)
- [2D latent manifold](results/vae_manifold.png)

## Task 2 - UNet

`unet.py` manually implements a UNet in PyTorch for categorical semantic
segmentation. It takes input of shape `[B, 1, 256, 256]` and produces four
raw logits per pixel, with output shape `[B, 4, 256, 256]`. Skip connections
connect matching encoder and decoder features.

Mask intensities are mapped to integer class indices:

| Mask intensity | Class |
| --- | --- |
| 0 | class 0 |
| 85 | class 1 |
| 170 | class 2 |
| 255 | class 3 |

Training combines `CrossEntropyLoss`, applied to raw logits, with multiclass
soft Dice loss. The best checkpoint is selected by the highest minimum
validation Dice Similarity Coefficient (DSC) among all four classes.

Verified test results:

| Measure | DSC |
| --- | ---: |
| Class 0 | 0.999320 |
| Class 1 | 0.962835 |
| Class 2 | 0.965061 |
| Class 3 | 0.979312 |
| Mean | 0.976632 |
| Minimum | 0.962835 |

All four classes achieved DSC > 0.90 on the supplied test split.

Training:

```bash
python -B unet.py
```

Saved-model test inference, without retraining:

```bash
python -B unet.py --eval-only
```

Output: [Test segmentation examples](results/unet_segmentation.png).

The trained `unet_best.pt` checkpoint is retained on Rangpur beside the
script for live demonstration inference. It is not included in Git and
must not be added to the repository.

## Task 3 - GAN

`gan.py` manually implements a DCGAN in PyTorch, trained for 60 epochs using
OASIS MRI slices 8 through 23 inclusive. The latent dimension is 128.

The generator uses `ConvTranspose2d`, BatchNorm and ReLU, followed by Tanh,
to produce images of shape `[1, 256, 256]`. Real MRI intensities are scaled
to `[-1, 1]` to match the generator output. The discriminator uses strided
`Conv2d` and LeakyReLU and outputs one raw logit per image. Training uses
`BCEWithLogitsLoss` and separate Adam optimisers for the two networks.

Fixed latent vectors show development across training stages. The final
6 x 6 sample grid uses 36 independent random latent vectors. Realism and
diversity are inspected visually for mode collapse; loss curves alone do
not establish either.

Training:

```bash
python -B gan.py
```

Generate fresh samples from the saved model, without retraining:

```bash
python -B gan.py --generate-only
```

Outputs:

- [Generated samples](results/gan_samples.png)
- [Fixed-noise training progress](results/gan_progress.png)
- [Generator and discriminator loss curves](results/gan_losses.png)

The trained `gan_final.pt` checkpoint is retained on Rangpur beside the
script for the live demonstration. It is not included in Git and must not
be added to the repository.

## Project structure

```text
part4/
├── README.md
├── vae.py
├── unet.py
├── gan.py
└── results/
    ├── vae_reconstructions.png
    ├── vae_manifold.png
    ├── unet_segmentation.png
    ├── gan_samples.png
    ├── gan_progress.png
    └── gan_losses.png
```
