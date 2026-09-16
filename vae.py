"""COMP3710 Part 4 Task 1: a convolutional VAE for OASIS MRI slices.

Run on Rangpur with: python -B vae.py
Only the supplied MRI splits are read; segmentation masks are not used.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


DATA_ROOT = Path('/home/groups/comp3710/OASIS')
RESULTS = Path(__file__).resolve().parent / 'results'
LATENT_DIM = 2


class MRISlices(Dataset):
    def __init__(self, split):
        directory = DATA_ROOT / f'keras_png_slices_{split}'
        self.files = sorted(directory.glob('case_*.nii.png'))
        if not self.files:
            raise FileNotFoundError(f'No MRI PNG slices found in {directory}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            pixels = np.array(image.convert('L'), dtype=np.float32)
        if pixels.shape != (256, 256):
            raise ValueError(f'Unexpected MRI shape: {self.files[index]}: {pixels.shape}')
        # [height, width] -> [channel, height, width], with intensities in [0, 1].
        return torch.from_numpy(pixels).unsqueeze(0) / 255.0


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial dimensions: 256 -> 128 -> 64 -> 32 -> 16.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.mu = nn.Linear(128 * 16 * 16, LATENT_DIM)
        self.logvar = nn.Linear(128 * 16 * 16, LATENT_DIM)
        self.decoder_input = nn.Linear(LATENT_DIM, 128 * 16 * 16)
        # Reverse the downsampling; sigmoid bounds reconstructed intensities.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, images):
        features = self.encoder(images)
        return self.mu(features), self.logvar(features)

    @staticmethod
    def reparameterise(mu, logvar):
        # q(z|x) is a diagonal Gaussian; this sampling keeps gradients flowing.
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        features = self.decoder_input(z).reshape(-1, 128, 16, 16)
        return self.decoder(features)

    def forward(self, images):
        mu, logvar = self.encode(images)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(reconstructions, images, mu, logvar):
    # Sum squared pixel errors per image, then average over the batch.
    # MSE corresponds to a fixed-variance Gaussian reconstruction model.
    reconstruction = F.mse_loss(reconstructions, images, reduction='sum')
    # KL(q(z|x) || N(0, I)), summed over both latent coordinates and the batch.
    kl = -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp())
    return (reconstruction + kl) / images.size(0)


def run_epoch(model, loader, device, optimiser=None):
    training = optimiser is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    with torch.set_grad_enabled(training):
        for images in loader:
            images = images.to(device, non_blocking=True)
            if training:
                optimiser.zero_grad(set_to_none=True)
            reconstructions, mu, logvar = model(images)
            loss = vae_loss(reconstructions, images, mu, logvar)
            if training:
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * images.size(0)
            count += images.size(0)
    return total_loss / count


def grayscale_image(tensor):
    pixels = tensor.detach().cpu().squeeze(0).clamp(0, 1)
    return Image.fromarray((pixels.numpy() * 255).round().astype(np.uint8))


@torch.no_grad()
def save_reconstructions(model, dataset, device, path):
    model.eval()
    # Six different test cases, using a central slice rather than blank edges.
    files_by_case = {}
    for index, filename in enumerate(dataset.files):
        case = filename.name.split('_')[1]
        files_by_case.setdefault(case, []).append(index)
    case_ids = sorted(files_by_case)
    positions = np.linspace(0, len(case_ids) - 1, 6, dtype=int)
    indices = []
    for position in positions:
        members = files_by_case[case_ids[position]]
        members.sort(key=lambda i: int(dataset.files[i].name.split('_')[-1].split('.')[0]))
        indices.append(members[len(members) // 2])
    originals = torch.stack([dataset[i] for i in indices]).to(device)
    mu, _ = model.encode(originals)
    # Decode posterior means for repeatable reconstruction examples.
    reconstructions = model.decode(mu)
    canvas = Image.new('L', (120 + 6 * 256, 580), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.text((120, 8), 'OASIS test MRI: originals and VAE reconstructions (decode mu)', fill=0)
    draw.text((8, 150), 'Original MRI', fill=0)
    draw.text((8, 416), 'Reconstruction', fill=0)
    for column, index in enumerate(indices):
        x = 120 + column * 256
        draw.text((x + 4, 30), dataset.files[index].name, fill=0)
        canvas.paste(grayscale_image(originals[column]), (x, 52))
        canvas.paste(grayscale_image(reconstructions[column]), (x, 318))
    canvas.save(path, format='PNG')


@torch.no_grad()
def save_manifold(model, device, path):
    model.eval()
    # Each tile is decoder(z1, z2), with no dimensionality reduction.
    coordinates = torch.linspace(-3, 3, 11)
    tile = 128
    left, top = 56, 70
    canvas = Image.new('L', (left + 11 * tile, top + 11 * tile), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 8), 'VAE learned 2D latent manifold: decoder(z1, z2)', fill=0)
    draw.text((left, 26), 'z1 increases left to right; z2 increases bottom to top; range [-3, 3]', fill=0)
    for column, value in enumerate(coordinates):
        draw.text((left + column * tile + 50, 50), f'{value.item():.1f}', fill=0)
    for row, z2 in enumerate(coordinates.flip(0)):
        z = torch.stack((coordinates, torch.full_like(coordinates, z2.item())), dim=1)
        decoded = model.decode(z.to(device))
        draw.text((8, top + row * tile + 60), f'{z2.item():.1f}', fill=0)
        for column, sample in enumerate(decoded):
            image = grayscale_image(sample).resize((tile, tile), Image.Resampling.BILINEAR)
            canvas.paste(image, (left + column * tile, top + row * tile))
    canvas.save(path, format='PNG')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('epochs and batch-size must be positive; workers must be nonnegative')

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}', flush=True)
    train_data = MRISlices('train')
    validation_data = MRISlices('validate')
    print(f'MRI slices: train={len(train_data)}, validation={len(validation_data)}', flush=True)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == 'cuda')
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    model = VAE().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f'Training: {args.epochs} epochs, batch size {args.batch_size}, latent_dim={LATENT_DIM}', flush=True)
    for epoch in range(1, args.epochs + 1):
        training_loss = run_epoch(model, train_loader, device, optimiser)
        validation_loss = run_epoch(model, validation_loader, device)
        print(f'Epoch {epoch:02d}/{args.epochs} | Training loss: {training_loss:.4f} | '
              f'Validation loss: {validation_loss:.4f}', flush=True)

    # The test split is accessed only after training, for final examples.
    test_data = MRISlices('test')
    print(f'Test MRI slices: {len(test_data)}', flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    reconstruction_path = RESULTS / 'vae_reconstructions.png'
    manifold_path = RESULTS / 'vae_manifold.png'
    save_reconstructions(model, test_data, device, reconstruction_path)
    save_manifold(model, device, manifold_path)
    print(f'Saved: {reconstruction_path}', flush=True)
    print(f'Saved: {manifold_path}', flush=True)


if __name__ == '__main__':
    main()
