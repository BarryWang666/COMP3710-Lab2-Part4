"""COMP3710 Part 4 Task 3: a manual DCGAN for central OASIS MRI slices.

Train: python -B gan.py
Generate new samples from the saved model: python -B gan.py --generate-only
Realism and mode collapse must be judged visually, not from GAN losses alone.
"""

import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


TRAIN_DIR = Path('/home/groups/comp3710/OASIS/keras_png_slices_train')
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT = SCRIPT_DIR / 'gan_final.pt'
RESULTS = SCRIPT_DIR / 'results'
LATENT_DIM = 128


class MRISlices(Dataset):
    def __init__(self):
        self.files = []
        for path in sorted(TRAIN_DIR.glob('case_*.nii.png')):
            match = re.fullmatch(r'case_(\d+)_slice_(\d+)\.nii\.png', path.name)
            if match and 8 <= int(match[2]) <= 23:
                self.files.append(path)
        if not self.files:
            raise FileNotFoundError(f'No MRI slices 8-23 found in {TRAIN_DIR}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            pixels = np.array(image.convert('L'), dtype=np.float32)
        if pixels.shape != (256, 256):
            raise ValueError(f'Expected 256x256 MRI: {self.files[index]}, got {pixels.shape}')
        # Match the generator's Tanh range: uint8 [0, 255] -> float32 [-1, 1].
        return torch.from_numpy(pixels).unsqueeze(0) / 127.5 - 1.0


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # A latent noise vector becomes a 4x4 feature map, then doubles in size.
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),  # 4x4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # 64x64
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),  # 128x128
            nn.ConvTranspose2d(16, 1, 4, 2, 1, bias=False),
            nn.Tanh(),  # [B, 1, 256, 256], intensities in [-1, 1].
        )

    def forward(self, noise):
        return self.layers(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Strided convolutions halve the resolution: 256 -> ... -> 4.
        # LeakyReLU retains a small gradient for negative activations.
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # 128x128
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),  # 64x64
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),  # 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),  # 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),  # 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),  # 4x4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # 1x1
        )

    def forward(self, images):
        # One raw logit per image. BCEWithLogitsLoss includes the sigmoid.
        return self.layers(images).flatten()


def initialise_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)


def train_epoch(generator, discriminator, loader, device, optimiser_g, optimiser_d):
    generator.train()
    discriminator.train()
    criterion = nn.BCEWithLogitsLoss()
    total_g, total_d, count = 0.0, 0.0, 0
    for real in loader:
        real = real.to(device, non_blocking=True)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
        fake = generator(noise)

        # Teach D to distinguish real from fake; detach prevents updating G.
        optimiser_d.zero_grad(set_to_none=True)
        real_logits = discriminator(real)
        fake_logits = discriminator(fake.detach())
        real_loss = criterion(real_logits, torch.full_like(real_logits, 0.9))
        fake_loss = criterion(fake_logits, torch.zeros_like(fake_logits))
        loss_d = (real_loss + fake_loss) / 2
        loss_d.backward()
        optimiser_d.step()

        # Teach G to fool the updated D. Gradients flow through D into G,
        # but D's parameters are held fixed during this optimisation step.
        discriminator.requires_grad_(False)
        optimiser_g.zero_grad(set_to_none=True)
        fake_logits = discriminator(fake)
        loss_g = criterion(fake_logits, torch.ones_like(fake_logits))
        loss_g.backward()
        optimiser_g.step()
        discriminator.requires_grad_(True)

        total_g += loss_g.item() * batch_size
        total_d += loss_d.item() * batch_size
        count += batch_size
    return total_g / count, total_d / count


def display_image(sample, size=256):
    # Scaling back to [0, 1] is for display only, never for discriminator input.
    pixels = ((sample.detach().cpu()[0] + 1) / 2).clamp(0, 1)
    image = Image.fromarray((pixels.numpy() * 255).round().astype(np.uint8))
    if size != 256:
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return image


@torch.no_grad()
def save_samples(generator, device, path):
    generator.eval()
    # Independent latent vectors for all 36 tiles; no image is copied to fill a grid.
    noise = torch.randn(36, LATENT_DIM, 1, 1, device=device)
    samples = generator(noise).cpu()
    canvas = Image.new('L', (6 * 256, 44 + 6 * 256), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), 'OASIS DCGAN: 36 independent random latent vectors',
              font=ImageFont.load_default(), fill=0)
    for index, sample in enumerate(samples):
        row, column = divmod(index, 6)
        canvas.paste(display_image(sample), (column * 256, 44 + row * 256))
    canvas.save(path, format='PNG')


def save_progress(stages, path):
    tile, left, top = 128, 120, 54
    canvas = Image.new('L', (left + 8 * tile, top + len(stages) * tile), color=255)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((left, 8), 'Training progress: the same eight latent vectors at every stage', font=font, fill=0)
    for column in range(8):
        draw.text((left + column * tile + 30, 32), f'z {column + 1}', font=font, fill=0)
    for row, (epoch, samples) in enumerate(stages):
        y = top + row * tile
        draw.text((8, y + 52), f'Epoch {epoch}', font=font, fill=0)
        for column, sample in enumerate(samples):
            canvas.paste(display_image(sample, tile), (left + column * tile, y))
    canvas.save(path, format='PNG')


def save_losses(generator_losses, discriminator_losses, path):
    # A simple line plot drawn with Pillow avoids additional plotting dependencies.
    canvas = Image.new('RGB', (960, 540), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    left, right, top, bottom = 85, 920, 85, 465
    epochs = len(generator_losses)
    maximum = max(max(generator_losses), max(discriminator_losses), 1e-6) * 1.1
    draw.text((left, 12), 'DCGAN average losses per epoch', font=font, fill='black')
    draw.text((left, 48), 'Generator loss', font=font, fill='blue')
    draw.text((left + 220, 48), 'Discriminator loss', font=font, fill='darkorange')
    draw.text((8, top - 24), 'Loss', font=font, fill='black')
    draw.text((465, 506), 'Epoch', font=font, fill='black')
    for i in range(6):
        value = maximum * i / 5
        y = bottom - (bottom - top) * i / 5
        draw.line((left, y, right, y), fill='lightgray')
        draw.text((8, y - 10), f'{value:.2f}', font=font, fill='black')
    for epoch in sorted(set(np.linspace(1, epochs, min(7, epochs), dtype=int))):
        x = left + (epoch - 1) * (right - left) / max(1, epochs - 1)
        draw.text((x - 8, bottom + 10), str(epoch), font=font, fill='black')
    draw.line((left, top, left, bottom, right, bottom), fill='black', width=2)
    for losses, colour in [(generator_losses, 'blue'), (discriminator_losses, 'darkorange')]:
        points = [(left + i * (right - left) / max(1, epochs - 1),
                   bottom - value / maximum * (bottom - top)) for i, value in enumerate(losses)]
        if len(points) > 1:
            draw.line(points, fill=colour, width=3)
        else:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colour)
    canvas.save(path, format='PNG')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('epochs and batch-size must be positive; workers must be nonnegative')
    if args.generate_only and not CHECKPOINT.is_file():
        parser.error(f'Checkpoint missing: {CHECKPOINT}. Run training first.')

    # Seed training for repeatability. Generate-only deliberately uses fresh noise.
    if not args.generate_only:
        torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}', flush=True)
    generator = Generator().to(device)

    if args.generate_only:
        checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
        if checkpoint['latent_dim'] != LATENT_DIM:
            raise ValueError(f'Checkpoint latent dimension must be {LATENT_DIM}')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f'Loaded generator from epoch {checkpoint["epoch"]}', flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / 'gan_samples.png'
        save_samples(generator, device, path)
        print(f'Saved: {path}', flush=True)
        return  # No dataset, discriminator or optimiser is created in this mode.

    dataset = MRISlices()
    print(f'Selected training MRI images (slices 8-23 inclusive): {len(dataset)}', flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
    discriminator = Discriminator().to(device)
    generator.apply(initialise_weights)
    discriminator.apply(initialise_weights)
    optimiser_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimiser_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    fixed_noise = torch.randn(8, LATENT_DIM, 1, 1, device=device)
    capture_epochs = {1, 10, 20, 30, 40, 50, args.epochs}
    stages, generator_losses, discriminator_losses = [], [], []

    for epoch in range(1, args.epochs + 1):
        loss_g, loss_d = train_epoch(generator, discriminator, loader, device, optimiser_g, optimiser_d)
        if not np.isfinite([loss_g, loss_d]).all():
            raise RuntimeError(f'Non-finite GAN loss at epoch {epoch}; training did not complete.')
        generator_losses.append(loss_g)
        discriminator_losses.append(loss_d)
        # Adversarial losses need not decrease monotonically: both opponents learn.
        print(f'Epoch {epoch:02d}/{args.epochs} | Generator loss: {loss_g:.4f} | '
              f'Discriminator loss: {loss_d:.4f}', flush=True)
        if epoch in capture_epochs:
            generator.eval()
            with torch.no_grad():
                stages.append((epoch, generator(fixed_noise).cpu()))

    torch.save({'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'epoch': args.epochs, 'latent_dim': LATENT_DIM}, CHECKPOINT)
    print(f'Saved: {CHECKPOINT}', flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    save_samples(generator, device, RESULTS / 'gan_samples.png')
    save_progress(stages, RESULTS / 'gan_progress.png')
    save_losses(generator_losses, discriminator_losses, RESULTS / 'gan_losses.png')
    for name in ('gan_samples.png', 'gan_progress.png', 'gan_losses.png'):
        print(f'Saved: {RESULTS / name}', flush=True)
    print('Inspect the sample grid for realism and diversity; losses alone do not rule out mode collapse.',
          flush=True)


if __name__ == '__main__':
    main()
