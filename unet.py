"""COMP3710 Part 4 Task 2: categorical OASIS segmentation with a manual UNet.

Train: python -B unet.py
Demonstrate saved-model inference: python -B unet.py --eval-only
Classes 0-3 correspond to mask intensities 0, 85, 170, 255; no anatomical
names are assumed. Dataset files are only read, never changed.
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
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT = SCRIPT_DIR / 'unet_best.pt'
FIGURE = SCRIPT_DIR / 'results' / 'unet_segmentation.png'
NUM_CLASSES = 4
MASK_VALUES = np.array([0, 85, 170, 255], dtype=np.uint8)
# One consistent categorical colour per class in both ground truth and predictions.
COLOURS = np.array([[0, 0, 0], [0, 160, 230], [240, 160, 0], [210, 60, 170]],
                   dtype=np.uint8)


class OASISSegmentation(Dataset):
    def __init__(self, split):
        image_dir = DATA_ROOT / f'keras_png_slices_{split}'
        mask_dir = DATA_ROOT / f'keras_png_slices_seg_{split}'
        self.images = sorted(image_dir.glob('case_*.nii.png'))
        if not self.images:
            raise FileNotFoundError(f'No MRI PNG slices found in {image_dir}')
        self.masks = [mask_dir / p.name.replace('case_', 'seg_', 1) for p in self.images]
        expected = {p.name for p in self.masks}
        found = {p.name for p in mask_dir.glob('seg_*.nii.png')}
        if expected != found:
            raise ValueError(f'MRI/mask pairing mismatch in {split}: '
                             f'missing masks={sorted(expected - found)[:3]}, '
                             f'unmatched masks={sorted(found - expected)[:3]}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as image:
            pixels = np.array(image.convert('L'), dtype=np.float32)
        with Image.open(self.masks[index]) as mask_image:
            mask = np.array(mask_image)
        if pixels.shape != (256, 256) or mask.shape != (256, 256):
            raise ValueError(f'Expected 256x256 MRI/mask: {self.images[index]}, '
                             f'got {pixels.shape} and {mask.shape}')
        if mask.dtype != np.uint8 or not np.isin(mask, MASK_VALUES).all():
            raise ValueError(f'Unexpected mask values in {self.masks[index]}: '
                             f'dtype={mask.dtype}, values={np.unique(mask).tolist()}; '
                             'expected uint8 values 0, 85, 170, 255')
        image_tensor = torch.from_numpy(pixels).unsqueeze(0) / 255.0
        # Safe only AFTER validation: 0->0, 85->1, 170->2, 255->3.
        target = torch.from_numpy((mask // 85).astype(np.int64))
        return image_tensor, target


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.convolutions = DoubleConv(2 * out_channels, out_channels)

    def forward(self, x, skip):
        # Concatenate fine spatial features from the matching encoder level.
        # They help recover boundaries lost during downsampling.
        return self.convolutions(torch.cat([self.up(x), skip], dim=1))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up4 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up1 = UpBlock(64, 32)
        self.output = nn.Conv2d(32, NUM_CLASSES, 1)

    def forward(self, images):
        # Encoder resolutions: 256, 128, 64, 32; bottleneck resolution: 16.
        e1 = self.enc1(images)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.up4(x, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        # Raw categorical logits [B, 4, 256, 256], not softmax probabilities.
        return self.output(x)


def segmentation_loss(logits, targets):
    # Compute loss in float32 even when convolutions use mixed precision.
    logits = logits.float()
    cross_entropy = F.cross_entropy(logits, targets)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(targets, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dimensions = (0, 2, 3)  # Sum over batch and pixels, retaining four classes.
    intersection = (probabilities * one_hot).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + one_hot.sum(dim=dimensions)
    soft_dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    # Equal weight for every class, including classes with fewer pixels.
    return cross_entropy + (1 - soft_dice.mean())


def dice_from_confusion(confusion):
    # Rows are true classes; columns are predicted classes.
    intersection = confusion.diag().double()
    target_count = confusion.sum(dim=1).double()
    prediction_count = confusion.sum(dim=0).double()
    denominator = target_count + prediction_count
    # An absent class has undefined DSC, not an automatic perfect score.
    return torch.where(denominator > 0, 2 * intersection / denominator,
                       torch.full_like(denominator, float('nan')))


def train_epoch(model, loader, device, optimiser, scaler):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(images)
        loss = segmentation_loss(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits = model(images)
        total_loss += segmentation_loss(logits, targets).item() * images.size(0)
        predictions = logits.argmax(dim=1)
        pairs = NUM_CLASSES * targets.reshape(-1) + predictions.reshape(-1)
        confusion += torch.bincount(pairs, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    # Accumulate counts over the ENTIRE split before computing each class DSC.
    return total_loss / len(loader.dataset), dice_from_confusion(confusion).cpu()


@torch.no_grad()
def save_segmentation(model, dataset, device, path):
    model.eval()
    # Show central slices from four different test cases, chosen without scores.
    cases = {}
    for i, filename in enumerate(dataset.images):
        cases.setdefault(filename.name.split('_')[1], []).append(i)
    case_ids = sorted(cases)
    indices = []
    for position in np.linspace(0, len(case_ids) - 1, min(4, len(case_ids)), dtype=int):
        members = sorted(cases[case_ids[position]],
                         key=lambda i: int(dataset.images[i].name.split('_')[-1].split('.')[0]))
        indices.append(members[len(members) // 2])
    canvas = Image.new('RGB', (130 + 256 * len(indices), 874), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((130, 8), 'OASIS test segmentation: four categorical classes', fill='black')
    for c, colour in enumerate(COLOURS):
        x = 130 + c * 180
        draw.rectangle((x, 29, x + 18, 47), fill=tuple(colour))
        draw.text((x + 24, 32), f'class {c}', fill='black')
    for label, y in [('MRI', 190), ('Ground truth', 456), ('Prediction', 722)]:
        draw.text((8, y), label, fill='black')
    for column, index in enumerate(indices):
        image, target = dataset[index]
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0].cpu()
        x = 130 + column * 256
        draw.text((x + 4, 58), dataset.images[index].name, fill='black')
        mri = Image.fromarray((image[0].numpy() * 255).round().astype(np.uint8)).convert('RGB')
        canvas.paste(mri, (x, 76))
        canvas.paste(Image.fromarray(COLOURS[target.numpy()]), (x, 342))
        canvas.paste(Image.fromarray(COLOURS[prediction.numpy()]), (x, 608))
    canvas.save(path, format='PNG')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('epochs and batch-size must be positive; workers must be nonnegative')
    if args.eval_only and not CHECKPOINT.is_file():
        parser.error(f'Checkpoint missing: {CHECKPOINT}. Run training first.')

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(device)}', flush=True)
    model = UNet().to(device)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=device.type == 'cuda')

    if not args.eval_only:
        train_data = OASISSegmentation('train')
        validation_data = OASISSegmentation('validate')
        print(f'MRI/mask pairs: train={len(train_data)}, validation={len(validation_data)}', flush=True)
        train_loader = DataLoader(train_data, shuffle=True, **loader_options)
        validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        best_minimum = -float('inf')
        for epoch in range(1, args.epochs + 1):
            training_loss = train_epoch(model, train_loader, device, optimiser, scaler)
            validation_loss, dsc = evaluate(model, validation_loader, device)
            if not torch.isfinite(dsc).all():
                raise ValueError('Undefined validation DSC: a class has no true or predicted pixels.')
            scores = ' | '.join(f'Validation DSC class {c}: {dsc[c]:.4f}' for c in range(NUM_CLASSES))
            print(f'Epoch {epoch:02d}/{args.epochs} | Training loss: {training_loss:.4f} | '
                  f'Validation loss: {validation_loss:.4f} | {scores} | '
                  f'Validation mean DSC: {dsc.mean():.4f}', flush=True)
            minimum = dsc.min().item()
            if minimum > best_minimum:
                best_minimum = minimum
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                            'validation_dsc': dsc.tolist()}, CHECKPOINT)
                print(f'Saved best checkpoint: epoch {epoch}, minimum validation DSC {minimum:.4f}', flush=True)

    # Both paths load the selected checkpoint. Eval-only never creates an optimiser
    # or accesses the training/validation datasets.
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint: epoch {checkpoint["epoch"]}, '
          f'validation DSC={checkpoint["validation_dsc"]}', flush=True)
    test_data = OASISSegmentation('test')
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)
    print(f'Test MRI/mask pairs: {len(test_data)}', flush=True)
    _, test_dsc = evaluate(model, test_loader, device)
    for c in range(NUM_CLASSES):
        print(f'Test DSC class {c}: {test_dsc[c]:.6f}', flush=True)
    print(f'Test mean DSC: {test_dsc.mean():.6f}', flush=True)
    print(f'Minimum test DSC: {test_dsc.min():.6f}', flush=True)
    passed = bool(torch.isfinite(test_dsc).all() and (test_dsc > 0.90).all())
    print(f'All four test DSC values > 0.90: {"YES" if passed else "NO"}', flush=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    save_segmentation(model, test_data, device, FIGURE)
    print(f'Saved: {FIGURE}', flush=True)


if __name__ == '__main__':
    main()
