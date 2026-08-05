"""Train a boundary-oriented DeepLabV3+ comparison on the 68 labelled photographs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class SherdDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], training: bool):
        self.pairs = pairs
        transforms = [A.Resize(640, 640)]
        if training:
            transforms += [
                A.HorizontalFlip(p=.5), A.VerticalFlip(p=.2),
                A.Affine(scale=(.9, 1.1), translate_percent=(-.06, .06), rotate=(-12, 12), p=.7),
                A.RandomBrightnessContrast(.15, .15, p=.35),
                A.GaussNoise(std_range=(.01, .04), p=.15),
            ]
        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(np.float32)
        item = self.transform(image=image, mask=mask)
        image_tensor = torch.from_numpy(np.array(item["image"], copy=True)).permute(2, 0, 1).float().div(255)
        mask_tensor = torch.from_numpy(np.array(item["mask"], copy=True)).float().unsqueeze(0)
        return image_tensor, mask_tensor


def dice_score(logits: torch.Tensor, truth: torch.Tensor, threshold: float = .5) -> float:
    prediction = torch.sigmoid(logits) >= threshold
    intersection = (prediction & truth.bool()).sum((1, 2, 3)).float()
    total = prediction.sum((1, 2, 3)).float() + truth.sum((1, 2, 3)).float()
    return float(((2 * intersection + 1) / (total + 1)).mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--dataset", type=Path, default=root.parent / "DL ArchProject" / "dataset_clean" / "train")
    parser.add_argument("--output", type=Path, default=root / "models" / "deeplabv3plus_pottery_best.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    images = {path.stem: path for path in (args.dataset / "images").glob("*.png")}
    masks = {path.stem: path for path in (args.dataset / "masks").glob("*.png")}
    keys = sorted(images.keys() & masks.keys())
    if len(keys) != 68:
        raise RuntimeError(f"Expected 68 paired images and masks, found {len(keys)}")
    train_keys, validation_keys = train_test_split(keys, test_size=.2, random_state=args.seed)
    train_pairs = [(images[key], masks[key]) for key in train_keys]
    validation_pairs = [(images[key], masks[key]) for key in validation_keys]
    train_loader = DataLoader(SherdDataset(train_pairs, True), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(SherdDataset(validation_pairs, False), batch_size=args.batch_size,
                                   shuffle=False, num_workers=0, pin_memory=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.DeepLabV3Plus("resnet50", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    bce = torch.nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, stale, history = -1.0, 0, []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss = 0.0
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(image); loss = .5 * bce(logits, mask) + .5 * dice_loss(logits, mask)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_loss += float(loss.item())
        model.eval(); val_loss = 0.0; validation_logits = []; validation_masks = []
        with torch.inference_mode():
            for image, mask in validation_loader:
                image, mask = image.to(device), mask.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(image); loss = .5 * bce(logits, mask) + .5 * dice_loss(logits, mask)
                val_loss += float(loss.item()); validation_logits.append(logits.float().cpu()); validation_masks.append(mask.cpu())
        logits = torch.cat(validation_logits); truth = torch.cat(validation_masks)
        candidates = np.arange(.35, .91, .05)
        scores = [(float(threshold), dice_score(logits, truth, float(threshold))) for threshold in candidates]
        threshold, validation_dice = max(scores, key=lambda item: item[1])
        record = {"epoch": epoch, "train_loss": train_loss / len(train_loader),
                  "validation_loss": val_loss / len(validation_loader),
                  "validation_dice": validation_dice, "threshold": threshold}
        history.append(record)
        print(f"epoch {epoch:02d}: train={record['train_loss']:.4f} val={record['validation_loss']:.4f} "
              f"dice={validation_dice:.4f} threshold={threshold:.2f}", flush=True)
        if validation_dice > best + 1e-4:
            best, stale = validation_dice, 0
            torch.save({"model_state": model.state_dict(), "threshold": threshold, "epoch": epoch,
                        "validation_dice": validation_dice, "seed": args.seed,
                        "train_images": train_keys, "validation_images": validation_keys}, args.output)
        else:
            stale += 1
            if stale >= 7:
                print("Early stopping after seven validation checks without improvement.")
                break
    args.output.with_suffix(".json").write_text(json.dumps({"best_validation_dice": best, "history": history}, indent=2), encoding="utf-8")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
