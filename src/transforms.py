from __future__ import annotations
import random
import torch
from PIL import Image
from torchvision import transforms


def to_rgb(img: Image.Image) -> Image.Image:
    #so that it's picklable on Windows DataLoader workers
    return img.convert("RGB")

def build_train_transforms(image_size: int, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    return transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize(image_size + 32),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

def build_eval_transforms(image_size: int):
    return transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])