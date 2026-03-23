import torch
from torchvision import transforms

class AddGaussianNoise:
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)

def build_default_transforms(img_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, saturation=0.15, contrast=0.05, hue=0.02),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.02),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, test_transform
