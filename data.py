"""
Manages the data for the experiments.
"""

from torch import Tensor, tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor


class CIFAR10Dataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None) -> None:
        dataset = CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
        # TODO: remove below limits, only for testing
        self.data = tensor(dataset.data).permute(0, 3, 1, 2).float() / 255.0
        self.labels = tensor(dataset.targets)

    def __getitem__(self, i) -> tuple[Tensor, Tensor]:
        image = self.data[i]
        label = self.labels[i].clone().detach()
        return image, label

    def __len__(self) -> int:
        return len(self.data)


def get_data_loaders(
    root: str = "./data/CIFAR10",
    transforms: list = [],
    batch_size: int = 32,
) -> tuple[DataLoader]:
    test_dataset = CIFAR10Dataset(root=root, train=False, transform=transforms)
    train_dataset = CIFAR10Dataset(root=root, train=True, transform=transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


def append_to_file(filename: str, data: any) -> None:
    with open(filename, "a") as file:
        file.write(str(data) + "\n")
