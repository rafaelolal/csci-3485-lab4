"""
Manages the data for the experiments.
"""

from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: list = [],
        size: int = -1,
    ) -> None:
        self.dataset = CIFAR10(
            root=root, train=train, download=True, transform=Compose(transform)
        )

        if size != -1:
            indices = range(300)
            self.dataset = Subset(self.dataset, indices)

    def __getitem__(self, i) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[i]
        return image, label

    def __len__(self) -> int:
        return len(self.dataset)


def get_data_loaders(
    root: str = "./data/CIFAR10",
    transforms: list = [],
    batch_size: int = 32,
    size: int = -1,
) -> tuple[DataLoader]:
    test_dataset = CIFAR10Dataset(
        root=root, train=False, transform=transforms, size=size
    )
    train_dataset = CIFAR10Dataset(
        root=root, train=True, transform=transforms, size=size
    )

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
