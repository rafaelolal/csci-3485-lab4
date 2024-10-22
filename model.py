"""
Defines the experiments to be run.
"""

import torch.nn as nn
from torch import no_grad, tensor
from torch.optim import Adam

from data import get_data_loaders


class Model:
    def __init__(self, model, weights, classes: int) -> None:
        self.model = None
        self.train_loader = None
        self.test_loader = None

        self.weights = weights
        self.classes = classes
        self.set_model_and_data(model)

    def set_model_and_data(self, model) -> None:
        """Sets the last layer of the model and the data with the correction transformation"""

        self.model = model(weights=self.weights)

        # freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # change last layer
        if hasattr(self.model, "classifier"):
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, self.classes)

        elif hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.classes)

        else:
            raise ValueError("Model architecture not supported")

        # get data loaders with correct transformations
        # TODO: change batch size
        self.train_loader, self.test_loader = get_data_loaders(
            transforms=[self.weights.transforms()], batch_size=32
        )

    def train(
        self,
        device: str,
        max_epochs: int = None,
        lr: float = 1e-3,
        min_delta: float = 1e-2,
    ) -> dict:
        """
        Trains the model following the given parameters.
        """

        self.model.to(device)
        self.model.train()

        optimizer = Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # info to return about the training
        # could include the loss/accuracy per epoch for example
        epochs = 0

        best_loss = float("inf")
        while max_epochs is None or epochs < max_epochs:
            epoch_loss = 0
            # training on each batch
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)

                # avoids error:
                # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
                x = x.requires_grad_(True)

                optimizer.zero_grad()
                outputs = self.model(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epochs += 1
            avg_loss = epoch_loss / len(self.train_loader)

            # stop if the loss has not improved enough
            if abs(best_loss - avg_loss) < min_delta:
                break

            best_loss = min(best_loss, avg_loss)

        return {"epochs": epochs}

    @no_grad()
    def test(self, device: str) -> float:
        """
        Tests the accuracy of the model.
        """

        self.model.to(device)
        self.model.eval()
        # maintain the calculation in the gpu until returning
        correct = tensor(0, device=device)
        total = 0

        for x, y in self.test_loader:
            x, y = x.to(device), y.to(device)
            predicted = self.model(x)
            correct += (predicted.argmax(dim=1) == y).sum()
            total += y.size(0)

        accuracy = (correct / total).item()
        return accuracy

    def get_parameter_count(self) -> int:
        """
        Custom function that returns the number of total and trainable parameters in a model like torchvision's summary function
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        return total_params, trainable_params
