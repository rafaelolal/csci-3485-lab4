"""
Defines the experiments to be run.
"""

from datetime import datetime
from time import time

from torch import cuda, save
from torch.backends import mps
from torchsummary import summary
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    VGG11_Weights,
    VGG13_Weights,
    resnet18,
    resnet34,
    vgg11,
    vgg13,
)

from data import append_to_file, classify_images
from model import Model

if cuda.is_available():
    device = "cuda"
elif mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


def time_it(f):
    """
    Times the execution of a function.
    """

    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        return (time() - start, result)

    return wrapper


MODELS = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "vgg11": vgg11,
    "vgg13": vgg13,
}

WEIGHTS = {
    "resnet18": ResNet18_Weights,
    "resnet34": ResNet34_Weights,
    "vgg11": VGG11_Weights,
    "vgg13": VGG13_Weights,
}

for name in MODELS:
    print("my name", name)

    model = Model(
        MODELS[name],
        WEIGHTS[name].IMAGENET1K_V1,
        classes=10,
        batch_size=32,
        data_size=-1,
    )
    training_time, info = time_it(model.train)(
        device=device, max_epochs=5, min_delta=0, lr=1e-3
    )

    # save(model.model, f"./models/{name}.pth")

    accuracy = model.test(device=device)

    file_name = "results.txt"
    append_to_file(file_name, datetime.now())
    append_to_file(file_name, f"{name}")
    append_to_file(file_name, f"param count: {model.get_parameter_count()}")
    append_to_file(file_name, f"time: {training_time}")
    append_to_file(file_name, f"info: {info}")
    append_to_file(file_name, f"accuracy: {accuracy}")

    classify_images(
        model.model,
        WEIGHTS[name].IMAGENET1K_V1.transforms(),
        device,
        "./images",
    )
