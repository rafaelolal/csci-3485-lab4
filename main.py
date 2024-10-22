"""
Defines the experiments to be run.
"""

from datetime import datetime
from time import time

from torch import cuda
from torch.backends import mps
from torchsummary import summary
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    VGG11_BN_Weights,
    VGG13_BN_Weights,
    resnet18,
    resnet34,
    vgg11_bn,
    vgg13_bn,
)

from data import append_to_file
from model import Model

if mps.is_available():
    device = "mps"
elif cuda.is_available():
    device = "cuda"
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
    "vgg11_bn": vgg11_bn,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "vgg13_bn": vgg13_bn,
}

WEIGHTS = {
    "resnet18": ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": ResNet34_Weights.IMAGENET1K_V1,
    "vgg11_bn": VGG11_BN_Weights.IMAGENET1K_V1,
    "vgg13_bn": VGG13_BN_Weights.IMAGENET1K_V1,
}

for name in MODELS:
    print("my name", name)

    model = Model(MODELS[name], WEIGHTS[name], 10)
    # TODO: change the number of epochs below
    # TODO: change the learning rate
    training_time, info = time_it(model.train)(
        device=device, max_epochs=20, min_delta=0, lr=1e-5
    )
    accuracy = model.test(device=device)

    file_name = "results.txt"
    append_to_file(file_name, datetime.now())
    append_to_file(file_name, f"{name}")
    append_to_file(file_name, f"param count: {model.get_parameter_count()}")
    append_to_file(file_name, f"time: {training_time}")
    append_to_file(file_name, f"info: {info}")
    append_to_file(file_name, f"accuracy: {accuracy}")
