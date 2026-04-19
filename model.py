import torch.nn as nn
from torchvision import models

# -------------------------
# RESNET (TRANSFER LEARNING)
# -------------------------
def get_resnet():
    model = models.resnet18(weights=None)

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )

    return model