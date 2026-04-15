import torch
import random
import numpy as np
from sklearn.metrics import classification_report

from data_utils import get_loaders
from model import get_resnet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(42)

    train_loader, val_loader, test_loader = get_loaders(
        "Hey-Waldo", "64", "", limit_per_class=30
    )

    model = get_resnet()
    model.load_state_dict(torch.load("waldo_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)

            pred = model(x)  # add sigmoid here if needed
            pred = (pred > 0.5).cpu().numpy().astype(int)

            y_pred.extend(pred.flatten())
            y_true.extend(y.numpy())

    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()