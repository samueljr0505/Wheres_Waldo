import torch
import matplotlib.pyplot as plt

from data_utils import get_loaders
from model import get_resnet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    _, _, test_loader = get_loaders("Hey-Waldo", "64", "")

    model = get_resnet()
    model.load_state_dict(torch.load("waldo_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    x, y = next(iter(test_loader))
    x = x.to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        pred = (pred > 0.5).cpu()

    plt.figure(figsize=(10, 5))

    for i in range(8):
        plt.subplot(2, 4, i+1)
        img = x[i].cpu().permute(1, 2, 0)
        plt.imshow(img)
        plt.title(f"P:{int(pred[i])} T:{y[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/sample_predictions.png")
    plt.show()


if __name__ == "__main__":
    main()