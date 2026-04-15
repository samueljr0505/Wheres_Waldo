import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_utils import get_loaders
from model import SimpleCNN, get_resnet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0

    for x, y in tqdm(loader):
        x, y = x.to(DEVICE), y.float().unsqueeze(1).to(DEVICE)

        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    return total / len(loader)


def evaluate(model, loader, loss_fn):
    model.eval()
    correct, total, loss_sum = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE).float().unsqueeze(1)

            pred = model(x)
            loss_sum += loss_fn(pred, y).item()

            pred_label = (pred > 0.5).float()
            correct += (pred_label == y).sum().item()
            total += y.size(0)

    return loss_sum / len(loader), correct / total


def main():

    DATA_ROOT = "Hey-Waldo"
    RES = "64"        # 64 / 128 / 256
    VARIANT = ""      # "", "-BW", "-gray"

    train_loader, val_loader, _ = get_loaders(
        DATA_ROOT, RES, VARIANT, limit_per_class=30
    )

    model = get_resnet()   # or SimpleCNN()
    model.to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, opt, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "waldo_model.pth")


if __name__ == "__main__":
    main()