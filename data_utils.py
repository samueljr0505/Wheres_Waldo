import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# -------------------------
# DATASET CLASS
# -------------------------
class WaldoDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# -------------------------
# LOAD ALL IMAGES
# -------------------------
def load_data(data_dir):
    waldo_dir = os.path.join(data_dir, "waldo")
    not_dir = os.path.join(data_dir, "notwaldo")

    waldo = [os.path.join(waldo_dir, f) for f in os.listdir(waldo_dir) if f.endswith(".jpg")]
    not_waldo = [os.path.join(not_dir, f) for f in os.listdir(not_dir) if f.endswith(".jpg")]

    paths = waldo + not_waldo
    labels = [1] * len(waldo) + [0] * len(not_waldo)

    return paths, labels


# -------------------------
# SPLIT DATA
# -------------------------
def split_data(paths, labels, train=0.7, val=0.15):
    data = list(zip(paths, labels))
    random.shuffle(data)

    paths, labels = zip(*data)

    n = len(paths)
    t_end = int(n * train)
    v_end = int(n * (train + val))

    return (
        (paths[:t_end], labels[:t_end]),
        (paths[t_end:v_end], labels[t_end:v_end]),
        (paths[v_end:], labels[v_end:])
    )


# -------------------------
# TRANSFORMS
# -------------------------
def get_transforms(size):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    return train_tf, test_tf


# -------------------------
# MAIN LOADER
# -------------------------
def get_loaders(data_root, resolution="64", variant="", batch_size=32):

    data_dir = os.path.join(data_root, f"{resolution}{variant}")

    paths, labels = load_data(data_dir)

    train, val, test = split_data(paths, labels)

    train_tf, test_tf = get_transforms(int(resolution))

    train_set = WaldoDataset(*train, transform=train_tf)
    val_set = WaldoDataset(*val, transform=test_tf)
    test_set = WaldoDataset(*test, transform=test_tf)

    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )