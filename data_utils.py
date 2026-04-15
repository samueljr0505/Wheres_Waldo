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
def load_data(data_dir, limit_per_class=30):
    waldo_dir = os.path.join(data_dir, "waldo")
    not_dir = os.path.join(data_dir, "notwaldo")

    waldo = sorted([
        os.path.join(waldo_dir, f)
        for f in os.listdir(waldo_dir)
        if f.endswith(".jpg")
    ])[:limit_per_class]

    not_waldo = sorted([
        os.path.join(not_dir, f)
        for f in os.listdir(not_dir)
        if f.endswith(".jpg")
    ])[:limit_per_class]

    paths = waldo + not_waldo
    labels = [1] * len(waldo) + [0] * len(not_waldo)

    return paths, labels


# -------------------------
# SPLIT DATA
# -------------------------
def split_data(paths, labels, train=0.7, val=0.15):
    # split by class first
    waldo = [(p, l) for p, l in zip(paths, labels) if l == 1]
    not_waldo = [(p, l) for p, l in zip(paths, labels) if l == 0]

    random.shuffle(waldo)
    random.shuffle(not_waldo)

    def split_class(data):
        n = len(data)
        t_end = int(n * train)
        v_end = int(n * (train + val))

        train_part = data[:t_end]
        val_part = data[t_end:v_end]
        test_part = data[v_end:]

        return train_part, val_part, test_part

    w_train, w_val, w_test = split_class(waldo)
    n_train, n_val, n_test = split_class(not_waldo)

    train = w_train + n_train
    val = w_val + n_val
    test = w_test + n_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # ---- PRINT DEBUG INFO (THIS IS WHAT YOU WANTED) ----
    def count_labels(data):
        c0 = sum(1 for _, y in data if y == 0)
        c1 = sum(1 for _, y in data if y == 1)
        return c0, c1

    print("\n===== SPLIT CHECK =====")
    tr0, tr1 = count_labels(train)
    va0, va1 = count_labels(val)
    te0, te1 = count_labels(test)

    print(f"TRAIN: {tr1} waldo / {tr0} notwaldo  (total {len(train)})")
    print(f"VAL:   {va1} waldo / {va0} notwaldo  (total {len(val)})")
    print(f"TEST:  {te1} waldo / {te0} notwaldo  (total {len(test)})")
    print("=======================\n")

    return (
        tuple(zip(*train)),
        tuple(zip(*val)),
        tuple(zip(*test))
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
def get_loaders(data_root, resolution="64", variant="", batch_size=32, limit_per_class=None):

    data_dir = os.path.join(data_root, f"{resolution}{variant}")

    paths, labels = load_data(data_dir, limit_per_class)

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