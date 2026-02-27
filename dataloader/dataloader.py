
import torch
from torch.utils.data import DataLoader, Dataset
import os

class Load_Dataset(Dataset):
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()
        self.training_mode = dataset.get("training_mode", "supervised")
        X = dataset["samples"]
        y = dataset["labels"]
        # Ensure [N, 1, 200] format
        if len(X.shape) < 3: X = X.unsqueeze(2)
        if X.shape[2] == 1: X = X.permute(0, 2, 1)

        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return len(self.x_data)

def data_generator(data_path, configs, training_mode, subset=True):
    # data_path 传进来是 'vd' 或 'vd_p2'
    # 映射到 data/vd ...

    if os.path.exists(data_path):
        root = data_path
    else:
        root = os.path.join("data", data_path)

    train = torch.load(os.path.join(root, "train.pt"))
    val = torch.load(os.path.join(root, "val.pt"))
    test = torch.load(os.path.join(root, "test.pt"))

    train_ds = Load_Dataset(train)
    val_ds = Load_Dataset(val)
    test_ds = Load_Dataset(test)

    train_dl = DataLoader(train_ds, batch_size=configs.batch_size, shuffle=True, drop_last=configs.drop_last)
    val_dl = DataLoader(val_ds, batch_size=configs.batch_size, shuffle=False, drop_last=configs.drop_last)
    test_dl = DataLoader(test_ds, batch_size=configs.batch_size, shuffle=False, drop_last=False)

    return train_dl, val_dl, test_dl
