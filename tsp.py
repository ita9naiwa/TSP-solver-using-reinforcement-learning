import torch
from torch.utils.data import DataLoader, Dataset


class TSPDataset(Dataset):

    def __init__(self, num_nodes, num_samples, random_seed=111):
        super(TSPDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.data_set = []
        for l in range(num_samples):
            x = torch.FloatTensor(num_nodes, 2).uniform_(0, 1)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx, self.data_set[idx]


def test():
    train_loader = DataLoader(TSPDataset(10, 100), batch_size=32, shuffle=True, num_workers=1)
    for  (a, b) in train_loader:
        print(a, b)

if __name__ == "__main__":
    test()
