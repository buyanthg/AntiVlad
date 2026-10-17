from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        state, policy, value = self.data[index]

        return state, policy, value