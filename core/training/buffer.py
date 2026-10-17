from collections import deque
from pathlib import Path
import torch

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)

    def add_game(self, game_data):
        if game_data is None:
            return

        self.data.extend(game_data)

    def sample_all(self):
        return list(self.data)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"capacity": self.capacity, "data": list(self.data)}, path)

    def load(self, path):
        path = Path(path)

        if not path.exists():
            return False

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        saved_data = checkpoint["data"]

        self.data = deque(saved_data[-self.capacity:], maxlen=self.capacity)

        return True

    def __len__(self):
        return len(self.data)

