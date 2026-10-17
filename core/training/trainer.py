import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.training.dataset import ChessDataset
from core.constants import (CHECKPOINT_DIR, DEVICE)


class Trainer:
    def __init__(self, model, device=DEVICE, learning_rate=3e-4,
                 weight_decay=1e-4, batch_size=256):
        self.device = device

        self.model = model.to(self.device)
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train(self, data, epochs=20):
        if not data:
            raise ValueError("Training data is empty.")

        dataset = ChessDataset(data)
        loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        history = []

        for epoch in range(epochs):
            self.model.train()

            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_samples = 0

            for states, policies, values in loader:
                states = states.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)

                self.optimizer.zero_grad()

                policy_logits, predicted_values = self.model(states)

                policy_loss = -(policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()

                value_loss = F.mse_loss(predicted_values.squeeze(1), values)

                loss = policy_loss + value_loss

                loss.backward()
                self.optimizer.step()

                batch_size = states.size(0)
                total_samples += batch_size

                total_loss += loss.item() * batch_size
                total_policy_loss += policy_loss.item() * batch_size
                total_value_loss += value_loss.item() * batch_size

            metrics = {
                "loss": total_loss / total_samples,
                "policy_loss": total_policy_loss / total_samples,
                "value_loss": total_value_loss / total_samples,
            }
            history.append(metrics)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss: {metrics['loss']:.4f} | "
                f"policy: {metrics['policy_loss']:.4f} | "
                f"value: {metrics['value_loss']:.4f}",
                flush=True,
            )

        return history

    def save_checkpoint(self, path, iteration, history=None):
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": history,
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device,)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["iteration"], checkpoint.get("history")
