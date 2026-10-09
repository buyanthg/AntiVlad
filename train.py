import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import time
import os

from dataset import ChessDataset
from self_play import SelfPlay
from antivlad_net import AntiVladNet
from constants import (
    CHECKPOINT_DIR,
    NETWORK_CHANNELS,
    POLICY_CHANNELS,
    VALUE_CHANNELS,
    VALUE_HIDDEN,
    NUM_RESIDUAL_BLOCKS,
)


class Trainer:
    def __init__(self, model, learning_rate=3e-4, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    def train(self, data, epochs=1):
        dataset = ChessDataset(data)

        loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        self.model.train()

        history = []

        for epoch in range(epochs):
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0

            for states, policies, values in loader:
                states = states.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)

                self.optimizer.zero_grad()

                policy_logits, predicted_values = self.model(states)

                policy_loss = - (policies * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(predicted_values.squeeze(1), values)

                loss = policy_loss + value_loss

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

            num_batches = len(loader)
            metrics = {
                "loss": total_loss / num_batches,
                "policy_loss": total_policy_loss / num_batches,
                "value_loss": total_value_loss / num_batches,
            }
            history.append(metrics)
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss: {metrics['loss']:.4f} | "
                f"policy: {metrics['policy_loss']:.4f} | "
                f"value: {metrics['value_loss']:.4f}"
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
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint["iteration"], checkpoint.get("history")


model = AntiVladNet(
    NETWORK_CHANNELS,
    POLICY_CHANNELS,
    VALUE_CHANNELS,
    VALUE_HIDDEN,
    NUM_RESIDUAL_BLOCKS,
)

num_games = 10
num_iterations = 10

trainer = Trainer(model)
self_play = SelfPlay(model)

start_iteration = 0

latest_path = os.path.join(CHECKPOINT_DIR, 'latest.pt')

if os.path.exists(latest_path):
    start_iteration, _ = trainer.load_checkpoint(latest_path)
    print(f"Loaded checkpoint from iteration {start_iteration}")

total_iterations = start_iteration + num_iterations
for iteration in tqdm(range(start_iteration, total_iterations), desc="Iterations"):
    data = []

    for game in tqdm(range(num_games), desc="Self-play", leave=False):
        game_data = self_play.play_game()
        data.extend(game_data)

    iteration_number = iteration + 1
    print(
        f"Iteration {iteration_number}/{total_iterations} | "
        f"games: {num_games} | "
        f"positions: {len(data)}"
    )

    history = trainer.train(data)

    trainer.save_checkpoint(latest_path, iteration_number, history)

    if iteration_number % 10 == 0:
        trainer.save_checkpoint(os.path.join(CHECKPOINT_DIR, f"iteration_{iteration_number:04d}.pt"),
                                iteration_number,
                                history
        )
