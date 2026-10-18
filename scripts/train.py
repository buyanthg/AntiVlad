from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from core.constants import CHECKPOINT_DIR, DEVICE
from core.network.antivlad_net import get_standart_model
from core.training.buffer import ReplayBuffer
from core.training.self_play import SelfPlay
from core.training.trainer import Trainer


# Настройки эксперимента
NUM_ITERATIONS = 5
NUM_GAMES = 100
SIMULATIONS = 800
EPOCHS = 10
BUFFER_CAPACITY = 100_000

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_PATH = PROJECT_ROOT / CHECKPOINT_DIR / "latest.pt"
BUFFER_PATH = DATA_DIR / "replay_buffer.pt"
SAVE_EVERY = 10

def main():
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = get_standart_model().to(DEVICE)
    trainer = Trainer(model, device=DEVICE)

    buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

    start_iteration = 0
    history = []

    # Восстанавливаем модель и оптимизатор
    if CHECKPOINT_PATH.exists():
        start_iteration, history = trainer.load_checkpoint(
            CHECKPOINT_PATH
        )
        history = history or []
        print(f"Checkpoint loaded. Completed iterations: {start_iteration}")

    # Восстанавливаем replay buffer
    if buffer.load(BUFFER_PATH):
        print(f"Replay buffer loaded: {len(buffer)} positions")
    else:
        print("Replay buffer not found. Starting with an empty buffer.")

    self_play = SelfPlay(
        model,
        simulations=SIMULATIONS,
    )

    for iteration in range(start_iteration + 1, start_iteration + NUM_ITERATIONS + 1):
        print(f"\n{'=' * 20} Iteration {iteration} {'=' * 20}")

        iteration_start = perf_counter()
        new_positions = 0
        completed_games = 0

        # 1. Генерируем партии
        for _ in tqdm(range(1, NUM_GAMES + 1), desc="Self-play"):
            game_data = self_play.play_game()

            # None означает незавершённую партию
            if game_data is None:
                print("Партия слишком длинная")
                continue

            buffer.add_game(game_data)
            new_positions += len(game_data)
            completed_games += 1
            print(f"Длина партии: {len(game_data)}")

        print(
            f"Completed games: {completed_games}/{NUM_GAMES} | "
            f"New positions: {new_positions} | "
            f"Buffer size: {len(buffer)}"
        )

        if len(buffer) == 0:
            raise RuntimeError("Replay buffer is empty. No training data available.")

        # 2. Обучаем модель на накопленных данных
        training_history = trainer.train(buffer.sample_all(), epochs=EPOCHS)

        history.append({"iteration": iteration, "games": completed_games, "new_positions": new_positions,
                        "buffer_size": len(buffer),"epochs": training_history,
        })

        # 3. Сохраняем состояние для возобновления обучения
        buffer.save(BUFFER_PATH)
        trainer.save_checkpoint(CHECKPOINT_PATH, iteration=iteration, history=history)

        if iteration % SAVE_EVERY == 0:
            snapshot_path = CHECKPOINT_PATH.parent / f"iteration_{iteration}.pt"
            trainer.save_checkpoint(snapshot_path, iteration=iteration, history=history)
        elapsed = perf_counter() - iteration_start

        print(
            f"Iteration {iteration} completed in "
            f"{elapsed / 60:.1f} min"
        )
        print(f"Checkpoint: {CHECKPOINT_PATH}")
        print(f"Replay buffer: {BUFFER_PATH}")


if __name__ == "__main__":
    main()
