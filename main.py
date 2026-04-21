import torch
import chess
import random
import os
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Импортируем твои модули
from model import AntiVlad
from mcts import mcts_search
from utils import get_device, board_to_tensor, move_to_idx

# --- Конфигурация ---
CONFIG = {
    "save_dir": "checkpoints",
    "log_dir": "logs",
    "lr": 0.001,
    "batch_size": 64,
    "memory_size": 30000,
    "mcts_iterations": 60,
    "save_every": 10,
    "device": str(get_device())
}

# Инициализация
os.makedirs(CONFIG["save_dir"], exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(os.path.join(CONFIG["log_dir"], run_id))

device = torch.device(CONFIG["device"])
model = AntiVlad().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
memory = []


def train_step(game_num):
    if len(memory) < CONFIG["batch_size"]:
        return None

    batch = random.sample(memory, CONFIG["batch_size"])
    states, target_p, target_v = zip(*batch)

    # Подготовка тензоров
    s_tensor = torch.cat(states).float().to(device)
    tp_tensor = torch.stack(target_p).float().to(device)
    tv_tensor = torch.tensor(target_v).unsqueeze(1).float().to(device)

    model.train()
    p_out, v_out = model(s_tensor)

    # Считаем лоссы
    loss_v = F.mse_loss(v_out, tv_tensor)
    loss_p = F.cross_entropy(p_out, tp_tensor)
    total_loss = loss_v + loss_p

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Логируем метрики обучения
    writer.add_scalar("Loss/Total", total_loss.item(), game_num)
    writer.add_scalar("Loss/Value", loss_v.item(), game_num)
    writer.add_scalar("Loss/Policy", loss_p.item(), game_num)

    return total_loss.item()


def self_play(game_num):
    board = chess.Board()
    history = []

    while not board.is_game_over():
        # MCTS поиск
        root = mcts_search(board, model, iterations=CONFIG["mcts_iterations"])

        # Создаем таргет для Policy (распределение визитов)
        p_target = torch.zeros(4096)
        for m, child in root.children.items():
            p_target[move_to_idx(m)] = child.visits / root.visits

        # Сохраняем состояние (переносим на CPU для экономии VRAM)
        history.append((board_to_tensor(board).cpu(), p_target))

        # Выбор хода: жадный для качества, или с долей случайности для обучения
        if len(board.move_stack) < 12:
            # Exploration: выбираем пропорционально визитам
            moves = list(root.children.values())
            weights = [n.visits for n in moves]
            move = random.choices(moves, weights=weights)[0].move
        else:
            move = max(root.children.values(), key=lambda n: n.visits).move

        board.push(move)

    # Финальный результат
    res_str = board.result()
    v_result = 1 if res_str == "1-0" else -1 if res_str == "0-1" else 0

    # Добавляем в память с инверсией value для каждого хода
    for s, pt in history:
        memory.append((s, pt, v_result))
        v_result = -v_result

    # Логируем данные игры
    writer.add_scalar("Game/Moves", len(history), game_num)
    return res_str


# --- Главный цикл ---
print(f"Запуск AntiVlad. Логи: {os.path.join(CONFIG['log_dir'], run_id)}")

try:
    game_idx = 0
    while True:
        game_idx += 1
        res = self_play(game_idx)

        # Шаг обучения
        loss = train_step(game_idx)

        status = f"Game {game_idx} | Res: {res} | Loss: {f'{loss:.4f}' if loss else 'Waiting...'}"
        print(status)

        # Управление памятью
        if len(memory) > CONFIG["memory_size"]:
            memory = memory[-CONFIG["memory_size"]:]

        # Сохранение чекпоинта
        if game_idx % CONFIG["save_every"] == 0:
            checkpoint_path = os.path.join(CONFIG["save_dir"], "vlad_latest.pt")
            torch.save({
                'game_idx': game_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG
            }, checkpoint_path)
            print(f"--- Модель сохранена ({game_idx}) ---")

except KeyboardInterrupt:
    print("\nОстановка обучения. Дампим логи...")
    writer.close()