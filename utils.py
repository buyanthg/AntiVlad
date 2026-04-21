import chess
import numpy as np
import torch


def board_to_tensor(board):
    device = get_device()
    # Создаем тензор сразу на GPU
    tensor = torch.zeros((13, 8, 8), dtype=torch.float32, device=device)

    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        tensor[piece_map[piece.symbol()], 7 - row, col] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    return tensor.unsqueeze(0)  # (1, 13, 8, 8)


def move_to_idx(move):
    return move.from_square * 64 + move.to_square

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")