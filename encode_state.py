import chess

import torch
import numpy as np
import time
from constants import *

def encode_state(history: list[chess.Board]) -> torch.Tensor:
    """
    Encodes the last HISTORY_LENGTH positions into a tensor.

    Returns:
        Tensor of shape (TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE).
    """
    state_tensor = np.zeros((TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    current_board = history[-1]
    current_player = history[-1].turn
    opponent = not current_player

    # History: N planes per position
    for history_index, board in enumerate(history):
        offset = history_index * PLANES_PER_POSITION

        for square, piece in board.piece_map().items():
            file = chess.square_file(square)
            rank = chess.square_rank(square)

            row = BOARD_SIZE - 1 - rank
            col = file

            piece_offset = piece.piece_type - 1
            if piece.color != current_player:
                piece_offset += PIECE_TYPES

            plane = piece_offset + offset
            state_tensor[plane, row, col] = 1
    # Castling rights
    if current_board.has_kingside_castling_rights(current_player):
        state_tensor[CASTLING_OFFSET] = 1.0
    if current_board.has_queenside_castling_rights(current_player):
        state_tensor[CASTLING_OFFSET + 1] = 1.0
    if current_board.has_kingside_castling_rights(opponent):
        state_tensor[CASTLING_OFFSET + 2] = 1.0
    if current_board.has_queenside_castling_rights(opponent):
        state_tensor[CASTLING_OFFSET + 3] = 1.0

    # En passant: Last Plane
    if current_board.ep_square is not None:
        file = chess.square_file(current_board.ep_square)
        rank = chess.square_rank(current_board.ep_square)

        row = BOARD_SIZE - 1 - rank
        col = file

        state_tensor[EN_PASSANT_OFFSET, row, col] = 1.0

    return torch.from_numpy(state_tensor)