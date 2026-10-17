import numpy as np
from constants import *


def encode_state(history: list[chess.Board]) -> torch.Tensor:
    state = np.zeros((TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    history = history[-HISTORY_LENGTH:]
    current_board = history[-1]
    current_player = current_board.turn
    opponent = not current_player

    for history_index, board in enumerate(history):
        offset = history_index * PLANES_PER_POSITION

        for color in (chess.WHITE, chess.BLACK):
            color_offset = 0 if color == current_player else PIECE_TYPES

            for piece_type in range(1, PIECE_TYPES + 1):
                plane = offset + color_offset + piece_type - 1
                squares = int(board.pieces(piece_type, color))

                while squares:
                    bit = squares & -squares
                    square = bit.bit_length() - 1
                    row = BOARD_SIZE - 1 - (square >> 3)
                    col = square & 7

                    state[plane, row, col] = 1
                    squares ^= bit

    if current_board.has_kingside_castling_rights(current_player):
        state[CASTLING_OFFSET] = 1
    if current_board.has_queenside_castling_rights(current_player):
        state[CASTLING_OFFSET + 1] = 1
    if current_board.has_kingside_castling_rights(opponent):
        state[CASTLING_OFFSET + 2] = 1
    if current_board.has_queenside_castling_rights(opponent):
        state[CASTLING_OFFSET + 3] = 1

    if current_board.ep_square is not None:
        square = current_board.ep_square
        row = BOARD_SIZE - 1 - (square >> 3)
        col = square & 7
        state[EN_PASSANT_OFFSET, row, col] = 1

    return torch.from_numpy(state)
