import chess
import torch

from constants import *
from encode_state import encode_state


def test_shape():
    board = chess.Board()
    history = [board.copy() for _ in range(HISTORY_LENGTH)]

    state = encode_state(history)

    assert state.shape == (
        TOTAL_PLANES,
        BOARD_SIZE,
        BOARD_SIZE,
    )


def test_initial_position():
    board = chess.Board()
    history = [board.copy() for _ in range(HISTORY_LENGTH)]

    state = encode_state(history)

    # White pieces are "own"
    assert state[0].sum() == 8       # own pawns
    assert state[1].sum() == 2       # own knights
    assert state[2].sum() == 2       # own bishops
    assert state[3].sum() == 2       # own rooks
    assert state[4].sum() == 1       # own queen
    assert state[5].sum() == 1       # own king

    # Black pieces are "opponent"
    assert state[6].sum() == 8
    assert state[7].sum() == 2
    assert state[8].sum() == 2
    assert state[9].sum() == 2
    assert state[10].sum() == 1
    assert state[11].sum() == 1


def test_history():
    board = chess.Board()

    history = []

    for _ in range(HISTORY_LENGTH):
        history.append(board.copy())

        if not board.is_game_over():
            move = list(board.legal_moves)[0]
            board.push(move)

    state = encode_state(history)

    for i in range(HISTORY_LENGTH):
        offset = i * PLANES_PER_POSITION

        assert state[offset:offset + PLANES_PER_POSITION].sum() > 0


def test_current_player_perspective():
    board = chess.Board()

    # 1. e4
    board.push_san("e4")

    # 1... e5
    board.push_san("e5")

    # 2. Nf3
    board.push_san("Nf3")

    # Now Black to move
    history = [board.copy() for _ in range(HISTORY_LENGTH)]

    state = encode_state(history)

    # Black must be "own"
    assert state[0].sum() == 8
    assert state[1].sum() == 2
    assert state[5].sum() == 1

    # White must be "opponent"
    assert state[6].sum() == 8
    assert state[7].sum() == 2
    assert state[11].sum() == 1


def test_board_orientation():
    white_board = chess.Board()
    white_history = [white_board.copy() for _ in range(HISTORY_LENGTH)]

    white_state = encode_state(white_history)

    # White rook on a1
    assert white_state[3, 7, 0] == 1

    black_board = chess.Board()
    black_board.turn = chess.BLACK

    black_history = [black_board.copy() for _ in range(HISTORY_LENGTH)]

    black_state = encode_state(black_history)

    # From Black's perspective a1 becomes h8
    assert black_state[3, 0, 7] == 1

import chess
import torch

from constants import (
    CASTLING_OFFSET,
    EN_PASSANT_OFFSET,
)
from encode_state import encode_state


def test_castling_rights():
    board = chess.Board()

    state = encode_state([board] * 8)

    # White to move:
    # own kingside
    assert torch.all(state[CASTLING_OFFSET] == 1)

    # own queenside
    assert torch.all(state[CASTLING_OFFSET + 1] == 1)

    # opponent kingside
    assert torch.all(state[CASTLING_OFFSET + 2] == 1)

    # opponent queenside
    assert torch.all(state[CASTLING_OFFSET + 3] == 1)


def test_castling_rights_relative_to_player():
    board = chess.Board()
    board.turn = chess.BLACK

    state = encode_state([board] * 8)

    # Black is now the current player,
    # so its castling rights are "own".
    assert torch.all(state[CASTLING_OFFSET] == 1)
    assert torch.all(state[CASTLING_OFFSET + 1] == 1)

    # White rights are "opponent".
    assert torch.all(state[CASTLING_OFFSET + 2] == 1)
    assert torch.all(state[CASTLING_OFFSET + 3] == 1)


def test_castling_rights_absent():
    board = chess.Board(
        "r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"
    )

    state = encode_state([board] * 8)

    for plane in range(4):
        assert torch.all(state[CASTLING_OFFSET + plane] == 0)


def test_castling_rights_partial():
    board = chess.Board(
        "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"
    )

    state = encode_state([board] * 8)

    # White kingside: есть
    assert torch.all(state[CASTLING_OFFSET] == 1)

    # White queenside: нет
    assert torch.all(state[CASTLING_OFFSET + 1] == 0)

    # Black kingside: нет
    assert torch.all(state[CASTLING_OFFSET + 2] == 0)

    # Black queenside: есть
    assert torch.all(state[CASTLING_OFFSET + 3] == 1)

def test_en_passant():
    board = chess.Board()

    board.ep_square = chess.E6

    state = encode_state([board] * 8)

    plane = state[EN_PASSANT_OFFSET]

    # e6 -> file 4, rank 5
    # row = 8 - 1 - 5 = 2
    assert plane[2, 4] == 1

    # Only one cell should be active.
    assert plane.sum() == 1


def test_en_passant_absent():
    board = chess.Board()

    board.ep_square = None

    state = encode_state([board] * 8)

    assert torch.all(state[EN_PASSANT_OFFSET] == 0)
