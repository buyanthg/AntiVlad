import unittest
import chess
import torch

from action_encoder import encode_move, decode_move, action_to_index, index_to_action, legal_action_mask, encode_action, decode_action


def test_encode_decode():
    board = chess.Board()

    for move in board.legal_moves:
        encoded = encode_move(move, board)
        decoded = decode_move(encoded, board)

        assert decoded == move

def test_knight_moves():
    board = chess.Board()

    for move in board.legal_moves:
        if board.piece_at(move.from_square).piece_type == chess.KNIGHT:
            encoded = encode_move(move, board)
            decoded = decode_move(encoded, board)

            assert decoded == move

def test_castling():
    board = chess.Board(
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    )

    for move in board.legal_moves:
        encoded = encode_move(move, board)
        decoded = decode_move(encoded, board)

        assert decoded == move

def test_promotions():
    board = chess.Board(
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
    )

    for move in board.legal_moves:
        encoded = encode_move(move, board)
        decoded = decode_move(encoded, board)

        assert decoded == move

def test_action_roundtrip():
    boards = [
        chess.Board(),
        chess.Board(
            "rnbqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3"
        ),
    ]

    for board in boards:
        mask = legal_action_mask(board)

        for action in torch.where(mask)[0].tolist():
            move = decode_action(action, board)
            encoded = encode_action(move, board)

            assert encoded == action

def test_action_index_range():
    assert action_to_index((0, 0)) == 0
    assert action_to_index((63, 72)) == 4671

def test_encode_decode_action():
    board = chess.Board()

    for move in board.legal_moves:
        index = encode_action(move, board)

        decoded_move = decode_action(index, board)

        assert decoded_move == move

def test_legal_action_mask():
    board = chess.Board()

    mask = legal_action_mask(board)

    assert mask.shape == (4672,)
    assert mask.dtype == torch.bool
    assert mask.sum().item() == board.legal_moves.count()

def test_legal_action_mask_contains_legal_moves():
    board = chess.Board()
    mask = legal_action_mask(board)

    for move in board.legal_moves:
        index = encode_action(move, board)
        assert mask[index]

POSITIONS = [
    chess.Board(),
    chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
]

def test_encode_decode_positions():
    for board in POSITIONS:
        for move in board.legal_moves:
            index = encode_action(move, board)
            decoded = decode_action(index, board)

            assert decoded == move
