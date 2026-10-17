import chess
import torch

from constants import *

KNIGHT_DISPLACEMENTS = frozenset({
    (1, 2),
    (2, 1),
})

PROMOTION_PIECES = frozenset({
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
})

def encode_move(move: chess.Move, board) -> tuple[int, int]:
    square_from = move.from_square
    square_to = move.to_square

    file_from = square_from & 7
    rank_from = square_from >> 3
    file_to = square_to & 7
    rank_to = square_to >> 3

    if board.turn == chess.BLACK:
        rank_from = 7 - rank_from
        rank_to = 7 - rank_to

    dx = file_to - file_from
    dy = rank_to - rank_from
    if (abs(dx), abs(dy)) in KNIGHT_DISPLACEMENTS:
        move_type = KNIGHT_MOVES[(dx, dy)]
    elif move.promotion in PROMOTION_PIECES:
        move_type = PROMOTION_MOVES[(dx, dy, move.promotion)]
    else:
        sign_dx = 0 if dx == 0 else dx // abs(dx)
        sign_dy = 0 if dy == 0 else dy // abs(dy)
        direction = DIRECTION_MAP[(sign_dx, sign_dy)]
        distance = max(abs(dx), abs(dy))
        move_type = direction * 7 + (distance - 1)
    square = chess.square(file_from, rank_from)
    return square, move_type


def decode_move(encoded_move, board):
    square, move_type = encoded_move
    file_from = chess.square_file(square)
    rank_from = chess.square_rank(square)

    if move_type < 56:
        direction = move_type // 7
        distance = move_type % 7 + 1

        sign_dx, sign_dy = DIRECTION_MAP_REVERSE[direction]
        dx = sign_dx * distance
        dy = sign_dy * distance

        promotion = None
    elif move_type < 64:
        dx, dy = KNIGHT_MOVES_REVERSE[move_type]
        promotion = None
    else:
        dx, dy, promotion = PROMOTION_MOVES_REVERSE[move_type]

    file_to = file_from + dx
    rank_to = rank_from + dy

    if board.turn == chess.BLACK:
        rank_from = 7 - rank_from
        rank_to = 7 - rank_to
    square_from = chess.square(file_from, rank_from)
    square_to = chess.square(file_to, rank_to)

    if promotion is None:
        piece = board.piece_at(square_from)
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(square_to) in (0, 7):
            promotion = chess.QUEEN

    return chess.Move(square_from, square_to, promotion=promotion)


def action_to_index(encoded_move):
    square, move_type = encoded_move
    return square * 73 + move_type


def index_to_action(index):
    square = index // 73
    move_type = index % 73
    return square, move_type


def encode_action(move, board):
    encoded_move = encode_move(move, board)
    return action_to_index(encoded_move)


def decode_action(index, board):
    encoded_move = index_to_action(index)
    return decode_move(encoded_move, board)


def legal_action_mask(board):
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)

    for move in board.legal_moves:
        index = encode_action(move, board)
        mask[index] = True
    return mask
