import chess
import torch

# ========================
# Network architecture
# ========================
NETWORK_CHANNELS = 128
POLICY_CHANNELS = 32
VALUE_CHANNELS = 32
VALUE_HIDDEN = 16
NUM_RESIDUAL_BLOCKS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Train
# =========================
CHECKPOINT_DIR = "../checkpoints"

# =========================
# Board / state
# =========================

BOARD_SIZE = 8

HISTORY_LENGTH = 8

PIECE_TYPES = 6
PLANES_PER_POSITION = 12

CASTLING_PLANES = 4
EN_PASSANT_PLANES = 1

HISTORY_PLANES = HISTORY_LENGTH * PLANES_PER_POSITION

CASTLING_OFFSET = HISTORY_PLANES
EN_PASSANT_OFFSET = CASTLING_OFFSET + CASTLING_PLANES

TOTAL_PLANES = HISTORY_PLANES + EN_PASSANT_PLANES + CASTLING_PLANES


# =========================
# Self-play
# =========================

TEMPERATURE_MOVES = 30


# =========================
# Action space
# =========================

ACTION_SIZE = BOARD_SIZE * BOARD_SIZE * 73


# =========================
# Action encoding
# =========================

DIRECTION_MAP = {
    (0, 1): 0,    # ↑
    (1, 1): 1,    # ↗
    (1, 0): 2,    # →
    (1, -1): 3,   # ↘
    (0, -1): 4,   # ↓
    (-1, -1): 5,  # ↙
    (-1, 0): 6,   # ←
    (-1, 1): 7,   # ↖
}

KNIGHT_MOVES = {
    (1, 2): 56,
    (2, 1): 57,
    (2, -1): 58,
    (1, -2): 59,
    (-1, -2): 60,
    (-2, -1): 61,
    (-2, 1): 62,
    (-1, 2): 63,
}

PROMOTION_MOVES = {
    (-1, 1, chess.KNIGHT): 64,
    (0, 1, chess.KNIGHT): 65,
    (1, 1, chess.KNIGHT): 66,

    (-1, 1, chess.BISHOP): 67,
    (0, 1, chess.BISHOP): 68,
    (1, 1, chess.BISHOP): 69,

    (-1, 1, chess.ROOK): 70,
    (0, 1, chess.ROOK): 71,
    (1, 1, chess.ROOK): 72,
}

KNIGHT_TABLE = [[-1] * 5 for _ in range(5)]
for (dx, dy), value in KNIGHT_MOVES.items():
    KNIGHT_TABLE[dx + 2][dy + 2] = value

DIRECTION_TABLE = [[-1] * 3 for _ in range(3)]
for (dx, dy), value in DIRECTION_MAP.items():
    DIRECTION_TABLE[dx + 1][dy + 1] = value

PROMOTION_INDEX = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK: 2,
}

PROMOTION_TABLE = [
    [[-1] * 3 for _ in range(3)]
    for _ in range(3)
]
for (dx, dy, piece), value in PROMOTION_MOVES.items():
    PROMOTION_TABLE[dx + 1][dy + 1][PROMOTION_INDEX[piece]] = value
# =========================
# Reverse maps
# =========================

KNIGHT_MOVES_REVERSE = {v: k for k, v in KNIGHT_MOVES.items()}
PROMOTION_MOVES_REVERSE = {v: k for k, v in PROMOTION_MOVES.items()}
DIRECTION_MAP_REVERSE = {v: k for k, v in DIRECTION_MAP.items()}
