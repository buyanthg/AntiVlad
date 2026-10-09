import chess
import torch
from torch import nn
import pytest
from constants import *
from mcts import MCTS, Node, backup, terminal_value, update_history
from antivlad_net import AntiVladNet

class DummyModel(nn.Module):
    def __init__(self, action_size: int):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.action_size = action_size

    def forward(self, x):
        batch_size = x.shape[0]

        policy = torch.zeros(
            batch_size,
            self.action_size,
            device=x.device,
        )

        value = torch.zeros(
            batch_size,
            1,
            device=x.device,
        )

        return policy, value

@pytest.fixture
def model():
    return AntiVladNet(
        64,
        32,
        32,
        16,
        5,
    )

def make_history(board):
    return [board.copy(stack=False) for _ in range(HISTORY_LENGTH)]


def test_backup():
    root = Node(chess.Board())
    child = Node(chess.Board(), parent=root)

    backup(child, 1.0)

    assert child.visit_count == 1
    assert child.value_sum == 1.0

    assert root.visit_count == 1
    assert root.value_sum == -1.0


def test_node_value():
    node = Node(chess.Board())

    assert node.value() == 0.0

    node.visit_count = 2
    node.value_sum = 1.0

    assert node.value() == 0.5


def test_terminal_value_white_wins():
    board = chess.Board()

    moves = [
        "f2f3",
        "e7e5",
        "g2g4",
        "d8h4",
    ]

    for move in moves:
        board.push_uci(move)

    assert board.is_checkmate()
    assert board.turn == chess.WHITE
    assert terminal_value(board) == -1.0


def test_terminal_value_black_wins():
    board = chess.Board()

    moves = [
        "e2e4",
        "e7e5",
        "f1c4",
        "b8c6",
        "d1h5",
        "g8f6",
        "h5f7",
    ]

    for move in moves:
        board.push_uci(move)

    assert board.is_checkmate()
    assert board.turn == chess.BLACK
    assert terminal_value(board) == -1.0


def test_terminal_value_draw():
    board = chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")

    assert terminal_value(board) == 0.0


def test_search_accepts_history():
    board = chess.Board()

    history = make_history(board)

    model = DummyModel(8 * 8 * 73)
    mcts = MCTS(model, simulations=1)

    root = mcts.search(history)

    assert root.board == board
    assert root.is_expanded()


def test_search_uses_current_board_from_history():
    board = chess.Board()

    history = make_history(board)

    model = DummyModel(8 * 8 * 73)
    mcts = MCTS(model, simulations=1)

    root = mcts.search(history)

    assert root.board == history[-1]
    assert root.board is not history[-1]


def test_update_history():
    board = chess.Board()
    history = [
        board.copy(stack=False)
        for _ in range(HISTORY_LENGTH)
    ]

    new_board = board.copy(stack=False)
    new_board.push(chess.Move.from_uci("e2e4"))

    updated = update_history(history, new_board)

    assert len(updated) == HISTORY_LENGTH
    assert updated[-1].fen() == new_board.fen()
    assert updated[:-1] == history[1:]

def test_mcts_policy_visits(model):
    mcts = MCTS(model, simulations=10)

    board = chess.Board()
    history = [board.copy(stack=False) for _ in range(HISTORY_LENGTH)]

    root = mcts.search(history)

    total_visits = sum(
        child.visit_count
        for child in root.children.values()
    )

    assert total_visits == 10