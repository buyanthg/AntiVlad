import chess
import numpy as np
import torch
import pytest

from core.constants import ACTION_SIZE, HISTORY_LENGTH
from core.network.antivlad_net import AntiVladNet
from core.training.self_play import (
    SelfPlay,
    create_history,
    create_training_data,
    get_policy,
    get_result,
    select_action,
)

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


def test_create_history():
    board = chess.Board()

    history = create_history(board)

    assert len(history) == HISTORY_LENGTH

    for historical_board in history:
        assert historical_board == board
        assert historical_board is not board


def test_get_result_white_wins():
    board = chess.Board()

    for move in [
        "e2e4",
        "e7e5",
        "f1c4",
        "b8c6",
        "d1h5",
        "g8f6",
        "h5f7",
    ]:
        board.push_uci(move)

    assert board.is_checkmate()
    assert get_result(board) == chess.WHITE


def test_get_result_black_wins():
    board = chess.Board()

    for move in [
        "f2f3",
        "e7e5",
        "g2g4",
        "d8h4",
    ]:
        board.push_uci(move)

    assert board.is_checkmate()
    assert get_result(board) == chess.BLACK


def test_get_result_draw():
    board = chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")

    assert board.is_stalemate()
    assert get_result(board) is None


def test_get_policy():
    board = chess.Board()

    class DummyChild:
        def __init__(self, visit_count):
            self.visit_count = visit_count

    class DummyRoot:
        def __init__(self):
            self.children = {
                10: DummyChild(2),
                20: DummyChild(3),
                30: DummyChild(5),
            }

    root = DummyRoot()

    policy = get_policy(root)

    assert policy.shape == (ACTION_SIZE,)
    assert policy.dtype == np.float32

    assert np.isclose(policy[10], 0.2)
    assert np.isclose(policy[20], 0.3)
    assert np.isclose(policy[30], 0.5)

    assert np.isclose(policy.sum(), 1.0)


def test_get_policy_without_visits():
    class DummyRoot:
        children = {}

    policy = get_policy(DummyRoot())

    assert policy.shape == (ACTION_SIZE,)
    assert np.all(policy == 0)


def test_select_action_temperature_one(monkeypatch):
    class DummyChild:
        def __init__(self, visit_count):
            self.visit_count = visit_count

    class DummyRoot:
        children = {
            10: DummyChild(1),
            20: DummyChild(3),
            30: DummyChild(6),
        }

    captured = {}

    def fake_choice(actions, p):
        captured["actions"] = actions
        captured["probabilities"] = p
        return actions[0]

    monkeypatch.setattr(np.random, "choice", fake_choice)

    select_action(DummyRoot(), 0)

    assert captured["actions"] == [10, 20, 30]
    assert np.allclose(
        captured["probabilities"],
        [0.1, 0.3, 0.6],
    )


def test_select_action_after_temperature():
    class DummyChild:
        def __init__(self, visit_count):
            self.visit_count = visit_count

    class DummyRoot:
        children = {
            10: DummyChild(1),
            20: DummyChild(7),
            30: DummyChild(3),
        }

    action = select_action(DummyRoot(), 30)

    assert action == 20


def test_create_training_data_white_win():
    state = torch.zeros(101, 8, 8)
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)

    data = create_training_data(
        [state, state],
        [policy, policy],
        [chess.WHITE, chess.BLACK],
        chess.WHITE,
    )

    assert len(data) == 2

    assert torch.equal(data[0][0], state)
    assert torch.equal(data[1][0], state)

    assert data[0][2].item() == 1.0
    assert data[1][2].item() == -1.0


def test_create_training_data_draw():
    state = torch.zeros(101, 8, 8)
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)

    data = create_training_data(
        [state, state],
        [policy, policy],
        [chess.WHITE, chess.BLACK],
        None,
    )

    assert len(data) == 2
    assert data[0][2].item() == 0.0
    assert data[1][2].item() == 0.0


def test_create_training_data_policy_is_tensor():
    state = torch.zeros(101, 8, 8)
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)

    data = create_training_data(
        [state],
        [policy],
        [chess.WHITE],
        chess.WHITE,
    )

    _, policy_tensor, value = data[0]

    assert isinstance(policy_tensor, torch.Tensor)
    assert policy_tensor.shape == (ACTION_SIZE,)
    assert policy_tensor.dtype == torch.float32
    assert isinstance(value, torch.Tensor)
    assert value.dtype == torch.float32

def test_self_play_produces_data(model):
    self_play = SelfPlay(
        model,
        simulations=1,
    )

    data = self_play.play_game()

    assert len(data) > 0

    for state, policy, value in data:
        assert state.shape == (101, 8, 8)
        assert policy.shape == (4672,)
        assert value.ndim == 0