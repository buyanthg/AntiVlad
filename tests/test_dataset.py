import torch

from core.constants import ACTION_SIZE, BOARD_SIZE, TOTAL_PLANES
from core.training.dataset import ChessDataset


def test_dataset_length():
    data = [
        (
            torch.zeros(TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE),
            torch.zeros(ACTION_SIZE),
            torch.tensor(1.0),
        ),
        (
            torch.ones(TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE),
            torch.ones(ACTION_SIZE),
            torch.tensor(-1.0),
        ),
    ]

    dataset = ChessDataset(data)

    assert len(dataset) == 2


def test_dataset_getitem():
    state = torch.zeros(
        TOTAL_PLANES,
        BOARD_SIZE,
        BOARD_SIZE,
    )
    policy = torch.zeros(ACTION_SIZE)
    value = torch.tensor(1.0)

    dataset = ChessDataset([
        (state, policy, value),
    ])

    result_state, result_policy, result_value = dataset[0]

    assert torch.equal(result_state, state)
    assert torch.equal(result_policy, policy)
    assert torch.equal(result_value, value)


def test_dataset_preserves_data():
    data = [
        (
            torch.randn(
                TOTAL_PLANES,
                BOARD_SIZE,
                BOARD_SIZE,
            ),
            torch.randn(ACTION_SIZE),
            torch.tensor(-1.0),
        )
    ]

    dataset = ChessDataset(data)

    assert dataset.data is data


def test_dataset_returns_correct_types():
    data = [
        (
            torch.zeros(
                TOTAL_PLANES,
                BOARD_SIZE,
                BOARD_SIZE,
            ),
            torch.zeros(ACTION_SIZE),
            torch.tensor(0.0),
        )
    ]

    dataset = ChessDataset(data)

    state, policy, value = dataset[0]

    assert isinstance(state, torch.Tensor)
    assert isinstance(policy, torch.Tensor)
    assert isinstance(value, torch.Tensor)


def test_dataset_returns_individual_samples():
    state_0 = torch.zeros(
        TOTAL_PLANES,
        BOARD_SIZE,
        BOARD_SIZE,
    )
    state_1 = torch.ones(
        TOTAL_PLANES,
        BOARD_SIZE,
        BOARD_SIZE,
    )

    policy_0 = torch.zeros(ACTION_SIZE)
    policy_1 = torch.ones(ACTION_SIZE)

    value_0 = torch.tensor(-1.0)
    value_1 = torch.tensor(1.0)

    data = [
        (state_0, policy_0, value_0),
        (state_1, policy_1, value_1),
    ]

    dataset = ChessDataset(data)

    state, policy, value = dataset[0]

    assert torch.equal(state, state_0)
    assert torch.equal(policy, policy_0)
    assert torch.equal(value, value_0)

    state, policy, value = dataset[1]

    assert torch.equal(state, state_1)
    assert torch.equal(policy, policy_1)
    assert torch.equal(value, value_1)