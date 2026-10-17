
import chess
import pytest
import torch
from torch import nn

from mcts import MCTS


class DummyModel(nn.Module):
    """Минимальная модель с тем же интерфейсом, что AntiVladNet."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.batch_sizes = []

    def forward(self, states):
        batch_size = states.shape[0]
        self.batch_sizes.append(batch_size)

        policy = torch.zeros(
            batch_size, 8 * 8 * 73,
            device=states.device,
        )
        value = torch.zeros(
            batch_size,
            device=states.device,
        )

        return policy, value


def test_search_batched_returns_expanded_root():
    model = DummyModel()
    mcts = MCTS(model=model, simulations=8)

    board = chess.Board()
    root = mcts.search_batched([board], batch_size=4)

    assert root.board == board
    assert root.is_expanded()
    assert len(root.children) > 0
    assert root.visit_count == 8


def test_search_batched_rejects_invalid_batch_size():
    mcts = MCTS(model=DummyModel(), simulations=4)

    with pytest.raises(ValueError, match="batch_size"):
        mcts.search_batched([chess.Board()], batch_size=0)

    with pytest.raises(ValueError, match="batch_size"):
        mcts.search_batched([chess.Board()], batch_size=-1)


def test_search_batched_terminal_position():
    # Пат: чёрные не имеют легальных ходов, но король не под шахом.
    board = chess.Board(
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    )
    assert board.is_stalemate()

    model = DummyModel()
    mcts = MCTS(model=model, simulations=8)

    root = mcts.search_batched([board], batch_size=4)

    assert root.board == board
    assert not root.is_expanded()
    assert root.visit_count == 0
    assert model.batch_sizes == []


def test_search_batched_falls_back_without_model(monkeypatch):
    mcts = MCTS(model=None, simulations=4)
    history = [chess.Board()]
    expected_root = object()

    def fake_search(received_history):
        assert received_history is history
        return expected_root

    monkeypatch.setattr(mcts, "search", fake_search)

    result = mcts.search_batched(history, batch_size=4)

    assert result is expected_root


def test_search_batched_calls_model_in_batches():
    model = DummyModel()
    simulations = 7
    batch_size = 3

    mcts = MCTS(model=model, simulations=simulations)
    root = mcts.search_batched(
        [chess.Board()],
        batch_size=batch_size,
    )

    assert root.visit_count == simulations

    # Первый вызов модели расширяет корень отдельно.
    # Остальные вызовы обрабатывают листья батчами.
    assert model.batch_sizes[0] == 1
    assert all(
        1 <= size <= batch_size
        for size in model.batch_sizes[1:]
    )

    assert sum(model.batch_sizes[1:]) == simulations


def test_search_batched_with_batch_size_one():
    model = DummyModel()
    mcts = MCTS(model=model, simulations=4)

    root = mcts.search_batched(
        [chess.Board()],
        batch_size=1,
    )

    assert root.is_expanded()
    assert root.visit_count == 4
    assert all(size == 1 for size in model.batch_sizes)


def test_search_batched_does_not_modify_original_board():
    model = DummyModel()
    mcts = MCTS(model=model, simulations=4)

    board = chess.Board()
    original_fen = board.fen()

    mcts.search_batched([board], batch_size=2)

    assert board.fen() == original_fen



def test_virtual_loss_apply_and_remove_restore_node_stats():
    from mcts import Node

    root = Node(chess.Board())
    child = Node(
        root.board.copy(stack=False),
        parent=root,
    )
    leaf = Node(
        child.board.copy(stack=False),
        parent=child,
    )

    path = [root, child, leaf]

    # Задаём исходную статистику, чтобы проверить восстановление.
    root.visit_count = 5
    root.value_sum = 1.5

    child.visit_count = 3
    child.value_sum = -0.5

    leaf.visit_count = 2
    leaf.value_sum = 0.25

    original = [
        (node.visit_count, node.value_sum)
        for node in path
    ]

    MCTS.apply_virtual_loss(path)

    assert [
        node.visit_count for node in path
    ] == [count + 1 for count, _ in original]

    assert [
        node.value_sum for node in path
    ] != [value for _, value in original]

    MCTS.remove_virtual_loss(path)

    for node, (visits, value_sum) in zip(path, original):
        assert node.visit_count == visits
        assert node.value_sum == pytest.approx(value_sum)


def test_virtual_loss_restored_after_batched_search(monkeypatch):
    model = DummyModel()
    mcts = MCTS(model=model, simulations=8)

    original_apply = mcts.apply_virtual_loss
    original_remove = mcts.remove_virtual_loss

    applied_paths = []
    removed_paths = []

    def track_apply(path):
        applied_paths.append(tuple(id(node) for node in path))
        original_apply(path)

    def track_remove(path):
        removed_paths.append(tuple(id(node) for node in path))
        original_remove(path)

    monkeypatch.setattr(mcts, "apply_virtual_loss", track_apply)
    monkeypatch.setattr(mcts, "remove_virtual_loss", track_remove)

    root = mcts.search_batched(
        [chess.Board()],
        batch_size=4,
    )

    assert len(applied_paths) == len(removed_paths)
    assert sorted(applied_paths) == sorted(removed_paths)

    # Все реальные посещения должны остаться неотрицательными.
    def check_tree(node):
        assert node.visit_count >= 0

        for child in node.children.values():
            check_tree(child)

    check_tree(root)
