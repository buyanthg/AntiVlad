import chess
import numpy as np
import torch

from action_encoder import encode_action
from encode_state import encode_state


def backup(node, value):
    while node is not None:
        node.visit_count += 1
        node.value_sum += value

        value = -value
        node = node.parent


def terminal_value(board):
    result = board.result(claim_draw=True)

    if result == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0

    if result == "0-1":
        return 1.0 if board.turn == chess.BLACK else -1.0

    return 0.0


def update_history(history, board):
    return history[1:] + [board]


class Node:
    def __init__(self, board, parent=None, prior=0.0, action=None):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.action = action

        self.visit_count = 0
        self.value_sum = 0.0

        # Материализованные дочерние узлы.
        # Ключ: индекс действия.
        self.children = {}

        # None: узел ещё не раскрыт.
        # dict: индекс действия -> (chess.Move, prior).
        self.legal_actions = None

    def value(self):
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    def is_expanded(self):
        return bool(self.legal_actions)


class MCTS:
    def __init__(
        self,
        model=None,
        simulations=800,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        add_noise=False,
    ):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.add_noise = add_noise

        if self.model is not None:
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device("cpu")

    # --------------------------------------------------
    # SEARCH: SEQUENTIAL
    # --------------------------------------------------

    def search(self, history):
        root = Node(history[-1].copy(stack=False))

        if root.board.is_game_over(claim_draw=True):
            return root

        self.expand(root, history)

        if self.add_noise:
            self.add_dirichlet_noise(root)

        for _ in range(self.simulations):
            node, node_history = self.select(root, history)

            if node.board.is_game_over(claim_draw=True):
                value = terminal_value(node.board)
            else:
                value = self.expand(node, node_history)

            backup(node, value)

        return root

    # --------------------------------------------------
    # SEARCH: BATCHED INFERENCE
    # --------------------------------------------------

    def search_batched(self, history, batch_size=8):
        if batch_size < 1:
            raise ValueError("batch_size должен быть >= 1")

        if self.model is None:
            return self.search(history)

        root = Node(history[-1].copy(stack=False))

        if root.board.is_game_over(claim_draw=True):
            return root

        self.expand(root, history)

        if self.add_noise:
            self.add_dirichlet_noise(root)

        completed = 0

        while completed < self.simulations:
            batch_limit = min(batch_size, self.simulations - completed)

            pending = []
            pending_nodes = set()
            slots = 0

            # Выбираем листья для текущего пакета.
            while slots < batch_limit:
                result = self.select_batched(root, history, pending_nodes)

                if result is None:
                    break

                node, node_history, path = result
                slots += 1

                if node.board.is_game_over(claim_draw=True):
                    backup(node, terminal_value(node.board))
                    completed += 1
                    continue

                self.apply_virtual_loss(path)

                pending.append((node, node_history, path))
                pending_nodes.add(node)

            if not pending:
                if completed >= self.simulations:
                    break

                if slots == 0:
                    raise RuntimeError("Не удалось выбрать лист для MCTS. Проверь структуру дерева.")

                continue

            # Кодируем состояния одним пакетом.
            states = torch.stack([encode_state(node_history) for _, node_history, _ in pending]).to(self.device)

            # Один вызов сети на весь пакет.
            with torch.inference_mode():
                batch_policy, batch_values = self.model(states)

            batch_policy = batch_policy.cpu()
            batch_values = batch_values.cpu()

            # Раскрываем листья и обновляем значения дерева.
            for i, (node, node_history, path) in enumerate(pending):
                self.remove_virtual_loss(path)

                prediction = (batch_policy[i], batch_values[i])

                value = self.expand(node, node_history, prediction=prediction,)

                backup(node, value)
                completed += 1

        return root

    # --------------------------------------------------
    # SELECTION
    # --------------------------------------------------

    def create_child(self, node, action_index):
        """Создать дочерний узел при первом выборе хода."""
        child = node.children.get(action_index)

        if child is not None:
            return child

        if not node.legal_actions or action_index not in node.legal_actions:
            raise KeyError(f"Действие {action_index} отсутствует среди легальных ходов узла.")

        move, prior = node.legal_actions[action_index]

        child_board = node.board.copy(stack=False)
        child_board.push(move)

        child = Node(board=child_board, parent=node, prior=prior, action=action_index)

        node.children[action_index] = child
        return child

    def select_child(self, node):
        """Выбрать ход по PUCT среди всех легальных действий."""
        legal_actions = node.legal_actions

        if not legal_actions:
            raise RuntimeError("Попытка выбрать ход из пустого узла.")

        sqrt_parent_visits = np.sqrt(node.visit_count)

        best_action = None
        best_score = -float("inf")

        for action_index, (_, prior) in legal_actions.items():
            child = node.children.get(action_index)

            if child is None:
                q = 0.0
                visits = 0
                effective_prior = prior
            else:
                visits = child.visit_count
                q = -child.value() if visits else 0.0
                effective_prior = child.prior

            score = q + self.c_puct * effective_prior * sqrt_parent_visits / (1 + visits)

            if score > best_score:
                best_score = score
                best_action = action_index

        return self.create_child(node, best_action)

    def select(self, node, history):
        current_history = history

        while node.is_expanded() and not node.board.is_game_over(claim_draw=True):
            node = self.select_child(node)
            current_history = update_history(current_history, node.board)

        return node, current_history

    def select_batched(self, node, history, pending_nodes):
        path = [node]

        while node.is_expanded() and not node.board.is_game_over(claim_draw=True):
            node = self.select_child(node)
            path.append(node)

            if node in pending_nodes:
                return None

        if node in pending_nodes:
            return None

        # История нужна для кодирования конечного листа.
        current_history = history

        for path_node in path[1:]:
            current_history = update_history(current_history, path_node.board)

        return node, current_history, path

    # --------------------------------------------------
    # VIRTUAL LOSS
    # --------------------------------------------------

    @staticmethod
    def apply_virtual_loss(path):
        value = -1.0

        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    @staticmethod
    def remove_virtual_loss(path):
        value = -1.0

        for node in reversed(path):
            node.visit_count -= 1
            node.value_sum -= value
            value = -value

    # --------------------------------------------------
    # EXPANSION
    # --------------------------------------------------

    def expand(self, node, history, prediction=None):
        board = node.board

        if board.is_game_over(claim_draw=True):
            return terminal_value(board)

        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return terminal_value(board)

        # Без нейросети: равномерное распределение по ходам.
        if self.model is None:
            prior = 1.0 / len(legal_moves)

            node.legal_actions = {encode_action(move, board): (move, prior) for move in legal_moves}

            return 0.0

        # Если предсказание не передано, запускаем сеть отдельно.
        if prediction is None:
            state = encode_state(history).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                policy_logits, value = self.model(state)

            policy_logits = policy_logits[0]
            value = value[0]
        else:
            policy_logits, value = prediction

        # Кодируем только легальные действия.
        legal_indices = [encode_action(move, board) for move in legal_moves]

        # Преобразуем logits в вероятности по легальным ходам.
        policy_np = policy_logits.detach().cpu().numpy()
        legal_logits = policy_np[legal_indices]

        legal_priors = np.exp(legal_logits - legal_logits.max())
        legal_priors /= legal_priors.sum()

        legal_priors = legal_priors.tolist()

        # Сохраняем метаданные действий, не создавая всё дерево.
        node.legal_actions = {legal_indices[i]: (legal_moves[i], float(legal_priors[i]))
                              for i in range(len(legal_indices))
        }

        return value.item()

    # --------------------------------------------------
    # DIRICHLET NOISE
    # --------------------------------------------------

    def add_dirichlet_noise(self, root):
        if not root.legal_actions:
            return

        actions = list(root.legal_actions.items())

        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))

        epsilon = self.dirichlet_eps

        for (action_index, (move, prior)), noise_value in zip(actions, noise):
            new_prior = (1 - epsilon) * prior + epsilon * float(noise_value)

            root.legal_actions[action_index] = (move, new_prior)

            child = root.children.get(action_index)

            if child is not None:
                child.prior = new_prior