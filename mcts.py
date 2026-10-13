
import chess
import numpy as np
import torch
import time

from action_encoder import encode_action, legal_action_mask, decode_action
from encode_state import encode_state
from constants import *
from antivlad_net import AntiVladNet, get_standart_model


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
        self.children = {}

    def value(self):
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0


class MCTS:
    def __init__(self, model=None, simulations=800, c_puct=1.5,
                dirichlet_alpha=0.3, dirichlet_eps=0.25, add_noise=False,):
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
    # SEARCH: ORIGINAL SEQUENTIAL VERSION
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

        # Без модели пакетный инференс не имеет смысла.
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
            batch_limit = min(
                batch_size,
                self.simulations - completed,
            )

            pending = []
            pending_nodes = set()
            slots = 0

            # ------------------------------------------
            # 1. COLLECT LEAF NODES
            # ------------------------------------------

            while slots < batch_limit:
                result = self.select_batched(root, history, pending_nodes,)

                # Столкнулись с листом, уже зарезервированным
                # другой траекторией этого пакета.
                if result is None:
                    break

                node, node_history, path = result
                slots += 1

                # Терминальные узлы не требуют инференса.
                if node.board.is_game_over(claim_draw=True):
                    backup(node, terminal_value(node.board))
                    completed += 1
                    continue

                self.apply_virtual_loss(path)

                pending.append(
                    (node, node_history, path)
                )
                pending_nodes.add(node)

            # Все слоты могли оказаться терминальными.
            if not pending:
                if completed >= self.simulations:
                    break

                if slots == 0:
                    raise RuntimeError(
                        "Не удалось выбрать лист для MCTS. "
                        "Проверь структуру дерева."
                    )

                continue

            # ------------------------------------------
            # 2. ENCODE ALL STATES
            # ------------------------------------------

            states = torch.stack([
                encode_state(node_history)
                for _, node_history, _ in pending
            ]).to(self.device)

            # ------------------------------------------
            # 3. SINGLE MODEL CALL FOR THE BATCH
            # ------------------------------------------

            with torch.inference_mode():
                batch_policy, batch_values = self.model(states)

            # ------------------------------------------
            # 4. EXPAND AND BACK UP
            # ------------------------------------------

            for i, (node, node_history, path) in enumerate(pending):
                # Убираем временные посещения и значения.
                self.remove_virtual_loss(path)

                prediction = (
                    batch_policy[i],
                    batch_values[i],
                )

                value = self.expand(
                    node,
                    node_history,
                    prediction=prediction,
                )

                backup(node, value)
                completed += 1

        return root

    # --------------------------------------------------
    # SELECTION
    # --------------------------------------------------

    def select(self, node, history):
        current_history = history

        while (
            node.is_expanded()
            and not node.board.is_game_over(claim_draw=True)
        ):
            node = max(
                node.children.values(),
                key=lambda child: self.ucb_score(node, child),
            )

            current_history = update_history(
                current_history,
                node.board,
            )

        return node, current_history

    def select_batched(self, node, history, pending_nodes):
        current_history = history
        path = [node]

        while (
            node.is_expanded()
            and not node.board.is_game_over(claim_draw=True)
        ):
            node = max(
                node.children.values(),
                key=lambda child: self.ucb_score(node, child),
            )

            path.append(node)

            current_history = update_history(
                current_history,
                node.board,
            )

            # Этот лист уже обрабатывается в текущем батче.
            if node in pending_nodes:
                return None

        if node in pending_nodes:
            return None

        return node, current_history, path

    def ucb_score(self, parent, child):
        if child.visit_count == 0:
            q = 0.0
        else:
            q = -child.value()

        u = self.c_puct * child.prior * np.sqrt(parent.visit_count) / (1 + child.visit_count)

        return u + q

    # --------------------------------------------------
    # VIRTUAL LOSS
    # --------------------------------------------------

    @staticmethod
    def apply_virtual_loss(path):
        # Временная оценка листа: проигрыш для игрока,
        # которому принадлежит ход в этом листе.
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

        # ----------------------------------------------
        # NO NEURAL NETWORK
        # ----------------------------------------------

        if self.model is None:
            prior = 1.0 / len(legal_moves)

            for move in legal_moves:
                index = encode_action(move, board)

                child_board = board.copy(stack=False)
                child_board.push(move)

                node.children[index] = Node(
                    board=child_board,
                    parent=node,
                    prior=prior,
                    action=index,
                )

            return 0.0

        # ----------------------------------------------
        # INFERENCE
        # ----------------------------------------------

        if prediction is None:
            state = encode_state(history).unsqueeze(0).to(
                self.device
            )

            with torch.inference_mode():
                policy_logits, value = self.model(state)

            policy_logits = policy_logits[0]
            value = value[0]

        else:
            policy_logits, value = prediction

        # ----------------------------------------------
        # POLICY ON CPU
        # ----------------------------------------------

        mask = legal_action_mask(board)
        legal_indices = torch.where(mask)[0]

        policy_cpu = policy_logits.detach().cpu()
        legal_logits = policy_cpu[legal_indices]

        legal_priors = torch.softmax(legal_logits, dim=0)

        legal_indices = legal_indices.tolist()
        legal_priors = legal_priors.tolist()

        # ----------------------------------------------
        # CREATE CHILDREN
        # ----------------------------------------------

        for index, prior in zip(legal_indices, legal_priors):
            move = decode_action(index, board)

            child_board = board.copy(stack=False)
            child_board.push(move)

            node.children[index] = Node(
                board=child_board,
                parent=node,
                prior=prior,
                action=index,
            )

        return value.item()

    # --------------------------------------------------
    # DIRICHLET NOISE
    # --------------------------------------------------

    def add_dirichlet_noise(self, root):
        children = list(root.children.values())

        if not children:
            return

        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))

        epsilon = self.dirichlet_eps

        for child, noise_value in zip(children, noise):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise_value

