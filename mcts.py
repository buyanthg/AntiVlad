import chess
import numpy as np
import torch

import time

from action_encoder import encode_action, legal_action_mask, decode_action
from encode_state import encode_state

from constants import *
from antivlad_net import AntiVladNet

import matplotlib.pyplot as plt

def backup(node, value):
    while node is not None:
        node.visit_count += 1
        node.value_sum += value

        value = -value
        node = node.parent


def terminal_value(board):
    result = board.result()
    if result == '1-0':
        return 1.0 if board.turn == chess.WHITE else -1.0

    if result == '0-1':
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
        self.value_sum = 0
        self.children = {}

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0


class MCTS:
    def __init__(self, model, simulations=100, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

        self.device = next(self.model.parameters()).device

    def search(self, history):
        root = Node(history[-1].copy(stack=False))

        self.expand(root, history)
        self.add_dirichlet_noise(root)

        for _ in range(self.simulations):
            node, node_history = self.select(root, history)

            if node.board.is_game_over():
                value = terminal_value(node.board)
            else:
                value = self.expand(node, node_history)

            backup(node, value)

        return root

    def select(self, node, history):
        current_history = history

        while node.is_expanded() and not node.board.is_game_over():
            node = max(
                node.children.values(),
                key=lambda child: self.ucb_score(node, child)
            )
        current_history = update_history(current_history, node.board)
        return node, current_history

    def ucb_score(self, parent, child):
        if child.visit_count == 0:
            q = 0.0
        else:
            q = - child.value()

        u = (
            self.c_puct *
            child.prior *
            np.sqrt(parent.visit_count) /
            (1 + child.visit_count)
        )

        return u + q

    def expand(self, node, history):
        board = node.board

        state = encode_state(history)
        state = state.unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(state)

        policy_logits = policy_logits[0]
        value = value.item()


        mask = legal_action_mask(board)

        legal_logits = policy_logits[mask.to(self.device)]
        legal_priors = torch.softmax(legal_logits, dim=0)

        legal_indices = torch.where(mask)[0]

        # Один перенос GPU -> CPU вместо .item() для каждого хода
        legal_indices = legal_indices.tolist()
        legal_priors = legal_priors.cpu().tolist()

        for index, prior in zip(legal_indices, legal_priors):
            move = decode_action(index, board)

            child_board = board.copy(stack=False)
            child_board.push(move)

            # prior уже обычный Python float
            prior_value = prior

            child = Node(
                board=child_board,
                parent=node,
                prior=prior_value,
                action=index,
            )

            node.children[index] = child

        return value

    def add_dirichlet_noise(self, root):
        children = list(root.children.values())

        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))

        epsilon = self.dirichlet_eps

        for child, noise_value in zip(children, noise):
            child.prior = (
                (1 - epsilon) * child.prior + epsilon * noise_value
            )




