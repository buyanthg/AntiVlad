import torch
import math
from utils import board_to_tensor, move_to_idx


class Node:
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def q(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def ucb(self, parent_visits):
        return self.q() + 1.4 * self.prior * math.sqrt(parent_visits) / (1 + self.visits)

def run_inference(model, board):
    model.eval()
    with torch.no_grad():
        p_logits, v = model(board_to_tensor(board))

    # Оставляем тензор на устройстве, просто берем Softmax
    probs = torch.softmax(p_logits[0], dim=0)

    legal_moves = list(board.legal_moves)
    move_probs = {}
    probs_sum = 0.0

    for m in legal_moves:
        p = probs[move_to_idx(m)].item()  # .item() превращает тензор-скаляр в число Python
        move_probs[m] = p
        probs_sum += p

    if probs_sum > 0:
        move_probs = {m: p / probs_sum for m, p in move_probs.items()}
    else:
        move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}

    return move_probs, v.item()


def mcts_search(board, model, iterations=100):
    root = Node(board)

    for _ in range(iterations):
        node = root
        while node.is_expanded and not node.board.is_game_over():
            node = max(node.children.values(), key=lambda n: n.ucb(node.visits))

        if not node.board.is_game_over():
            policy, value = run_inference(model, node.board)
            node.is_expanded = True
            for m, p in policy.items():
                nb = node.board.copy()
                nb.push(m)
                node.children[m] = Node(nb, parent=node, move=m, prior=p)
        else:
            res = node.board.result()
            value = 1 if res == "1-0" else -1 if res == "0-1" else 0

        # Backprop
        v = value
        while node:
            node.visits += 1
            node.value_sum += v
            v = -v
            node = node.parent

    return root
