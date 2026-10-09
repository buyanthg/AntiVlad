import chess
import numpy as np
import torch

from mcts import MCTS, update_history
from encode_state import encode_state

from constants import ACTION_SIZE, HISTORY_LENGTH, TEMPERATURE_MOVES


def create_history(board):
    return [board.copy(stack=False) for _ in range(HISTORY_LENGTH)]

def get_result(board):
    result = board.result()

    if result == "1-0":
        return chess.WHITE

    if result == "0-1":
        return chess.BLACK

    return None


def create_training_data(states, policies, players, result):
    data = []

    for state, policy, player in zip(states, policies, players):
        if result is None:
            value = 0.0
        elif player == result:
            value = 1.0
        else:
            value = -1.0

        data.append(
            (state, torch.tensor(policy), torch.tensor(value, dtype=torch.float32))
        )

    return data


def get_policy(root):
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)

    total_visits = sum(
        child.visit_count
        for child in root.children.values()
    )

    if total_visits == 0:
        return policy

    for action, child in root.children.items():
        policy[action] = child.visit_count / total_visits

    return policy


def select_action(root, move_number):
    temperature = 1.0 if move_number < TEMPERATURE_MOVES else 0.0
    actions = list(root.children.keys())

    visits = np.array([root.children[action].visit_count for action in actions], dtype=np.float64)

    if temperature == 0.0:
        return actions[np.argmax(visits)]

    probabilities = visits ** (1.0 / temperature)
    probabilities /= probabilities.sum()

    return np.random.choice(actions, p=probabilities)


class SelfPlay:
    def __init__(self, model, simulations=100, c_puct=1.5):
        self.model = model
        self.mcts = MCTS(model, simulations, c_puct)

    def play_game(self):
        board = chess.Board()
        history = create_history(board)

        states = []
        policies = []
        players = []

        move_number = 0

        while not board.is_game_over():
            root = self.mcts.search(history)

            state = encode_state(history)

            policy = get_policy(root)

            states.append(state)
            policies.append(policy)
            players.append(board.turn)

            action = select_action(root, move_number)
            child = root.children[action]
            board = child.board

            history = update_history(history, board)

            move_number += 1

        result = get_result(board)

        return create_training_data(
            states,
            policies,
            players,
            result
        )


