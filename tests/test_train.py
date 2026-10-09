import torch

from antivlad_net import AntiVladNet
from constants import TOTAL_PLANES
from train import Trainer
from action_encoder import ACTION_SIZE


def create_model():
    return AntiVladNet(
        out_channels=64,
        policy_channels=32,
        value_channels=32,
        value_hidden=256,
        num_residual_blocks=5
    )


def create_data(num_samples=8):
    states = torch.randn(num_samples, TOTAL_PLANES, 8, 8)

    policies = torch.zeros(num_samples, ACTION_SIZE)

    # Простая валидная policy target
    for i in range(num_samples):
        policies[i, i] = 1.0

    values = torch.tensor(
        [1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0][:num_samples]
    )

    return list(zip(states, policies, values))


def test_trainer():
    model = create_model()

    trainer = Trainer(
        model=model,
        learning_rate=3e-4,
        batch_size=4
    )

    data = create_data()

    trainer.train(data, epochs=1)


def test_parameters_change():
    model = create_model()

    trainer = Trainer(
        model=model,
        learning_rate=3e-4,
        batch_size=4
    )

    data = create_data()

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    trainer.train(data, epochs=1)

    changed = False

    for name, parameter in model.named_parameters():
        if not torch.equal(before[name], parameter):
            changed = True
            break

    assert changed


def test_multiple_epochs():
    model = create_model()

    trainer = Trainer(
        model=model,
        learning_rate=3e-4,
        batch_size=4
    )

    data = create_data()

    trainer.train(data, epochs=3)