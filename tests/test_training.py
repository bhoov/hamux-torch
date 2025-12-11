"""Minimal training test to verify complete training logic works."""

import pytest
import torch
from torch import nn

import hamux_torch as hmx


class DenseSynapseHid(nn.Module):
    """Dense Associative Memory synapse (from MNIST example)."""
    def __init__(self, x1_dim: int, x2_dim: int, beta: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(x1_dim, x2_dim) * 0.02)
        self.beta = beta

    @property
    def nW(self):
        nc = torch.sqrt(torch.sum(self.W ** 2, dim=0, keepdim=True))
        return self.W / nc

    def forward(self, xhat1):
        x2 = xhat1 @ self.nW
        return -1 / self.beta * torch.logsumexp(self.beta * x2, dim=-1)


@pytest.fixture
def simple_ham():
    """Small HAM for fast training test."""
    neurons = {
        "x": hmx.NeuronLayer(hmx.lagr_spherical_norm, (16,)),
    }
    synapses = {
        "s": DenseSynapseHid(16, 8),
    }
    connections = [
        (["x"], "s"),
    ]
    return hmx.HAM(neurons, synapses, connections)


def test_training_loop(simple_ham):
    """Test that a minimal training loop runs without errors."""
    torch.manual_seed(42)

    ham = simple_ham.vectorize()
    opt = torch.optim.Adam(ham.parameters(), lr=1e-2)

    # Tiny synthetic dataset
    batch_size = 4
    data = torch.randn(batch_size, 16)
    data = data / data.norm(dim=-1, keepdim=True)  # normalize

    # Training params
    n_steps = 2
    alpha = 0.5

    initial_loss = None
    final_loss = None

    # Mini training loop (3 iterations)
    for iteration in range(3):
        opt.zero_grad()

        # Init states with noisy data
        xs = ham.init_states(bs=batch_size)
        xs["x"] = data + torch.randn_like(data) * 0.1

        # Energy descent
        for _ in range(n_steps):
            xhats = ham.activations(xs)
            egrad, energy = ham.dEdact(xhats, xs, return_energy=True)
            for key in xs:
                xs[key] = xs[key] - alpha * egrad[key]

        # Reconstruction loss
        xhats = ham.activations(xs)
        loss = ((xhats["x"] - data) ** 2).mean()

        if initial_loss is None:
            initial_loss = loss.item()
        final_loss = loss.item()

        loss.backward()
        opt.step()

    # Verify training ran
    assert initial_loss is not None
    assert final_loss is not None
    # Loss should be finite
    assert torch.isfinite(torch.tensor(final_loss))


def test_gradient_flow(simple_ham):
    """Test that gradients flow through the HAM."""
    ham = simple_ham.vectorize()

    xs = ham.init_states(bs=2)
    xs["x"] = torch.randn(2, 16, requires_grad=True)

    xhats = ham.activations(xs)
    energy = ham.energy(xhats, xs)
    total_energy = energy.sum()
    total_energy.backward()

    # Check gradients exist on synapse weights
    assert ham.hypersynapses["s"].W.grad is not None
    assert not torch.all(ham.hypersynapses["s"].W.grad == 0)
