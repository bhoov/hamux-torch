"""Tests for hamux_torch.core module."""

import pytest
import torch
from torch.func import grad
import functools as ft

from hamux_torch.core import (
    legendre_transform,
    IdentityLagrangian,
    NeuronLayer,
    LinearSynapse,
    HAM,
    VectorizedHAM,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def identity_lagrangian():
    return IdentityLagrangian()


@pytest.fixture
def simple_neuron_layer():
    """A simple neuron layer with identity lagrangian."""
    return NeuronLayer(IdentityLagrangian(), shape=(10,))


@pytest.fixture
def simple_ham():
    """A simple HAM with two neurons and one synapse."""
    neurons = {
        "x": NeuronLayer(IdentityLagrangian(), shape=(8,)),
        "y": NeuronLayer(IdentityLagrangian(), shape=(4,)),
    }
    synapses = {
        "W": LinearSynapse(8, 4),
    }
    connections = [
        (("x", "y"), "W"),
    ]
    return HAM(neurons, synapses, connections)


# ============================================================================
# IdentityLagrangian Tests
# ============================================================================

class TestIdentityLagrangian:
    def test_forward(self, identity_lagrangian):
        x = torch.randn(10)
        result = identity_lagrangian(x)
        expected = 0.5 * (x**2).sum()
        assert torch.allclose(result, expected)

    def test_gradient_is_identity(self, identity_lagrangian):
        x = torch.randn(10)
        g = grad(identity_lagrangian)(x)
        assert torch.allclose(g, x)


# ============================================================================
# legendre_transform Tests
# ============================================================================

class TestLegendreTransform:
    def test_identity_lagrangian(self, identity_lagrangian):
        x = torch.randn(10)
        xhat = x.clone()  # For identity, activation = state
        Fhat = legendre_transform(identity_lagrangian)
        result = Fhat(xhat, x)
        # Legendre: xhat * x - F(x) = x * x - 0.5 * x^2 = 0.5 * x^2
        expected = 0.5 * (x**2).sum()
        assert torch.allclose(result, expected)

    def test_gradient_flow(self, identity_lagrangian):
        x = torch.randn(10, requires_grad=True)
        xhat = torch.randn(10, requires_grad=True)
        Fhat = legendre_transform(identity_lagrangian)
        result = Fhat(xhat, x)
        result.backward()
        assert x.grad is not None
        assert xhat.grad is not None


# ============================================================================
# NeuronLayer Tests
# ============================================================================

class TestNeuronLayer:
    def test_init_no_batch(self, simple_neuron_layer):
        x = simple_neuron_layer.init()
        assert x.shape == (10,)
        assert torch.all(x == 0)

    def test_init_with_batch(self, simple_neuron_layer):
        x = simple_neuron_layer.init(bs=5)
        assert x.shape == (5, 10)
        assert torch.all(x == 0)

    def test_activations_identity(self, simple_neuron_layer):
        x = torch.randn(10)
        xhat = simple_neuron_layer.activations(x)
        # Identity lagrangian: gradient is x itself
        assert torch.allclose(xhat, x)

    def test_energy(self, simple_neuron_layer):
        x = torch.randn(10)
        xhat = simple_neuron_layer.activations(x)
        energy = simple_neuron_layer.energy(xhat, x)
        # For identity: E = xhat*x - 0.5*x^2 = x*x - 0.5*x^2 = 0.5*x^2
        expected = 0.5 * (x**2).sum()
        assert torch.allclose(energy, expected)

    def test_aliases(self, simple_neuron_layer):
        assert simple_neuron_layer.E == simple_neuron_layer.energy
        assert simple_neuron_layer.sigma == simple_neuron_layer.activations

    def test_shape_int_converted_to_tuple(self):
        layer = NeuronLayer(IdentityLagrangian(), shape=10)
        assert layer.shape == (10,)


# ============================================================================
# LinearSynapse Tests
# ============================================================================

class TestLinearSynapse:
    def test_forward_shape(self):
        synapse = LinearSynapse(8, 4)
        xhat1 = torch.randn(8)
        xhat2 = torch.randn(4)
        energy = synapse(xhat1, xhat2)
        assert energy.shape == ()  # scalar

    def test_forward_computation(self):
        synapse = LinearSynapse(3, 2)
        xhat1 = torch.ones(3)
        xhat2 = torch.ones(2)
        energy = synapse(xhat1, xhat2)
        # -einsum("c,d,cd->", xhat1, xhat2, W) = -sum(W)
        expected = -synapse.W.sum()
        assert torch.allclose(energy, expected)


# ============================================================================
# HAM Tests
# ============================================================================

class TestHAM:
    def test_init_states(self, simple_ham):
        xs = simple_ham.init_states()
        assert "x" in xs and "y" in xs
        assert xs["x"].shape == (8,)
        assert xs["y"].shape == (4,)

    def test_activations(self, simple_ham):
        xs = simple_ham.init_states()
        xs["x"] = torch.randn(8)
        xs["y"] = torch.randn(4)
        xhats = simple_ham.activations(xs)
        # Identity lagrangian: activations = states
        assert torch.allclose(xhats["x"], xs["x"])
        assert torch.allclose(xhats["y"], xs["y"])

    def test_energy_returns_scalar(self, simple_ham):
        xs = {"x": torch.randn(8), "y": torch.randn(4)}
        xhats = simple_ham.activations(xs)
        energy = simple_ham.energy(xhats, xs)
        assert energy.shape == ()

    def test_energy_tree(self, simple_ham):
        xs = {"x": torch.randn(8), "y": torch.randn(4)}
        xhats = simple_ham.activations(xs)
        tree = simple_ham.energy_tree(xhats, xs)
        assert "neurons" in tree
        assert "connections" in tree
        assert "x" in tree["neurons"]
        assert "y" in tree["neurons"]

    def test_dEdact(self, simple_ham):
        xs = {"x": torch.randn(8), "y": torch.randn(4)}
        xhats = simple_ham.activations(xs)
        grads = simple_ham.dEdact(xhats, xs)
        assert "x" in grads and "y" in grads

    def test_dEdact_with_energy(self, simple_ham):
        xs = {"x": torch.randn(8), "y": torch.randn(4)}
        xhats = simple_ham.activations(xs)
        grads, energy = simple_ham.dEdact(xhats, xs, return_energy=True)
        assert "x" in grads and "y" in grads
        assert energy.shape == ()

    def test_properties(self, simple_ham):
        assert simple_ham.n_neurons == 2
        assert simple_ham.n_hypersynapses == 1
        assert simple_ham.n_connections == 1

    def test_vectorize_returns_vectorized_ham(self, simple_ham):
        vham = simple_ham.vectorize()
        assert isinstance(vham, VectorizedHAM)

    def test_unvectorize_returns_self(self, simple_ham):
        assert simple_ham.unvectorize() is simple_ham


# ============================================================================
# VectorizedHAM Tests
# ============================================================================

class TestVectorizedHAM:
    def test_init_states_batched(self, simple_ham):
        vham = simple_ham.vectorize()
        xs = vham.init_states(bs=16)
        assert xs["x"].shape == (16, 8)
        assert xs["y"].shape == (16, 4)

    def test_activations_batched(self, simple_ham):
        vham = simple_ham.vectorize()
        xs = {"x": torch.randn(16, 8), "y": torch.randn(16, 4)}
        xhats = vham.activations(xs)
        assert xhats["x"].shape == (16, 8)
        assert xhats["y"].shape == (16, 4)

    def test_energy_batched(self, simple_ham):
        vham = simple_ham.vectorize()
        xs = {"x": torch.randn(16, 8), "y": torch.randn(16, 4)}
        xhats = vham.activations(xs)
        energy = vham.energy(xhats, xs)
        assert energy.shape == (16,)

    def test_dEdact_batched(self, simple_ham):
        vham = simple_ham.vectorize()
        xs = {"x": torch.randn(16, 8), "y": torch.randn(16, 4)}
        xhats = vham.activations(xs)
        grads = vham.dEdact(xhats, xs)
        assert grads["x"].shape == (16, 8)
        assert grads["y"].shape == (16, 4)

    def test_unvectorize_returns_ham(self, simple_ham):
        vham = simple_ham.vectorize()
        ham = vham.unvectorize()
        assert ham is simple_ham

    def test_vectorize_returns_self(self, simple_ham):
        vham = simple_ham.vectorize()
        assert vham.vectorize() is vham

    def test_vmap_matches_loop(self, simple_ham):
        """Verify batched computation matches individual loop computation."""
        vham = simple_ham.vectorize()
        ham = simple_ham

        batch_size = 4
        xs_batch = {"x": torch.randn(batch_size, 8), "y": torch.randn(batch_size, 4)}

        # Batched computation
        xhats_batch = vham.activations(xs_batch)
        energies_batch = vham.energy(xhats_batch, xs_batch)

        # Loop computation
        energies_loop = []
        for i in range(batch_size):
            xs_i = {"x": xs_batch["x"][i], "y": xs_batch["y"][i]}
            xhats_i = ham.activations(xs_i)
            energy_i = ham.energy(xhats_i, xs_i)
            energies_loop.append(energy_i)
        energies_loop = torch.stack(energies_loop)

        assert torch.allclose(energies_batch, energies_loop)
