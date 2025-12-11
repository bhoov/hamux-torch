"""Tests for hamux_torch.lagrangians module."""

import pytest
import torch
from torch.func import grad

from hamux_torch.lagrangians import (
    lagr_identity,
    lagr_repu,
    lagr_relu,
    lagr_exp,
    lagr_rexp,
    lagr_tanh,
    lagr_sigmoid,
    lagr_softmax,
    lagr_layernorm,
    lagr_spherical_norm,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def x():
    """Random test tensor."""
    return torch.randn(10)


@pytest.fixture
def x_positive():
    """Positive test tensor for relu-like functions."""
    return torch.abs(torch.randn(10)) + 0.1


# ============================================================================
# lagr_identity Tests
# ============================================================================

class TestLagrIdentity:
    def test_output_scalar(self, x):
        result = lagr_identity(x)
        assert result.shape == ()

    def test_formula(self, x):
        result = lagr_identity(x)
        expected = 0.5 * torch.pow(x, 2).sum()
        assert torch.allclose(result, expected)

    def test_gradient_is_identity(self, x):
        g = grad(lagr_identity)(x)
        assert torch.allclose(g, x)


# ============================================================================
# lagr_repu Tests
# ============================================================================

class TestLagrRepu:
    def test_output_scalar(self, x):
        result = lagr_repu(x, n=3)
        assert result.shape == ()

    def test_zero_for_negative(self):
        x = torch.tensor([-1.0, -2.0, -3.0])
        result = lagr_repu(x, n=3)
        assert result == 0.0

    def test_positive_values(self, x_positive):
        result = lagr_repu(x_positive, n=3)
        expected = (1/3) * torch.relu(x_positive).pow(3).sum()
        assert torch.allclose(result, expected)


# ============================================================================
# lagr_relu Tests
# ============================================================================

class TestLagrRelu:
    def test_output_scalar(self, x):
        result = lagr_relu(x)
        assert result.shape == ()

    def test_is_repu_degree_2(self, x):
        result = lagr_relu(x)
        expected = lagr_repu(x, n=2)
        assert torch.allclose(result, expected)


# ============================================================================
# lagr_exp Tests
# ============================================================================

class TestLagrExp:
    def test_output_scalar(self, x):
        result = lagr_exp(x)
        assert result.shape == ()

    def test_formula(self, x):
        beta = 2.0
        result = lagr_exp(x, beta=beta)
        expected = (1/beta) * torch.exp(beta * x).sum()
        assert torch.allclose(result, expected)

    def test_gradient_is_exp(self, x):
        beta = 1.0
        g = grad(lambda t: lagr_exp(t, beta=beta))(x)
        expected = torch.exp(beta * x)
        assert torch.allclose(g, expected)


# ============================================================================
# lagr_rexp Tests
# ============================================================================

class TestLagrRexp:
    def test_output_scalar(self, x):
        result = lagr_rexp(x)
        assert result.shape == ()

    def test_zero_region(self):
        x = torch.tensor([-1.0, -2.0])
        result = lagr_rexp(x, beta=1.0)
        # For negative x: exp(0)/beta - 0 = 1/beta
        expected = torch.tensor(2.0)  # 2 * (1/1)
        assert torch.allclose(result, expected)


# ============================================================================
# lagr_tanh Tests
# ============================================================================

class TestLagrTanh:
    def test_output_scalar(self, x):
        result = lagr_tanh(x)
        assert result.shape == ()

    def test_gradient_is_tanh(self, x):
        beta = 1.0
        g = grad(lambda t: lagr_tanh(t, beta=beta))(x)
        expected = torch.tanh(beta * x)
        assert torch.allclose(g, expected)

    def test_gradient_with_beta(self, x):
        beta = 2.0
        g = grad(lambda t: lagr_tanh(t, beta=beta))(x)
        expected = torch.tanh(beta * x)
        assert torch.allclose(g, expected)


# ============================================================================
# lagr_sigmoid Tests
# ============================================================================

class TestLagrSigmoid:
    def test_output_scalar(self, x):
        result = lagr_sigmoid(x)
        assert result.shape == ()

    def test_gradient_is_sigmoid(self, x):
        beta = 1.0
        g = grad(lambda t: lagr_sigmoid(t, beta=beta))(x)
        expected = torch.sigmoid(beta * x)
        assert torch.allclose(g, expected)

    def test_gradient_with_beta(self, x):
        beta = 2.0
        g = grad(lambda t: lagr_sigmoid(t, beta=beta))(x)
        expected = torch.sigmoid(beta * x)
        assert torch.allclose(g, expected)


# ============================================================================
# lagr_softmax Tests
# ============================================================================

class TestLagrSoftmax:
    def test_output_scalar(self, x):
        result = lagr_softmax(x)
        assert result.shape == ()

    def test_formula(self, x):
        beta = 2.0
        result = lagr_softmax(x, beta=beta)
        expected = (1/beta) * torch.logsumexp(beta * x, dim=-1)
        assert torch.allclose(result, expected)

    def test_gradient_is_softmax(self, x):
        beta = 1.0
        g = grad(lambda t: lagr_softmax(t, beta=beta))(x)
        expected = torch.softmax(beta * x, dim=-1)
        assert torch.allclose(g, expected)


# ============================================================================
# lagr_layernorm Tests
# ============================================================================

class TestLagrLayernorm:
    def test_output_scalar(self, x):
        result = lagr_layernorm(x)
        assert result.shape == ()

    def test_with_delta(self, x):
        delta = torch.randn(10)
        result = lagr_layernorm(x, delta=delta)
        assert result.shape == ()


# ============================================================================
# lagr_spherical_norm Tests
# ============================================================================

class TestLagrSphericalNorm:
    def test_output_scalar(self, x):
        result = lagr_spherical_norm(x)
        assert result.shape == ()

    def test_formula(self, x):
        gamma = 2.0
        eps = 1e-5
        result = lagr_spherical_norm(x, gamma=gamma, eps=eps)
        y = torch.sqrt(torch.pow(x, 2).sum(dim=-1, keepdim=True) + eps)
        expected = (gamma * y).sum()
        assert torch.allclose(result, expected)

    def test_with_delta(self, x):
        delta = torch.randn(10)
        result = lagr_spherical_norm(x, delta=delta)
        assert result.shape == ()
