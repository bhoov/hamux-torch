"""The well-behaved dynamics of associative memories is described by the Lagrangians of the neurons."""

__all__ = ['lagr_identity', 'lagr_repu', 'lagr_relu', 'lagr_exp', 'lagr_rexp', 'lagr_tanh', 'lagr_sigmoid',
           'lagr_softmax', 'lagr_layernorm', 'lagr_spherical_norm']

import torch
import torch.nn.functional as F
from typing import Union

Scalar = torch.Tensor
Tensor = torch.Tensor

def lagr_identity(x: torch.Tensor) -> torch.Tensor:
    """The Lagrangian whose activation function is simply the identity."""
    return 0.5 * torch.pow(x, 2).sum()

def lagr_repu(x: torch.Tensor, n: float) -> torch.Tensor:
    """Rectified Power Unit of degree `n`."""
    return (1 / n) * torch.relu(x).pow(n).sum()

def lagr_relu(x: torch.Tensor) -> torch.Tensor:
    """Rectified Linear Unit. Same as `lagr_repu` of degree 2."""
    return lagr_repu(x, 2)

def lagr_exp(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Exponential activation function, as in [Demicirgil et al.](https://arxiv.org/abs/1702.01929)."""
    return (1 / beta) * torch.exp(beta * x).sum()

def lagr_rexp(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Lagrangian of the rectified exponential activation function."""
    xclipped = torch.relu(x)
    return (torch.exp(beta * xclipped) / beta - xclipped).sum()

class _LagrangianTanhFunction(torch.autograd.Function):
    """Custom autograd function for tanh Lagrangian with custom gradient."""
    
    @staticmethod
    def forward(x, beta):
        return (1 / beta) * torch.log(torch.cosh(beta * x))
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, beta = inputs
        ctx.save_for_backward(x)
        ctx.beta = beta
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        beta = ctx.beta
        grad_x = grad_output * torch.tanh(beta * x)
        return grad_x, None

def lagr_tanh(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Lagrangian of the tanh activation function."""
    return _LagrangianTanhFunction.apply(x, beta).sum()

class _LagrangianSigmoidFunction(torch.autograd.Function):
    """Custom autograd function for sigmoid Lagrangian with custom gradient."""
    
    @staticmethod
    def forward(x, beta):
        return (1.0 / beta) * torch.log(torch.exp(beta * x) + 1)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, beta = inputs
        ctx.save_for_backward(x)
        ctx.beta = beta
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        beta = ctx.beta
        grad_x = grad_output * torch.sigmoid(beta * x)
        return grad_x, None

def lagr_sigmoid(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """The lagrangian of the sigmoid activation function."""
    return _LagrangianSigmoidFunction.apply(x, beta).sum()

def lagr_softmax(x: torch.Tensor, beta: float = 1.0, dim: int = -1) -> torch.Tensor:
    """The lagrangian of the softmax -- the logsumexp."""
    return (1 / beta) * torch.logsumexp(beta * x, dim=dim, keepdim=False)

def lagr_layernorm(
    x: torch.Tensor,
    gamma: float = 1.0,
    delta: Union[float, torch.Tensor] = 0.0,
    dim: int = -1,
    eps: float = 1e-5
) -> torch.Tensor:
    """Lagrangian of the layer norm activation function. `gamma` must be a float, not a vector."""
    D = x.shape[dim] if dim is not None else x.numel()
    xmean = x.mean(dim=dim, keepdim=True)
    xmeaned = x - xmean
    y = torch.sqrt(torch.pow(xmeaned, 2).mean(dim=dim, keepdim=True) + eps)
    return (D * gamma * y + (delta * x).sum()).sum()

def lagr_spherical_norm(
    x: torch.Tensor,
    gamma: float = 1.0,
    delta: Union[float, torch.Tensor] = 0.0,
    dim: int = -1,
    eps: float = 1e-5
) -> torch.Tensor:
    """Lagrangian of the spherical norm (L2 norm) activation function."""
    y = torch.sqrt(torch.pow(x, 2).sum(dim=dim, keepdim=True) + eps)
    return (gamma * y + (delta * x).sum()).sum()