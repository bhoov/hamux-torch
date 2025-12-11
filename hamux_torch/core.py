#%%
import torch
from torch import nn
import functools as ft
from torch import Tensor
from torch.func import vmap, grad, grad_and_value
import numpy as np

#%% Typing
from jaxtyping import Float
from typing import *
from warnings import warn

StateTensor = Float[Tensor, "..."]
ActivationTensor = Float[Tensor, "..."]
Scalar = Float[Tensor, ""]

#%% Lagrangians
class LegendreTransform(torch.autograd.Function):
    # Enable automatic vmap rule generation for torch.func transforms
    generate_vmap_rule = True
    
    @staticmethod
    def forward(F, xhat, x):
        return (xhat * x).sum() - F(x)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        F, xhat, x = inputs
        ctx.save_for_backward(xhat, x)
    
    @staticmethod
    def backward(ctx, grad_output):
        xhat, x = ctx.saved_tensors
        grad_xhat = grad_output * x # Gradient w.r.t. xhat is x
        grad_x = grad_output * xhat # Gradient w.r.t. x is xhat  
        return None, grad_xhat, grad_x

def legendre_transform(F):
    """Transform scalar F(x) into the dual Fhat(xhat, x) using the Legendre transform"""
    
    def Fhat(xhat, x):
        return LegendreTransform.apply(F, xhat, x)
    
    return Fhat

# For example
class IdentityLagrangian(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: StateTensor) -> Scalar:
        """Gradient of this is identity function"""
        return 0.5 * (x**2).sum()

# x = torch.randn(10)
# g = x.clone()
# Fhat = legendre_transform(IdentityLagrangian())
# Fhat(x, g)

#%% Neuron Layers
class NeuronLayer(nn.Module):
    """Neuron layers represent dynamic variables that evolve during inference (i.e., memory retrieval/error correction)"""
    lagrangian: Callable # The scalar-valued Lagrangian function:  x -> R
    shape: Tuple[int] # The shape of the neuron layer

    def __init__(self, 
                 lagrangian: nn.Module, # The scalar-valued Lagrangian function:  x -> R
                 shape: Tuple[int], # Shape of the neuron layer's state
                 ):
        super().__init__()
        self.lagrangian = lagrangian
        self.shape = (shape,) if isinstance(shape, int) else shape

        # aliases
        self.E = self.energy
        self.sigma = self.activations

    def activations(self, x: StateTensor) -> ActivationTensor:
        """Compute the activations of the neuron layer for a given input"""
        return grad(self.lagrangian)(x)
    
    def init(self, 
             bs: Optional[int] = None, # Batch size. If None, return a single state with no batch dim.
             ) -> StateTensor:
        if bs is None or bs == 0: return torch.zeros(self.shape)
        return torch.zeros((bs, *self.shape))
    
    def energy(self, xhat: ActivationTensor, x: StateTensor) -> Scalar:
        return legendre_transform(self.lagrangian)(xhat, x)

    def __repr__(self):
        return f"NeuronLayer(shape={self.shape}, lagrangian={self.lagrangian.__repr__()})"


# # Check nn.Module
# shape = (10,)
# layer = NeuronLayer(IdentityLagrangian(), shape)
# x = layer.init()
# g = layer.activations(x)
# assert x.shape == shape
# assert torch.allclose(g, x)

# # Check callable
# shape = (10,)
# layer = NeuronLayer(lambda x: 0.5 * (x**2).sum(), shape)
# x = layer.init()
# g = layer.activations(x)
# assert x.shape == shape
# assert torch.allclose(g, x)
