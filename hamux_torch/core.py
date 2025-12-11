#%%
__all__ = ["legendre_transform", "IdentityLagrangian", "NeuronLayer", "HAM", "VectorizedHAM"]

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


#%% Simple synapses
class LinearSynapse(nn.Module):
    """The energy synapse corrolary of the linear layer in standard neural networks"""
    def __init__(self, x1_dim: int, x2_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(x1_dim, x2_dim) * 0.02)

    def forward(self, xhat1: ActivationTensor, xhat2: ActivationTensor) -> Scalar:
        return -torch.einsum("...c,...d,cd->...", xhat1, xhat2, self.W)


# %%

StateCollection = Dict[str, StateTensor]
ActivationCollection = Dict[str, ActivationTensor]
Connection = Tuple[Tuple, str] # (('x', 'y', 'z'), 3) == "Connect neurons (X,Y,Z) via syanpse 3"

class HAM(nn.Module):
    """The HAMUX model"""
    def __init__(self, 
                 neurons: Dict[str, NeuronLayer],
                 hypersynapses: Dict[str, nn.Module],
                 connections: List[Connection],
                 ):
        super().__init__()
        self.neurons = nn.ModuleDict(neurons)
        self.hypersynapses = nn.ModuleDict(hypersynapses)
        self.connections = connections

        # aliases
        self.synapses = self.hypersynapses
        self.n_synapses = self.n_hypersynapses
        self.sigmas = self.activations

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_hypersynapses(self) -> int:
        return len(self.hypersynapses)

    @property
    def n_connections(self) -> int:
        return len(self.connections)

    def init_states(self, bs: Optional[int] = None) -> StateCollection:
        if bs is not None and bs > 0: warn("Vectorize with `ham.vectorize()` before processing batched states")
        xs = {k: v.init(bs) for k, v in self.neurons.items()}
        return xs

    def activations(self, xs: StateCollection) -> ActivationCollection:
        xhats = {k: v.sigma(xs[k]) for k, v in self.neurons.items()}
        return xhats

    def neuron_energies(self, xhats: ActivationCollection, xs: StateCollection) -> Dict[str, Scalar]:
        """Retrieve the energies of each neuron in the HAM"""
        return {k: self.neurons[k].energy(xhats[k], xs[k]) for k in self.neurons.keys()}

    def connection_energies(self, xhats: ActivationCollection) -> List[Scalar]:
        """Get the energy for each connection"""
        def get_energy(neuron_set, s):
            neighbor_xhats = [xhats[k] for k in neuron_set]
            return self.hypersynapses[s](*neighbor_xhats)
        return [get_energy(neuron_set, s) for neuron_set, s in self.connections]

    def energy_tree(self, xhats: ActivationCollection, xs: StateCollection) -> Dict[str, Dict[str, Scalar]]:
        """Return energies for each individual component in the HAM"""
        neuron_energies = self.neuron_energies(xhats, xs)
        connection_energies = self.connection_energies(xhats)
        return {"neurons": neuron_energies, "connections": connection_energies}

    def energy(self, xhats: ActivationCollection, xs: StateCollection) -> Scalar:
        """The complete energy of the HAM"""
        energy_tree = self.energy_tree(xhats, xs)
        neuron_energy = sum(energy_tree["neurons"].values())
        connection_energy = sum(energy_tree["connections"])
        return neuron_energy + connection_energy

    def dEdact(self, xhats: ActivationCollection, xs: StateCollection, return_energy: bool = False) -> Tuple[ActivationCollection, Scalar]:
        """Calculate gradient of system energy w.r.t. each activation"""
        if return_energy: 
            return grad_and_value(self.energy)(xhats, xs)
        return grad(self.energy)(xhats, xs)

    def __repr__(self):
        return f"HAM(neurons={list(self.neurons.keys())}, synapses={list(self.hypersynapses.keys())})"

    def unvectorize(self): return self
    def vectorize(self): return VectorizedHAM(self)

#%% Vectorized HAM
def _docstring_from(source_func):
    """Decorator that copies the docstring from source_func to the decorated function"""
    def decorator(target_func):
        if source_func.__doc__: target_func.__doc__ = source_func.__doc__
        return target_func
    return decorator

BatchStateTensor = Float[StateTensor, "batch"]
BatchActivationTensor = Float[ActivationTensor, "batch"]
BatchStateCollection = Dict[str, BatchStateTensor]
BatchScalar = Float[Scalar, "batch"]
BatchActivationCollection = Dict[str, BatchActivationTensor]

class VectorizedHAM(nn.Module):
    """Vectorized version of the HAM"""
    def __init__(self, ham: HAM):
        super().__init__()
        self._ham = ham

        # vmap axes
        self._batch_dims = {k: 0 for k in ham.neurons.keys()}

        # mapping
        self.neurons = self._ham.neurons
        self.hypersynapses = self._ham.hypersynapses
        self.connections = self._ham.connections

        self.n_neurons = self._ham.n_neurons
        self.n_hypersynapses = self._ham.n_hypersynapses
        self.n_connections = self._ham.n_connections

        # aliases
        self.synapses = self.hypersynapses
        self.n_synapses = self.n_hypersynapses
        self.sigmas = self.activations

    @_docstring_from(HAM.init_states)
    def init_states(self, bs: Optional[int] = None) -> BatchStateCollection:
        if bs is None or bs == 0: warn("This vectorized HAM should be initialized with `bs>0`. Call `ham.unvectorize()` if single inputs are desired instead.")
        xs = {k: v.init(bs) for k, v in self.neurons.items()}
        return xs

    @_docstring_from(HAM.activations)
    def activations(self, xs: BatchStateCollection) -> BatchActivationCollection:
        return vmap(self._ham.activations, in_dims=(self._batch_dims,))(xs)

    @_docstring_from(HAM.connection_energies)
    def connection_energies(self, xhats: BatchActivationCollection) -> List[Float[Tensor, "batch"]]: 
        return vmap(self._ham.connection_energies, in_dims=(self._batch_dims,))(xhats)

    @_docstring_from(HAM.neuron_energies)
    def neuron_energies(self, xhats: BatchActivationCollection, xs: BatchStateCollection) -> Dict[str, BatchScalar]: 
        return vmap(self._ham.neuron_energies, in_dims=(self._batch_dims, self._batch_dims))(xhats, xs)

    @_docstring_from(HAM.energy_tree)
    def energy_tree(self, xhats: BatchActivationCollection, xs: BatchStateCollection) -> Dict[str, Dict[str, BatchScalar]]:
        return vmap(self._ham.energy_tree, in_dims=(self._batch_dims, self._batch_dims))(xhats, xs)

    @_docstring_from(HAM.energy)
    def energy(self, xhats: BatchActivationCollection, xs: BatchStateCollection) -> BatchScalar:
        return vmap(self._ham.energy, in_dims=(self._batch_dims, self._batch_dims))(xhats, xs)

    @_docstring_from(HAM.dEdact)
    def dEdact(self, xhats: BatchActivationCollection, xs: BatchStateCollection, return_energy: bool = False) -> Tuple[BatchActivationCollection, BatchScalar]:
        return vmap(self._ham.dEdact, in_dims=(self._batch_dims, self._batch_dims, None))(xhats, xs, return_energy)

    def unvectorize(self): return self._ham
    def vectorize(self): return self