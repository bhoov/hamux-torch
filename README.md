# hamux-torch

An implementation of HAMUX using pytorch. Please see the original [`hamux`](https://github.com/bhoov/hamux) JAX package for motivation and description.

## Getting started

Install from github:

``` sh
pip install git+https://github.com/bhoov/hamux.git@main
```

Or clone and install dependencies

```sh
git clone git@github.com:bhoov/hamux-torch.git
uv sync
```

Then run `example_mnist.ipynb`. You may need to install the ipykernel for the env with 

```
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=hamux-torch
```
