# hamux-torch

An implementation of HAMUX using pytorch. Please see the original [`hamux`](https://github.com/bhoov/hamux) JAX package for motivation and description.

## Getting started

Clone and install dependencies

```sh
git clone git@github.com:bhoov/hamux-torch.git
uv sync
```

Then run `example_mnist.ipynb`. You may need to install the ipykernel for the env with

```sh
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=hamux-torch
```

## Testing

```sh
make test
```

Other test commands: `make test-quick`, `make test-cov`