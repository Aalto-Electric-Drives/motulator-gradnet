# Installation

The package requires Python 3.12 or newer, and it installs *motulator* and PyTorch as its dependencies. It supports *motulator* 0.15 and 0.16, and the support is extended after each new release of *motulator* has been tested.

## Option 1: Use pip

Install the package in a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). On Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install motulator-gradnet
```

On Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install motulator-gradnet
```

If PowerShell does not allow running the activation script, allow it with `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

````{tip}
On Linux, the PyTorch wheels on PyPI include CUDA support and are large. A CPU-only build is sufficient for the small networks of this package. To use it, install it in the virtual environment before the package:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```
````

The plots need a GUI library for *matplotlib*. On some Linux systems, it must be installed separately; see the [installation of *motulator*](https://aalto-electric-drives.github.io/motulator/installation.html).

## Option 2: Clone the Repository

Clone the repository to run the examples, which use the datasets and the trained models in `examples/`, or to develop the package. On Linux and macOS:

```bash
git clone https://github.com/Aalto-Electric-Drives/motulator-gradnet
cd motulator-gradnet
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,doc]"
```

On Windows, create and activate the virtual environment as in Option 1. The `dev` dependencies are needed for the tests and the code checks, and the `doc` dependencies for building this documentation. The CPU-only build of PyTorch can be installed before the package, as above. In [VS Code](https://code.visualstudio.com/docs/python/environments), the virtual environment can also be created with the Python extension.

Then run an example, e.g.:

```bash
python examples/plot_6kw_pmsyrm_flux_map_meas.py
```

The tests are run with `pytest`, and the documentation is built with the command:

```bash
sphinx-build -b html docs/source docs/build/html
```
