# motulator-gradnet

**Training of gradient-network magnetic models of synchronous machines for [motulator](https://github.com/Aalto-Electric-Drives/motulator)**

[![PyPI version](https://img.shields.io/pypi/v/motulator-gradnet.svg)](https://pypi.org/project/motulator-gradnet/)

Documentation: https://aalto-electric-drives.github.io/motulator-gradnet/

This package trains gradient networks (GradNets) that model the current and flux linkage maps of synchronous machines, including spatial harmonics, from measured or finite-element data [[1]](#references). A GradNet is the gradient of a scalar state function, the magnetic energy or co-energy, so the map is a conservative vector field by construction [[2]](#references), and the monotonicity of the flux-linkage–current map is ensured. The networks are defined and trained with PyTorch. A trained network is saved as a NumPy function, which the magnetic maps of *motulator* evaluate in the machine models and control systems without PyTorch. The package also compares the trained maps with the data.

## Installation

Install the package from PyPI in a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) (*motulator* and PyTorch are installed as dependencies). On Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install motulator-gradnet
```

On Windows, the environment is created and activated with `py -m venv .venv` and `.venv\Scripts\activate`. The PyTorch wheels on PyPI for Linux include CUDA support and are large. The small networks of this package are trained in a reasonable time on a CPU, so a CPU-only build is sufficient. It is installed in the activated environment before the package with the command:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

The package supports *motulator* 0.15 and 0.16, and the support is extended after each new release of *motulator* has been tested. For development, clone the repository and install the package with `pip install -e .[dev]`.

## Usage

A dataset is an npz file with the complex arrays `i_s_dq` (stator current, A) and `psi_s_dq` (stator flux linkage, Vs) of the same shape, peak-valued space vectors in rotor coordinates. A dataset with spatial harmonics has also the arrays `theta_m` (electrical rotor angle, rad) and `tau_m` (electromagnetic torque per pole pair, Nm). The following example trains a flux linkage map from the measured dataset of a 5.6-kW PM synchronous reluctance machine in `examples/datasets/`:

```python
import motulator.drive.gradnet as gn
from motulator.drive import utils

from motulator_gradnet import (
    PNormGradient,
    get_training_data,
    print_flux_map_errors_meas,
    train_gradnet,
)

# Base values of the machine, used to normalize the data
nom = utils.NominalValues(U=460, I=8.8, f=60, P=5.6e3, tau=29.7)
base = utils.BaseValues.from_nominal(nom, n_p=2)

# Train a flux map from every 10th data point and save it
dataset_path = "examples/datasets/baldor_meas.npz"
train_gradnet(
    dataset_path,
    base,
    is_flux_map=True,
    embed_dim=6,
    epochs=20000,
    subsample=10,
    activation=PNormGradient,
    save_model_path="flux_map.npz",
)

# The trained map is evaluated with motulator, without PyTorch
flux_map = gn.FluxMap(gn.load_gradnet("flux_map.npz"))
print(flux_map(5 + 8j))  # Flux linkage (Vs) at the current (A)

# Compare with the data points not used in the training
_, val_data = get_training_data(dataset_path, base, subsample=10)
print_flux_map_errors_meas(flux_map, val_data, base)
```

The map is used in the machine models and control systems of *motulator* through its parameter classes, e.g., `SaturatedSynchronousMachinePars(n_p=2, R_s=0.63, psi_s_dq_fcn=flux_map)`. Complete examples, which also train current maps and maps with spatial harmonics, are in the [example gallery](https://aalto-electric-drives.github.io/motulator-gradnet/examples/index.html), and simulation examples using the trained maps are in the [documentation of *motulator*](https://aalto-electric-drives.github.io/motulator/drive_examples/index.html).

## References

1. Li, Foissner, Martin, Piippo, Hinkkanen, "Gradient networks for universal magnetic modeling of synchronous machines," 2026, https://arxiv.org/abs/2602.14947
1. Chaudhari, Pranav, Moura, "Gradient networks," IEEE Trans. Signal Process., 2025, https://doi.org/10.1109/TSP.2024.3496692

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->

<!-- prettier-ignore-start -->

<!-- markdownlint-disable -->

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Exiaolibur"><img src="https://avatars.githubusercontent.com/u/134478666?v=4?s=50" width="50px;" alt="Junyi Li"/><br /><sub><b>Junyi Li</b></sub></a><br /><a href="https://github.com/Aalto-Electric-Drives/motulator-gradnet/commits?author=Exiaolibur" title="Code">💻</a> <a href="#example-Exiaolibur" title="Examples">💡</a> <a href="#ideas-Exiaolibur" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://research.aalto.fi/en/persons/marko-hinkkanen"><img src="https://avatars.githubusercontent.com/u/76600872?v=4?s=50" width="50px;" alt="Marko Hinkkanen"/><br /><sub><b>Marko Hinkkanen</b></sub></a><br /><a href="https://github.com/Aalto-Electric-Drives/motulator-gradnet/commits?author=mhinkkan" title="Code">💻</a> <a href="#ideas-mhinkkan" title="Ideas, Planning, & Feedback">🤔</a> <a href="#example-mhinkkan" title="Examples">💡</a> <a href="#mentoring-mhinkkan" title="Mentoring">🧑‍🏫</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->

<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## License

MIT
