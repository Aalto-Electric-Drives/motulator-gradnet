"""
Training of gradient networks (GradNets) for the magnetic models of motulator.

GradNets model the current and flux linkage maps of synchronous machines, including
spatial harmonics, learned from measured or finite-element data [#Li2026]_. A GradNet
is the gradient of a scalar state function, the magnetic energy or co-energy, so the
map is a conservative vector field by construction [#Cha2025]_, and the monotonicity of
the flux-linkage--current map is also ensured. The networks are defined and trained
with PyTorch (`GradNet`, `train_gradnet`). A trained network is saved as the numpy
function `motulator.drive.gradnet.GradNetFunction`, which the magnetic maps of
motulator (e.g., `motulator.drive.gradnet.FluxMap`) evaluate without PyTorch. The
trained maps are compared with the data using the plotting and error functions.

References
----------
.. [#Li2026] Li, Foissner, Martin, Piippo, Hinkkanen, "Gradient networks for universal
   magnetic modeling of synchronous machines," 2026, https://arxiv.org/abs/2602.14947

.. [#Cha2025] Chaudhari, Pranav, Moura, "Gradient networks," IEEE Trans. Signal
   Process., 2025, https://doi.org/10.1109/TSP.2024.3496692

"""

from motulator_gradnet._dataset import get_training_data
from motulator_gradnet._gn import (
    ACTIVATIONS,
    AlgebraicSigmoid,
    GradNet,
    PNormGradient,
    Softmax,
    Squareplus,
    to_function,
)
from motulator_gradnet._plot import (
    PlotOptions,
    plot_output_vs_angle,
    plot_surface_vs_current_and_angle,
)
from motulator_gradnet._stats import (
    print_current_map_errors_fem,
    print_current_map_errors_meas,
    print_flux_map_errors_fem,
    print_flux_map_errors_meas,
)
from motulator_gradnet._train import train_gradnet

__all__ = [
    "ACTIVATIONS",
    "AlgebraicSigmoid",
    "GradNet",
    "PNormGradient",
    "PlotOptions",
    "Softmax",
    "Squareplus",
    "get_training_data",
    "plot_output_vs_angle",
    "plot_surface_vs_current_and_angle",
    "print_current_map_errors_fem",
    "print_current_map_errors_meas",
    "print_flux_map_errors_fem",
    "print_flux_map_errors_meas",
    "to_function",
    "train_gradnet",
]
