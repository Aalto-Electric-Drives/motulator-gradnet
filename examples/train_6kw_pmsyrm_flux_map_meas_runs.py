"""
5.6-kW PM-SyRM, repeat training, measured data
==============================================

This example evaluates how the selection of the training points affects a GradNet
flux-linkage map of a 5.6-kW PM synchronous reluctance machine (Baldor
ECS101M0H7EF4). Every 10th point of the measured dataset is used for training, and
the training is repeated for each of the 10 possible starting points of this
selection. The seed of the random number generators, which determines the initial
weights, also differs between the runs. The mean and the standard deviation of the
validation errors over the runs are printed. The same procedure can be used for current
maps and other datasets. The 10 training runs take a few minutes in total.

"""

from pathlib import Path
from tempfile import TemporaryDirectory

import motulator.drive.gradnet as gn
import numpy as np
from motulator.drive import utils

from motulator_gradnet import PNormGradient, get_training_data, train_gradnet

# %%
# Compute base values based on the nominal values.

nom = utils.NominalValues(U=460, I=8.8, f=60, P=5.6e3, tau=29.7)
base = utils.BaseValues.from_nominal(nom, n_p=2)

# %%
# Set up the paths and parameters.

p = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

dataset_path = p / "datasets/baldor_meas.npz"
subsample = 10

# %%
# Train the model for each starting point and compute the errors of the flux linkage
# in the validation points, i.e., in the points not used in the training.

e_rms, e_max = [], []
with TemporaryDirectory() as tmp:
    for offset in range(subsample):
        model_path = Path(tmp) / f"flux_map_{offset}.npz"
        train_gradnet(
            dataset_path=dataset_path,
            base=base,
            save_model_path=model_path,
            is_flux_map=True,
            embed_dim=12,
            epochs=20000,
            subsample=subsample,
            activation=PNormGradient,
            subsample_offset=offset,
            seed=offset,
        )
        flux_map_fcn = gn.FluxMap(gn.load_gradnet(model_path))
        _, (psi_s_dq, i_s_dq) = get_training_data(
            dataset_path, base=base, subsample=subsample, subsample_offset=offset
        )
        err = np.abs(flux_map_fcn(i_s_dq) - psi_s_dq) / base.psi
        e_rms.append(np.sqrt(np.mean(err**2)))
        e_max.append(np.max(err))

# %%
# Print the mean and the standard deviation of the errors over the runs.

print(f"RMS error: {np.mean(e_rms):.3f} ± {np.std(e_rms):.3f} p.u.")
print(f"Max error: {np.mean(e_max):.3f} ± {np.std(e_max):.3f} p.u.")
