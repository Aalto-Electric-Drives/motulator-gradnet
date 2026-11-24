"""Tests of the training and the datasets."""

from pathlib import Path
from typing import cast

import motulator.drive.gradnet as gn
import numpy as np
import pytest
import torch
from motulator.drive.utils import BaseValues

from motulator_gradnet import (
    GradNet,
    PNormGradient,
    Softmax,
    get_training_data,
    train_gradnet,
)
from motulator_gradnet._dataset import BaseDataset, get_loader
from motulator_gradnet._train import Trainer

BASE = BaseValues(
    u=400.0,
    i=10.0,
    w=2 * np.pi * 50,
    psi=1.0,
    p=6e3,
    Z=40.0,
    L=0.1,
    C=1.0,
    tau=20.0,
    n_p=2,
)
PSI_F, L_D, L_Q = 0.5, 0.03, 0.07


def rms(x: np.ndarray) -> float:
    """Root-mean-square value."""
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def linear_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Save a dataset of a linear magnetic model on a current grid."""
    i_d, i_q = np.meshgrid(np.linspace(-10, 10, 21), np.linspace(-10, 10, 21))
    i_s_dq = i_d + 1j * i_q
    psi_s_dq = PSI_F + L_D * i_d + 1j * L_Q * i_q
    np.savez(path, i_s_dq=i_s_dq, psi_s_dq=psi_s_dq)
    return i_s_dq, psi_s_dq


def harmonic_dataset(path: Path, k: int) -> tuple[np.ndarray, ...]:
    """
    Save a dataset of a magnetic model with a spatial harmonic.

    The co-energy per pole pair divided by 1.5 is W_c = psi_f*i_d + L_d*i_d**2/2 +
    L_q*i_q**2/2 + a*|i|**2*cos(k*theta_m)/2, whose gradient gives the flux linkage
    and the torque. The harmonic torque component is up to 9 Nm (at |i| = 14 A).
    """
    a = 0.01
    i_d, i_q, theta_m = np.meshgrid(
        np.linspace(-10, 10, 9),
        np.linspace(-10, 10, 9),
        np.linspace(0, 2 * np.pi / k, 8, endpoint=False),
    )
    i_s_dq = i_d + 1j * i_q
    psi_s_dq = PSI_F + (L_D + a * np.cos(k * theta_m)) * i_d
    psi_s_dq = psi_s_dq + 1j * (L_Q + a * np.cos(k * theta_m)) * i_q
    dW_dtheta = -k * a * np.abs(i_s_dq) ** 2 * np.sin(k * theta_m) / 2
    tau_m = 1.5 * (np.imag(i_s_dq * np.conj(psi_s_dq)) + dW_dtheta)
    np.savez(path, i_s_dq=i_s_dq, psi_s_dq=psi_s_dq, theta_m=theta_m, tau_m=tau_m)
    return i_s_dq, psi_s_dq, theta_m, tau_m


@pytest.mark.parametrize(
    "activation, weight_low", [(PNormGradient, 1.0), (Softmax, 10.0)]
)
def test_train_flux_map(
    activation: type[torch.nn.Module], weight_low: float, tmp_path: Path
) -> None:
    """A flux map trained from a linear magnetic model reproduces it."""
    i_s_dq, psi_s_dq = linear_dataset(tmp_path / "data.npz")
    train_gradnet(
        tmp_path / "data.npz",
        BASE,
        is_flux_map=True,
        embed_dim=4,
        epochs=100,
        lr=1e-2,
        activation=activation,
        save_model_path=tmp_path / "flux_map.npz",
        i_low=5.0,
        weight_low=weight_low,
    )
    flux_map = gn.FluxMap(gn.load_gradnet(tmp_path / "flux_map.npz"))
    assert np.abs(flux_map(i_s_dq) - psi_s_dq).max() < 0.05 * BASE.psi


@pytest.mark.parametrize("is_flux_map", [True, False])
def test_train_map_with_harmonics(is_flux_map: bool, tmp_path: Path) -> None:
    """
    A map with harmonics learns the flux linkage or the current and the torque,
    including the harmonic torque component (3.1 Nm rms in the data; a wrong sign of
    the component in the training would give an rms error of about 6 Nm).
    """
    k = 6
    i_s_dq, psi_s_dq, theta_m, tau_m = harmonic_dataset(tmp_path / "data.npz", k)
    train_gradnet(
        tmp_path / "data.npz",
        BASE,
        is_flux_map=is_flux_map,
        k=k,
        embed_dim=8,
        epochs=500,
        lr=1e-2,
        activation=Softmax,
        save_model_path=tmp_path / "map.npz",
    )
    fcn = gn.load_gradnet(tmp_path / "map.npz")
    exp_j_theta_m = np.exp(1j * theta_m)
    if is_flux_map:
        out, tau_m_map = gn.FluxMapWithHarmonics(fcn, k)(i_s_dq, exp_j_theta_m)
        assert rms(out - psi_s_dq) < 0.1 * BASE.psi
    else:
        out, tau_m_map = gn.CurrentMapWithHarmonics(fcn, k)(psi_s_dq, exp_j_theta_m)
        assert rms(out - i_s_dq) < 0.1 * BASE.i
    assert rms(tau_m_map - tau_m) < 1.5


def test_get_training_data(tmp_path: Path) -> None:
    """The training data is every subsample-th point, the validation data the rest."""
    i_s_dq, psi_s_dq = linear_dataset(tmp_path / "data.npz")
    (psi_trn, i_trn), (psi_val, i_val) = get_training_data(
        str(tmp_path / "data.npz"), BASE, subsample=3
    )
    assert np.array_equal(psi_trn, psi_s_dq.flatten()[::3])
    assert np.array_equal(i_trn, i_s_dq.flatten()[::3])
    assert psi_trn.size + psi_val.size == psi_s_dq.size
    assert not np.isin(i_val, i_trn).any()


def test_get_training_data_offset(tmp_path: Path) -> None:
    """The subsampling offsets split the dataset into disjoint training sets."""
    i_s_dq, _ = linear_dataset(tmp_path / "data.npz")
    path = tmp_path / "data.npz"
    i_trn = [
        get_training_data(path, BASE, subsample=3, subsample_offset=offset)[0][1]
        for offset in range(3)
    ]
    assert np.array_equal(i_trn[1], i_s_dq.flatten()[1::3])
    assert sum(i.size for i in i_trn) == i_s_dq.size
    assert np.unique(np.concatenate(i_trn)).size == i_s_dq.size
    for offset in (-1, 3):
        with pytest.raises(ValueError, match="subsample_offset"):
            get_training_data(path, BASE, subsample=3, subsample_offset=offset)


def test_seed(tmp_path: Path) -> None:
    """The seed determines the trained model."""
    linear_dataset(tmp_path / "data.npz")
    weights = []
    for name, seed in [("a", 1), ("b", 1), ("c", 2)]:
        train_gradnet(
            tmp_path / "data.npz",
            BASE,
            is_flux_map=True,
            embed_dim=4,
            epochs=2,
            subsample=3,
            subsample_offset=1,
            seed=seed,
            save_model_path=tmp_path / f"{name}.npz",
        )
        weights.append(gn.load_gradnet(tmp_path / f"{name}.npz").W)
    assert np.array_equal(weights[0], weights[1])
    assert not np.array_equal(weights[0], weights[2])


@pytest.mark.parametrize("in_dim", [2, 4])
def test_symmetrized_output(in_dim: int, tmp_path: Path) -> None:
    """The output from which the loss is computed is symmetric about the d axis."""
    linear_dataset(tmp_path / "data.npz")
    loader = get_loader(tmp_path / "data.npz", BASE, k=None)
    torch.manual_seed(0)
    model = GradNet(in_dim=in_dim, activation=PNormGradient)
    with torch.no_grad():
        model.bias.copy_(torch.randn(in_dim))
    dataset = cast(BaseDataset, loader.dataset)
    trainer = Trainer(model, loader, dataset, "flux_map", None, lr=1e-3)
    x = torch.randn(16, in_dim)
    # The q-axis component and the sine of the angle change their signs
    conj = torch.tensor([1.0, -1.0]).repeat(in_dim // 2)
    with torch.no_grad():
        assert not torch.allclose(model(conj * x), conj * model(x))
        y, y_conj = trainer._output(x), trainer._output(conj * x)
    torch.testing.assert_close(y_conj, conj * y)


def test_symmetric_training(tmp_path: Path) -> None:
    """A map trained without the mirrored points is accurate for them."""
    i_s_dq, psi_s_dq = linear_dataset(tmp_path / "data.npz")
    keep = np.imag(i_s_dq) >= 0
    np.savez(tmp_path / "half.npz", i_s_dq=i_s_dq[keep], psi_s_dq=psi_s_dq[keep])
    train_gradnet(
        tmp_path / "half.npz",
        BASE,
        is_flux_map=True,
        embed_dim=4,
        epochs=300,
        lr=1e-2,
        save_model_path=tmp_path / "flux_map.npz",
    )
    flux_map = gn.FluxMap(gn.load_gradnet(tmp_path / "flux_map.npz"))
    assert np.abs(flux_map(i_s_dq) - psi_s_dq).max() < 0.05 * BASE.psi
