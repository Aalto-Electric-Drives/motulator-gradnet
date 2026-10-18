"""Tests of the training and the datasets."""

from pathlib import Path

import motulator.drive.gradnet as gn
import numpy as np
import pytest
import torch
from motulator.drive.utils import BaseValues

from motulator_gradnet import PNormGradient, Softmax, get_training_data, train_gradnet

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
