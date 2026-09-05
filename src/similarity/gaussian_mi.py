"""Mutual information under the joint-Gaussian assumption (MKA paper, Sec. 3.3).

For two k-dim Gaussian random vectors the mutual information has a closed form
involving only covariance determinants, so it is cheap and stable:

  I(Psi_l; Psi_m) = 0.5 * ln( |Sigma_l| * |Sigma_m| / |Sigma_l,m| )
  H(Psi)          = 0.5 * ln( (2 pi e)^k * |Sigma| )     (differential entropy)
  NMI             = I / sqrt( H(Psi_l) * H(Psi_m) )
"""

import numpy as np


def _standardize(psi: np.ndarray) -> np.ndarray:
    """Z-score each embedding dimension so covariance determinants stay
    well-conditioned (standard practice for Gaussian-MI estimation)."""
    mu = psi.mean(axis=0)
    sd = psi.std(axis=0)
    sd = np.where(sd < 1e-6, 1e-6, sd)
    return (psi - mu) / sd


def _log_det(S: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(S)
    return logdet if sign > 0 else float("-inf")


def gaussian_mi(psi_l: np.ndarray, psi_m: np.ndarray, eps: float = 1e-6) -> float:
    """MI between two (n, k) sample matrices, both Gaussian."""
    psi_l = _standardize(psi_l)
    psi_m = _standardize(psi_m)
    k = psi_l.shape[1]
    S_l = np.cov(psi_l, rowvar=False) + eps * np.eye(k)
    S_m = np.cov(psi_m, rowvar=False) + eps * np.eye(k)
    joint = np.hstack([psi_l, psi_m])
    S_lm = np.cov(joint, rowvar=False) + eps * np.eye(2 * k)
    return 0.5 * (_log_det(S_l) + _log_det(S_m) - _log_det(S_lm))


def gaussian_nmi(psi_l: np.ndarray, psi_m: np.ndarray,
                 eps: float = 1e-6, clip: float = 1.0) -> float:
    """Normalized MI in [0,1]. 1 = identical layers, 0 = unrelated."""
    psi_l = _standardize(psi_l)
    psi_m = _standardize(psi_m)
    k = psi_l.shape[1]
    mi = gaussian_mi(psi_l, psi_m, eps)
    logdet_l = _log_det(np.cov(psi_l, rowvar=False) + eps * np.eye(k))
    logdet_m = _log_det(np.cov(psi_m, rowvar=False) + eps * np.eye(k))
    H_l = 0.5 * (k * np.log(2 * np.pi * np.e) + logdet_l)
    H_m = 0.5 * (k * np.log(2 * np.pi * np.e) + logdet_m)
    if H_l <= 0 or H_m <= 0:
        return 0.0
    return float(np.clip(mi / np.sqrt(H_l * H_m), 0.0, clip))