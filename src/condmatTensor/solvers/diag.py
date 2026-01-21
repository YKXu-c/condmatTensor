"""Diagonalization solver for band structure."""

from typing import Optional
import torch


def diagonalize(
    Hk: torch.Tensor,
    hermitian: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Diagonalize Hamiltonian at each k-point.

    H_k |ψ_{nk}⟩ = ε_{nk} |ψ_{nk}⟩

    Args:
        Hk: Hamiltonian in k-space, shape (N_k, N_orb, N_orb)
        hermitian: If True, use eigh (faster, assumes Hermitian).
                   If False, use eig (general diagonalization)

    Returns:
        eigenvalues: Eigenvalues ε_{nk}, shape (N_k, N_orb)
        eigenvectors: Eigenvectors, shape (N_k, N_orb, N_orb)
                     Column n corresponds to ε_{nk}
    """
    N_k, N_orb, _ = Hk.shape

    if hermitian:
        # Use torch.linalg.eigh for Hermitian matrices
        eigenvalues, eigenvectors = torch.linalg.eigh(Hk)
    else:
        # Use torch.linalg.eig for general matrices
        eigenvalues_complex, eigenvectors = torch.linalg.eig(Hk)
        eigenvalues = eigenvalues_complex.real

    return eigenvalues, eigenvectors
