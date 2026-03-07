import torch
from torch import Tensor


class NullSpaceDecomposition:
    def __init__(self):
        self.w0: Tensor | None = None
        self.null_basis: Tensor | None = None

    def fit(self, Phi: Tensor, y: Tensor) -> None:
        N, K = Phi.shape
        if K <= N:
            raise ValueError(
                f"Require K > N for an underdetermined system, got K={K}, N={N}"
            )

        with torch.no_grad():
            # Full SVD: Phi = U @ diag(S) @ Vh, U:(N,N), S:(N,), Vh:(K,K)
            U, S, Vh = torch.linalg.svd(Phi, full_matrices=True)

            # Minimum-norm particular solution: w0 = Vh[:N].T @ diag(1/S) @ U.T @ y
            self.w0 = Vh[:N].T @ (torch.diag(1.0 / S) @ (U.T @ y))

            # Null-space basis: last K-N rows of Vh, transposed -> (K, K-N)
            # Phi @ null_basis = U @ diag(S) @ Vh[:N,:] @ Vh[N:].T = 0 (orthonormality of Vh)
            self.null_basis = Vh[N:].T
