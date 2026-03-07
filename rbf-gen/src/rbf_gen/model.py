import torch
import torch.nn as nn
from torch import Tensor
from rbf_gen.rbf import RBFBasis
from rbf_gen.null_space import NullSpaceDecomposition
from rbf_gen.generator import Generator


class RBFGenModel(nn.Module):
    def __init__(self, rbf_basis: RBFBasis, generator: Generator):
        super().__init__()
        self.rbf_basis = rbf_basis
        self.null_decomp = NullSpaceDecomposition()
        self.generator = generator

    def fit_null_space(self, X: Tensor, y: Tensor) -> None:
        Phi = self.rbf_basis.compute_matrix(X)
        self.null_decomp.fit(Phi, y)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        # z: (B, latent_dim)
        # returns: (N_eval,) if B==1, else (N_eval, B)
        alpha = self.generator(z)  # (B, null_dim)

        # w = w0 + null_basis @ alpha.T  ->  (K, B)
        w = self.null_decomp.w0.unsqueeze(1) + self.null_decomp.null_basis @ alpha.T

        Phi_x = self.rbf_basis.compute_matrix(x)  # (N_eval, K)
        result = Phi_x @ w  # (N_eval, B)

        if z.shape[0] == 1:
            return result.squeeze(1)  # (N_eval,)
        return result  # (N_eval, B)

    def _sample_ensemble(self, x: Tensor, n_samples: int) -> Tensor:
        # Returns (N_eval, n_samples) function evaluations over random z
        return self.forward(x, self.sample_z(n_samples))

    def predict_mean(self, x: Tensor, n_samples: int = 100) -> Tensor:
        return self._sample_ensemble(x, n_samples).mean(dim=1)

    def predict_std(self, x: Tensor, n_samples: int = 100) -> Tensor:
        return self._sample_ensemble(x, n_samples).std(dim=1)

    def sample_z(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.generator.latent_dim)
