import torch
import pytest
from rbf_gen.kernels import GaussianKernel
from rbf_gen.rbf import RBFBasis
from rbf_gen.null_space import NullSpaceDecomposition


@pytest.fixture
def setup():
    torch.manual_seed(42)
    N, K, d = 5, 15, 2
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    kernel = GaussianKernel(epsilon=1.0)
    basis = RBFBasis.from_uniform(K, bounds, kernel)
    X = torch.rand(N, d)
    y = torch.rand(N)
    Phi = basis.compute_matrix(X)
    decomp = NullSpaceDecomposition()
    decomp.fit(Phi, y)
    return decomp, Phi, y, K, N


class TestNullSpaceDecomposition:
    def test_particular_solution_satisfies_interpolation(self, setup):
        decomp, Phi, y, K, N = setup
        residual = torch.norm(Phi @ decomp.w0 - y)
        assert residual < 1e-5

    def test_null_basis_in_kernel(self, setup):
        decomp, Phi, y, K, N = setup
        product = Phi @ decomp.null_basis
        assert torch.norm(product) < 1e-5

    def test_null_basis_orthonormal(self, setup):
        decomp, Phi, y, K, N = setup
        I_approx = decomp.null_basis.T @ decomp.null_basis
        I_expected = torch.eye(K - N)
        assert torch.allclose(I_approx, I_expected, atol=1e-5)

    def test_general_solution_satisfies_interpolation(self, setup):
        torch.manual_seed(0)
        decomp, Phi, y, K, N = setup
        alpha = torch.randn(K - N)
        w = decomp.w0 + decomp.null_basis @ alpha
        residual = torch.norm(Phi @ w - y)
        assert residual < 1e-5

    def test_shapes(self, setup):
        decomp, Phi, y, K, N = setup
        assert decomp.w0.shape == (K,)
        assert decomp.null_basis.shape == (K, K - N)

    def test_underdetermined_requirement(self):
        N, K = 5, 3  # K < N -> should raise ValueError
        Phi = torch.rand(N, K)
        y = torch.rand(N)
        decomp = NullSpaceDecomposition()
        with pytest.raises(ValueError):
            decomp.fit(Phi, y)
