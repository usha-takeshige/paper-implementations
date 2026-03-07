import torch
import pytest
from rbf_gen.kernels import GaussianKernel
from rbf_gen.rbf import RBFBasis


@pytest.fixture
def setup():
    torch.manual_seed(42)
    K, d = 15, 2
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    kernel = GaussianKernel(epsilon=1.0)
    return K, d, bounds, kernel


class TestRBFBasisCenters:
    def test_from_uniform_center_count(self, setup):
        K, d, bounds, kernel = setup
        basis = RBFBasis.from_uniform(K, bounds, kernel)
        assert basis.centers.shape == (K, d)

    def test_from_uniform_in_bounds(self, setup):
        K, d, bounds, kernel = setup
        basis = RBFBasis.from_uniform(K, bounds, kernel)
        lb, ub = bounds[0], bounds[1]
        assert (basis.centers >= lb).all()
        assert (basis.centers <= ub).all()

    def test_from_quasi_random_center_count(self, setup):
        K, d, bounds, kernel = setup
        basis = RBFBasis.from_quasi_random(K, bounds, kernel)
        assert basis.centers.shape == (K, d)


class TestRBFBasisMatrix:
    def test_compute_matrix_shape(self, setup):
        torch.manual_seed(42)
        K, d, bounds, kernel = setup
        N = 5
        basis = RBFBasis.from_uniform(K, bounds, kernel)
        X = torch.rand(N, d)
        Phi = basis.compute_matrix(X)
        assert Phi.shape == (N, K)

    def test_compute_matrix_values(self, setup):
        torch.manual_seed(42)
        K, d, bounds, kernel = setup
        N = 5
        basis = RBFBasis.from_uniform(K, bounds, kernel)
        X = torch.rand(N, d)
        Phi = basis.compute_matrix(X)
        for i in range(N):
            for j in range(K):
                r = torch.norm(X[i] - basis.centers[j])
                expected = kernel(r)
                assert torch.isclose(Phi[i, j], expected, atol=1e-5)

    def test_compute_vector_shape(self, setup):
        torch.manual_seed(42)
        K, d, bounds, kernel = setup
        basis = RBFBasis.from_uniform(K, bounds, kernel)
        x = torch.rand(d)
        phi = basis.compute_vector(x)
        assert phi.shape == (K,)

    def test_compute_vector_values(self, setup):
        torch.manual_seed(42)
        K, d, bounds, kernel = setup
        basis = RBFBasis.from_uniform(K, bounds, kernel)
        x = torch.rand(d)
        phi = basis.compute_vector(x)
        for j in range(K):
            r = torch.norm(x - basis.centers[j])
            expected = kernel(r)
            assert torch.isclose(phi[j], expected, atol=1e-5)
