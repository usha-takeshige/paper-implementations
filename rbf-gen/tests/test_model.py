import torch
import pytest
from rbf_gen.kernels import GaussianKernel
from rbf_gen.rbf import RBFBasis
from rbf_gen.null_space import NullSpaceDecomposition
from rbf_gen.generator import Generator
from rbf_gen.model import RBFGenModel


@pytest.fixture
def setup():
    torch.manual_seed(42)
    N, K, d = 5, 15, 2
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    kernel = GaussianKernel(epsilon=1.0)
    rbf_basis = RBFBasis.from_uniform(K, bounds, kernel)
    X = torch.rand(N, d)
    y = torch.rand(N)
    null_decomp = NullSpaceDecomposition()
    null_decomp.fit(rbf_basis.compute_matrix(X), y)
    generator = Generator(latent_dim=K - N, null_dim=K - N)
    model = RBFGenModel(rbf_basis=rbf_basis, null_decomp=null_decomp, generator=generator)
    return model, X, y, N, K, d


class TestInterpolationCondition:
    def test_interpolation_condition_random_z(self, setup):
        torch.manual_seed(0)
        model, X, y, N, K, d = setup
        z = torch.randn(1, K - N)
        f_z = model.forward(X, z)
        assert torch.allclose(f_z, y, atol=1e-5)

    def test_interpolation_condition_after_training(self, setup):
        torch.manual_seed(0)
        model, X, y, N, K, d = setup
        # Perturb generator parameters (simulates training)
        for param in model.generator.parameters():
            param.data += 0.1 * torch.randn_like(param)
        z = torch.randn(1, K - N)
        f_z = model.forward(X, z)
        assert torch.allclose(f_z, y, atol=1e-5)


class TestInference:
    def test_forward_output_shape(self, setup):
        torch.manual_seed(0)
        model, X, y, N, K, d = setup
        N_eval = 10
        x_eval = torch.rand(N_eval, d)
        z = torch.randn(1, K - N)
        out = model.forward(x_eval, z)
        assert out.shape == (N_eval,)

    def test_predict_mean_shape(self, setup):
        torch.manual_seed(0)
        model, X, y, N, K, d = setup
        N_eval = 10
        x_eval = torch.rand(N_eval, d)
        mean = model.predict_mean(x_eval, n_samples=20)
        assert mean.shape == (N_eval,)

    def test_predict_std_nonnegative(self, setup):
        torch.manual_seed(0)
        model, X, y, N, K, d = setup
        N_eval = 10
        x_eval = torch.rand(N_eval, d)
        std = model.predict_std(x_eval, n_samples=20)
        assert (std >= 0).all()

    def test_sample_z_distribution(self, setup):
        from scipy import stats

        torch.manual_seed(0)
        model, X, y, N, K, d = setup
        n_samples = 1000
        z = model.sample_z(n_samples)
        # Check each dimension is approximately normal
        for i in range(z.shape[1]):
            _, p_value = stats.kstest(z[:, i].detach().numpy(), "norm")
            assert p_value > 0.01
