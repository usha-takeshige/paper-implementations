import torch
import pytest
from rbf_gen.kernels import GaussianKernel
from rbf_gen.rbf import RBFBasis
from rbf_gen.null_space import NullSpaceDecomposition
from rbf_gen.generator import Generator
from rbf_gen.model import RBFGenModel
from rbf_gen.losses import (
    MonotonicityPenalty,
    PositivityPenalty,
    LipschitzPenalty,
    SmoothnessPenalty,
    ConvexityPenalty,
    BoundaryPenalty,
    PointValueKL,
    RegionalAverageKL,
    ExtremalValueKL,
    GradientMagnitudeKL,
    RBFGenLoss,
)


@pytest.fixture
def grid():
    torch.manual_seed(42)
    return torch.linspace(0, 1, 20).unsqueeze(1)  # (20, 1)


@pytest.fixture
def model_setup():
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
    return model, K, N, d


class TestMonotonicityPenalty:
    def test_monotonicity_zero_for_increasing(self, grid):
        f_z = grid.squeeze() * 2  # strictly increasing
        pen = MonotonicityPenalty(increasing=True)
        assert pen(f_z, grid) < 1e-5

    def test_monotonicity_positive_for_decreasing(self, grid):
        f_z = -grid.squeeze() * 2  # strictly decreasing
        pen = MonotonicityPenalty(increasing=True)
        assert pen(f_z, grid) > 0

    def test_monotonicity_direction(self, grid):
        f_z = -grid.squeeze() * 2  # strictly decreasing
        pen = MonotonicityPenalty(increasing=False)
        assert pen(f_z, grid) < 1e-5


class TestPositivityPenalty:
    def test_positivity_zero_above_bound(self, grid):
        f_z = grid.squeeze() + 1.0  # all > 0
        pen = PositivityPenalty(lower_bound=0.0)
        assert pen(f_z, grid) < 1e-5

    def test_positivity_positive_below_bound(self, grid):
        f_z = grid.squeeze() - 0.5  # some < 0
        pen = PositivityPenalty(lower_bound=0.0)
        assert pen(f_z, grid) > 0


class TestLipschitzPenalty:
    def test_lipschitz_zero_within_bound(self):
        x = torch.tensor([[0.0], [1.0], [2.0]])
        f_z = x.squeeze()  # gradient = 1 <= L=2
        pen = LipschitzPenalty(L=2.0)
        assert pen(f_z, x) < 1e-5

    def test_lipschitz_positive_exceeds_bound(self):
        x = torch.tensor([[0.0], [1.0], [2.0]])
        f_z = x.squeeze() * 10  # gradient = 10 > L=2
        pen = LipschitzPenalty(L=2.0)
        assert pen(f_z, x) > 0


class TestSmoothnessPenalty:
    def test_smoothness_zero_for_linear(self, grid):
        f_z = grid.squeeze() * 3 + 1  # linear -> second difference = 0
        pen = SmoothnessPenalty()
        assert pen(f_z, grid) < 1e-5

    def test_smoothness_positive_for_bumpy(self, grid):
        f_z = torch.sin(grid.squeeze() * 20)  # high-frequency
        pen = SmoothnessPenalty()
        assert pen(f_z, grid) > 0


class TestConvexityPenalty:
    def test_convexity_zero_for_convex(self, grid):
        f_z = grid.squeeze() ** 2  # convex
        pen = ConvexityPenalty(convex=True)
        assert pen(f_z, grid) < 1e-5

    def test_convexity_positive_for_concave(self, grid):
        f_z = -(grid.squeeze() ** 2)  # concave
        pen = ConvexityPenalty(convex=True)
        assert pen(f_z, grid) > 0


class TestBoundaryPenalty:
    def test_boundary_zero_exact_match(self):
        boundary_points = torch.tensor([[0.0], [1.0]])
        boundary_values = torch.tensor([0.0, 1.0])
        f_z_at_boundary = boundary_values.clone()
        pen = BoundaryPenalty(boundary_points=boundary_points, boundary_values=boundary_values)
        assert pen(f_z_at_boundary, boundary_points) < 1e-5

    def test_boundary_positive_mismatch(self):
        boundary_points = torch.tensor([[0.0], [1.0]])
        boundary_values = torch.tensor([0.0, 1.0])
        f_z_at_boundary = torch.tensor([0.5, 0.5])  # mismatch
        pen = BoundaryPenalty(boundary_points=boundary_points, boundary_values=boundary_values)
        assert pen(f_z_at_boundary, boundary_points) > 0


class TestKLDivergence:
    def test_kl_returns_scalar(self):
        torch.manual_seed(42)
        x0 = torch.zeros(1, 1)
        kl = PointValueKL(x0=x0, target_mean=0.5, target_std=0.2)
        f_z_batch = torch.randn(50)
        result = kl(f_z_batch)
        assert result.shape == ()

    def test_kl_nonnegative(self):
        torch.manual_seed(42)
        x0 = torch.zeros(1, 1)
        kl = PointValueKL(x0=x0, target_mean=0.5, target_std=0.2)
        f_z_batch = torch.randn(50)
        assert kl(f_z_batch) >= 0

    def test_kl_zero_for_matching_distribution(self):
        torch.manual_seed(42)
        x0 = torch.zeros(1, 1)
        kl = PointValueKL(x0=x0, target_mean=0.0, target_std=1.0)
        f_z_batch = torch.randn(1000)  # N(0,1) matches target
        assert kl(f_z_batch) < 0.5

    def test_point_value_kl_shape(self):
        torch.manual_seed(42)
        x0 = torch.zeros(1, 1)
        kl = PointValueKL(x0=x0, target_mean=0.0, target_std=1.0)
        result = kl(torch.randn(50))
        assert result.shape == ()

    def test_regional_average_kl_shape(self):
        torch.manual_seed(42)
        region_points = torch.rand(10, 2)
        kl = RegionalAverageKL(region_points=region_points, target_mean=0.5, target_std=0.1)
        # f_z_batch: (B, n_region) - B function samples, each evaluated at n_region points
        result = kl(torch.randn(50, 10))
        assert result.shape == ()

    def test_extremal_value_kl_max_vs_min(self):
        torch.manual_seed(42)
        region_points = torch.rand(10, 2)
        kl_max = ExtremalValueKL(region_points=region_points, target_mean=1.0, target_std=0.1, use_max=True)
        kl_min = ExtremalValueKL(region_points=region_points, target_mean=0.0, target_std=0.1, use_max=False)
        f_z_batch = torch.rand(50, 10)
        val_max = kl_max(f_z_batch)
        val_min = kl_min(f_z_batch)
        assert val_max.shape == ()
        assert val_min.shape == ()

    def test_gradient_kl_requires_grad(self):
        torch.manual_seed(42)
        x0 = torch.rand(1, 2, requires_grad=True)
        kl = GradientMagnitudeKL(x0=x0, target_mean=0.5, target_std=0.1)
        f_z_batch = torch.randn(50)
        result = kl(f_z_batch)
        assert result.shape == ()


class TestRBFGenLoss:
    def test_loss_is_scalar(self, model_setup):
        torch.manual_seed(0)
        model, K, N, d = model_setup
        grid = torch.rand(20, d)
        penalty_terms = [MonotonicityPenalty(increasing=True)]
        loss_fn = RBFGenLoss(penalty_terms=penalty_terms, kl_terms=[])
        loss = loss_fn(model, grid, batch_size=8)
        assert loss.shape == ()

    def test_loss_nonnegative(self, model_setup):
        torch.manual_seed(0)
        model, K, N, d = model_setup
        grid = torch.rand(20, d)
        penalty_terms = [PositivityPenalty(lower_bound=0.0)]
        loss_fn = RBFGenLoss(penalty_terms=penalty_terms, kl_terms=[])
        loss = loss_fn(model, grid, batch_size=8)
        assert loss >= 0

    def test_loss_weighted_sum(self, model_setup):
        torch.manual_seed(0)
        model, K, N, d = model_setup
        grid = torch.rand(20, d)
        pen = MonotonicityPenalty(increasing=True, weight=0.0)
        loss_fn = RBFGenLoss(penalty_terms=[pen], kl_terms=[])
        loss = loss_fn(model, grid, batch_size=8)
        assert loss < 1e-5

    def test_loss_gradient_flows_to_generator(self, model_setup):
        torch.manual_seed(0)
        model, K, N, d = model_setup
        grid = torch.rand(20, d)
        penalty_terms = [MonotonicityPenalty(increasing=True)]
        loss_fn = RBFGenLoss(penalty_terms=penalty_terms, kl_terms=[])
        loss = loss_fn(model, grid, batch_size=8)
        loss.backward()
        for param in model.generator.parameters():
            assert param.grad is not None
