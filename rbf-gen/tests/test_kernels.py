import torch
import pytest
from rbf_gen.kernels import GaussianKernel, ThinPlateSplineKernel


class TestGaussianKernel:
    def test_gaussian_output_shape(self):
        torch.manual_seed(42)
        kernel = GaussianKernel(epsilon=1.0)
        r = torch.rand(5, 3)
        assert kernel(r).shape == r.shape

    def test_gaussian_at_zero(self):
        torch.manual_seed(42)
        kernel = GaussianKernel(epsilon=1.0)
        r = torch.tensor(0.0)
        assert kernel(r) == 1.0

    def test_gaussian_decreasing(self):
        torch.manual_seed(42)
        kernel = GaussianKernel(epsilon=1.0)
        r1 = torch.tensor(1.0)
        r2 = torch.tensor(2.0)
        assert kernel(r1) > kernel(r2)

    def test_gaussian_epsilon_effect(self):
        torch.manual_seed(42)
        kernel_small = GaussianKernel(epsilon=0.5)
        kernel_large = GaussianKernel(epsilon=2.0)
        r = torch.tensor(1.0)
        assert kernel_large(r) < kernel_small(r)

    def test_gaussian_positive(self):
        torch.manual_seed(42)
        kernel = GaussianKernel(epsilon=1.0)
        r = torch.rand(10) * 5 + 0.01
        assert (kernel(r) > 0).all()


class TestThinPlateSplineKernel:
    def test_tps_output_shape(self):
        torch.manual_seed(42)
        kernel = ThinPlateSplineKernel()
        r = torch.rand(5, 3)
        assert kernel(r).shape == r.shape

    def test_tps_at_zero(self):
        torch.manual_seed(42)
        kernel = ThinPlateSplineKernel()
        r = torch.tensor(0.0)
        assert kernel(r) == 0.0

    def test_tps_positive_for_nonzero(self):
        torch.manual_seed(42)
        kernel = ThinPlateSplineKernel()
        r = torch.tensor(2.0)
        assert kernel(r) > 0

    def test_tps_numerical_stability(self):
        torch.manual_seed(42)
        kernel = ThinPlateSplineKernel()
        r = torch.tensor(1e-10)
        assert torch.isfinite(kernel(r))
