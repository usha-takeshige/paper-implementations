from abc import ABC, abstractmethod
import torch
from torch import Tensor


class PenaltyTerm(ABC):
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def __call__(self, f_z: Tensor, grid: Tensor) -> Tensor:
        pass


class MonotonicityPenalty(PenaltyTerm):
    def __init__(self, increasing: bool = True, dim: int = 0, weight: float = 1.0):
        super().__init__(weight)
        self.increasing = increasing
        self.dim = dim

    def __call__(self, f_z: Tensor, grid: Tensor) -> Tensor:
        # f_z: (G,) values on grid points
        diff = f_z[1:] - f_z[:-1]  # (G-1,)
        if self.increasing:
            violations = torch.relu(-diff)  # penalise decreases
        else:
            violations = torch.relu(diff)   # penalise increases
        return self.weight * violations.mean()


class PositivityPenalty(PenaltyTerm):
    def __init__(self, lower_bound: float = 0.0, weight: float = 1.0):
        super().__init__(weight)
        self.lower_bound = lower_bound

    def __call__(self, f_z: Tensor, grid: Tensor) -> Tensor:
        return self.weight * torch.relu(self.lower_bound - f_z.min())


class LipschitzPenalty(PenaltyTerm):
    def __init__(self, L: float = 1.0, weight: float = 1.0):
        super().__init__(weight)
        self.L = L

    def __call__(self, f_z: Tensor, grid: Tensor) -> Tensor:
        # Pairwise slopes: |f(x) - f(y)| / ||x - y||
        x_diff = grid.unsqueeze(0) - grid.unsqueeze(1)        # (G, G, d)
        x_dist = torch.norm(x_diff, dim=-1)                    # (G, G)
        f_diff = torch.abs(f_z.unsqueeze(0) - f_z.unsqueeze(1))  # (G, G)

        safe_dist = torch.clamp(x_dist, min=1e-10)
        slopes = f_diff / safe_dist
        # Zero out diagonal (same-point pairs)
        eye = torch.eye(grid.shape[0], device=grid.device, dtype=grid.dtype)
        slopes = slopes * (1.0 - eye)

        violations = torch.relu(slopes - self.L)
        return self.weight * violations.sum()


class SmoothnessPenalty(PenaltyTerm):
    def __call__(self, f_z: Tensor, grid: Tensor) -> Tensor:
        # Second finite difference: f[k+1] - 2*f[k] + f[k-1]
        second_diff = f_z[2:] - 2.0 * f_z[1:-1] + f_z[:-2]
        return self.weight * (second_diff ** 2).sum()


class ConvexityPenalty(PenaltyTerm):
    def __init__(self, convex: bool = True, weight: float = 1.0):
        super().__init__(weight)
        self.convex = convex

    def __call__(self, f_z: Tensor, grid: Tensor) -> Tensor:
        second_diff = f_z[2:] - 2.0 * f_z[1:-1] + f_z[:-2]
        if self.convex:
            violations = torch.relu(-second_diff)  # penalise second_diff < 0
        else:
            violations = torch.relu(second_diff)   # penalise second_diff > 0
        return self.weight * violations.sum()


class BoundaryPenalty(PenaltyTerm):
    def __init__(self, boundary_points: Tensor, boundary_values: Tensor, weight: float = 1.0):
        super().__init__(weight)
        self.boundary_points = boundary_points
        self.boundary_values = boundary_values

    def __call__(self, f_z: Tensor, grid: Tensor) -> Tensor:
        # f_z: model evaluated at boundary_points
        return self.weight * ((f_z - self.boundary_values) ** 2).sum()


class KLDivergenceTerm(ABC):
    def __init__(self, target_mean: float, target_std: float, weight: float = 1.0):
        self.target_mean = target_mean
        self.target_std = target_std
        self.weight = weight

    @abstractmethod
    def __call__(self, f_z_batch: Tensor) -> Tensor:
        pass

    def _gaussian_kl(self, samples: Tensor) -> Tensor:
        # KL( N(mu_gen, sigma_gen^2) || N(target_mean, target_std^2) )
        mu = samples.mean()
        sigma = samples.std().clamp(min=1e-8)
        mu_t = torch.tensor(self.target_mean, dtype=samples.dtype)
        sigma_t = torch.tensor(self.target_std, dtype=samples.dtype)
        kl = torch.log(sigma_t / sigma) + (sigma ** 2 + (mu - mu_t) ** 2) / (2.0 * sigma_t ** 2) - 0.5
        return self.weight * kl


class PointValueKL(KLDivergenceTerm):
    def __init__(self, x0: Tensor, target_mean: float, target_std: float, weight: float = 1.0):
        super().__init__(target_mean, target_std, weight)
        self.x0 = x0

    def __call__(self, f_z_batch: Tensor) -> Tensor:
        # f_z_batch: (B,) scalar values at x0 from B different z samples
        return self._gaussian_kl(f_z_batch)


class RegionalAverageKL(KLDivergenceTerm):
    def __init__(self, region_points: Tensor, target_mean: float, target_std: float, weight: float = 1.0):
        super().__init__(target_mean, target_std, weight)
        self.region_points = region_points

    def __call__(self, f_z_batch: Tensor) -> Tensor:
        # f_z_batch: (B, n_region) -> regional averages: (B,)
        averages = f_z_batch.mean(dim=1)
        return self._gaussian_kl(averages)


class ExtremalValueKL(KLDivergenceTerm):
    def __init__(self, region_points: Tensor, target_mean: float, target_std: float,
                 use_max: bool = True, weight: float = 1.0):
        super().__init__(target_mean, target_std, weight)
        self.region_points = region_points
        self.use_max = use_max

    def __call__(self, f_z_batch: Tensor) -> Tensor:
        # f_z_batch: (B, n_region)
        if self.use_max:
            extremals = f_z_batch.max(dim=1).values
        else:
            extremals = f_z_batch.min(dim=1).values
        return self._gaussian_kl(extremals)


class GradientMagnitudeKL(KLDivergenceTerm):
    def __init__(self, x0: Tensor, target_mean: float, target_std: float, weight: float = 1.0):
        super().__init__(target_mean, target_std, weight)
        self.x0 = x0

    def __call__(self, f_z_batch: Tensor) -> Tensor:
        # f_z_batch: (B,) gradient magnitudes ||grad f_z(x0)|| for B z samples
        return self._gaussian_kl(f_z_batch)


class CurvatureKL(KLDivergenceTerm):
    def __init__(self, x0: Tensor, dim: int, target_mean: float, target_std: float, weight: float = 1.0):
        super().__init__(target_mean, target_std, weight)
        self.x0 = x0
        self.dim = dim

    def __call__(self, f_z_batch: Tensor) -> Tensor:
        # f_z_batch: (B,) curvature values d^2 f_z / dx_dim^2 at x0
        return self._gaussian_kl(f_z_batch)


class RBFGenLoss:
    def __init__(self, penalty_terms: list[PenaltyTerm], kl_terms: list[KLDivergenceTerm]):
        self.penalty_terms = penalty_terms
        self.kl_terms = kl_terms

    def __call__(self, model, eval_grid: Tensor, batch_size: int) -> Tensor:
        z = model.sample_z(batch_size)                 # (B, latent_dim)
        f_z_batch = model.forward(eval_grid, z)        # (N_eval, B)

        loss_parts: list[Tensor] = []

        for pen in self.penalty_terms:
            per_sample = torch.stack([pen(f_z_batch[:, b], eval_grid) for b in range(batch_size)])
            loss_parts.append(per_sample.mean())

        for kl_term in self.kl_terms:
            loss_parts.append(kl_term(f_z_batch))

        if not loss_parts:
            return torch.tensor(0.0)
        return torch.stack(loss_parts).sum()
