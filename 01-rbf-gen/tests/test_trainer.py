import torch
import pytest
from rbf_gen.kernels import GaussianKernel
from rbf_gen.rbf import RBFBasis
from rbf_gen.null_space import NullSpaceDecomposition
from rbf_gen.generator import Generator
from rbf_gen.model import RBFGenModel
from rbf_gen.losses import MonotonicityPenalty, RBFGenLoss
from rbf_gen.trainer import RBFGenTrainer


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
    eval_grid = torch.rand(20, d)
    penalty_terms = [MonotonicityPenalty(increasing=True)]
    loss_fn = RBFGenLoss(penalty_terms=penalty_terms, kl_terms=[])
    trainer = RBFGenTrainer(
        model=model,
        loss_fn=loss_fn,
        n_epochs=10,
        batch_size=8,
        eval_grid=eval_grid,
    )
    return trainer, model, X, y


class TestTrainer:
    def test_loss_decreases(self, setup):
        trainer, model, X, y = setup
        initial_loss = trainer._train_step()
        for _ in range(9):
            trainer._train_step()
        final_loss = trainer._train_step()
        assert final_loss < initial_loss

    def test_interpolation_preserved_after_training(self, setup):
        torch.manual_seed(0)
        trainer, model, X, y = setup
        trainer.train()
        z = model.sample_z(1)
        f_z = model.forward(X, z)
        assert torch.allclose(f_z, y, atol=1e-5)

    def test_trainer_runs_without_error(self, setup):
        trainer, model, X, y = setup
        trainer.train()  # should not raise
