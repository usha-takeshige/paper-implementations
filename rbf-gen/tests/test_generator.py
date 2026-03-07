import torch
import pytest
from rbf_gen.generator import Generator


@pytest.fixture
def setup():
    torch.manual_seed(42)
    latent_dim = 8
    null_dim = 10
    return latent_dim, null_dim


class TestGenerator:
    def test_output_shape(self, setup):
        latent_dim, null_dim = setup
        gen = Generator(latent_dim=latent_dim, null_dim=null_dim)
        B = 4
        z = torch.randn(B, latent_dim)
        alpha = gen(z)
        assert alpha.shape == (B, null_dim)

    def test_different_z_different_output(self, setup):
        latent_dim, null_dim = setup
        gen = Generator(latent_dim=latent_dim, null_dim=null_dim)
        z1 = torch.randn(1, latent_dim)
        z2 = torch.randn(1, latent_dim)
        assert not torch.allclose(gen(z1), gen(z2))

    def test_gradient_flows(self, setup):
        latent_dim, null_dim = setup
        gen = Generator(latent_dim=latent_dim, null_dim=null_dim)
        z = torch.randn(4, latent_dim, requires_grad=True)
        alpha = gen(z)
        loss = alpha.sum()
        loss.backward()
        assert z.grad is not None

    def test_batch_consistency(self, setup):
        latent_dim, null_dim = setup
        gen = Generator(latent_dim=latent_dim, null_dim=null_dim)
        gen.eval()
        z = torch.randn(4, latent_dim)
        batch_output = gen(z)
        for i in range(4):
            single_output = gen(z[i : i + 1])
            assert torch.allclose(batch_output[i], single_output[0], atol=1e-5)
