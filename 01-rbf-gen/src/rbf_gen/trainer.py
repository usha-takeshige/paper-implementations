import torch
from torch import Tensor
from rbf_gen.model import RBFGenModel
from rbf_gen.losses import RBFGenLoss



class RBFGenTrainer:
    def __init__(
        self,
        model: RBFGenModel,
        loss_fn: RBFGenLoss,
        n_epochs: int = 100,
        batch_size: int = 32,
        eval_grid: Tensor | None = None,
        lr: float = 1e-3,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.eval_grid = eval_grid
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self) -> None:
        for _ in range(self.n_epochs):
            self._train_step()

    def _train_step(self) -> Tensor:
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model, self.eval_grid, self.batch_size)
        loss.backward()
        self.optimizer.step()
        return loss.detach()
