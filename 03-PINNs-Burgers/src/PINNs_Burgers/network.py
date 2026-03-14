"""Feedforward neural network for PINNs."""

import torch
import torch.nn as nn

from .config import NetworkConfig


class PINN(nn.Module):
    """式(5) のフィードフォワードネットワーク u_theta(t,x) を実装する。

    論文に従い，活性化関数は tanh に固定し，Xavier 初期化を適用する。
    Algorithm 2 Step 1，式(5) に対応。
    """

    def __init__(self, config: NetworkConfig) -> None:
        """ネットワークを構築し，Xavier 初期化を適用する。

        Args:
            config: ネットワーク構造設定（層数・ニューロン数）。
        """
        super().__init__()

        layers: list[nn.Module] = []
        in_features = 2  # 入力：(t, x)

        for _ in range(config.n_hidden_layers):
            linear = nn.Linear(in_features, config.n_neurons)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.Tanh()])
            in_features = config.n_neurons

        output_layer = nn.Linear(in_features, 1)
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ネットワークの順伝播を計算する。

        Args:
            t: 時間座標，形状 (N, 1)。
            x: 空間座標，形状 (N, 1)。

        Returns:
            u_theta(t, x) の予測値，形状 (N, 1)。
        """
        inputs = torch.cat([t, x], dim=1)
        return self.network(inputs)
