from typing import Sequence, Any

import jax.numpy as jnp
from flax import linen as nn
from .components import ResidualConv, ProjectionHead

class ResNet(nn.Module):
    stem_channels: Sequence[int]
    stem_strides: Sequence[int]
    res_channels: Sequence[int]
    res_strides: Sequence[int]
    projector_hidden_dim: int = 1024
    projector_output_dim: int = 1024
    activation: str = "pre"
    dtype: Any = jnp.float32

    def setup(self):

        if len(self.stem_channels) != len(self.stem_strides):
            raise ValueError(
                "Stem channels and strides must have same length!"
            )

        if len(self.res_channels) != len(self.res_strides):
            raise ValueError(
                "Residual channels and strides must have same length!"
            )

        self.stem_convs = [
            nn.Conv(
                features=ch,
                kernel_size=(3, 3),
                strides=st,
                padding="SAME",
                use_bias=False,
                dtype=self.dtype,
            )
            for ch, st in zip(
                self.stem_channels,
                self.stem_strides,
            )
        ]

        self.stem_norms = [
            nn.BatchNorm(
                momentum=0.9,
                epsilon=1e-5,
                dtype=self.dtype,
            )
            for _ in self.stem_channels
        ]

        self.res_blocks = [
            ResidualConv(
                out_channels=ch,
                strides=st,
                activation=self.activation,
                dtype=self.dtype,
            )
            for ch, st in zip(
                self.res_channels,
                self.res_strides,
            )
        ]

        self.projector = ProjectionHead(
            hidden_dim=self.projector_hidden_dim,
            output_dim=self.projector_output_dim,
            dtype=self.dtype,
        )

    def encode(self, x, train: bool = True):

        for conv, norm in zip(
            self.stem_convs,
            self.stem_norms,
        ):

            x = conv(x)

            x = norm(
                x,
                use_running_average=not train,
            )

            x = nn.relu(x)

        for block in self.res_blocks:
            x = block(x, train=train)

        h = jnp.mean(x, axis=(1, 2))

        return h

    def __call__(self, x, train: bool = True):

        h = self.encode(x, train=train)

        z = self.projector(
            h[:, None, None, :],
            train=train,
        )

        return h, z
