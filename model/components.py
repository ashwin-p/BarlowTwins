from typing import Any
import jax.numpy as jnp
from flax import linen as nn


class ResidualConv(nn.Module):
    out_channels: int
    strides: int = 1
    activation: str = "pre"
    dtype: Any = jnp.float32

    def setup(self):

        if self.activation not in ("pre", "post"):
            raise ValueError("activation must be 'pre' or 'post'")

        self.conv1 = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=self.strides,
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
        )

        self.conv2 = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=1,
            padding="SAME",
            use_bias=False,
            dtype=self.dtype,
        )

        self.norm1 = nn.BatchNorm(
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        self.norm2 = nn.BatchNorm(
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        self.proj = None

    @nn.compact
    def __call__(self, x, train: bool = True):

        shortcut = x

        if self.activation == "pre":

            out = self.norm1(
                x,
                use_running_average=not train,
            )
            out = nn.relu(out)

            if (
                self.strides != 1
                or x.shape[-1] != self.out_channels
            ):
                shortcut = nn.Conv(
                    features=self.out_channels,
                    kernel_size=(1, 1),
                    strides=self.strides,
                    use_bias=False,
                    dtype=self.dtype,
                )(out)

            out = self.conv1(out)

            out = self.norm2(
                out,
                use_running_average=not train,
            )
            out = nn.relu(out)

            out = self.conv2(out)

            out = out + shortcut

        else:

            out = self.conv1(x)

            out = self.norm1(
                out,
                use_running_average=not train,
            )
            out = nn.relu(out)

            out = self.conv2(out)

            out = self.norm2(
                out,
                use_running_average=not train,
            )

            if (
                self.strides != 1
                or x.shape[-1] != self.out_channels
            ):
                shortcut = nn.Conv(
                    features=self.out_channels,
                    kernel_size=(1, 1),
                    strides=self.strides,
                    use_bias=False,
                    dtype=self.dtype,
                )(shortcut)

            out = nn.relu(out + shortcut)

        return out


class ProjectionHead(nn.Module):
    hidden_dim: int
    output_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):

        x = jnp.mean(x, axis=(1, 2))

        x = nn.Dense(
            features=self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
        )(x)

        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )(x)

        x = nn.relu(x)

        x = nn.Dense(
            features=self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
        )(x)

        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )(x)

        x = nn.relu(x)

        x = nn.Dense(
            features=self.output_dim,
            use_bias=False,
            dtype=self.dtype,
        )(x)

        return x
