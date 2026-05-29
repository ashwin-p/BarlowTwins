import os
import pandas as pd
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
import grain.python as grain
from tqdm import tqdm
from pathlib import Path
from typing import Any
from functools import partial
from flax.training import train_state

from model.resnet import ResNet
from dataloader.cifar_dataloader import CIFARSource
from .augment import AugmentConfig, augment


class TrainState(train_state.TrainState):
    batch_stats: Any


def create_train_state(
    rng,
    model,
    learning_rate,
    weight_decay,
    total_steps,
    input_shape,
):

    variables = model.init(
        rng,
        jnp.ones(input_shape, dtype=jnp.float32),
        train=True,
    )

    params = variables["params"]
    batch_stats = variables["batch_stats"]

    warmup_steps = int(0.05 * total_steps)

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )

    def decay_mask_fn(params):

        def mask_fn(path, _):
            path_str = jax.tree_util.keystr(
                path,
                simple=True,
                separator="/",
            )

            return not (
                "bias" in path_str
                or "scale" in path_str
            )

        return jax.tree_util.tree_map_with_path(
            mask_fn,
            params,
        )

    decay_mask = decay_mask_fn(params)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            mask=decay_mask,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        ),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=tx,
    )


mapped_augment_imgs = jax.vmap(augment, in_axes=(0, 0, None))


@partial(jax.jit, static_argnames=["augmentation_config"])
def train_step(rng, state, batch, augmentation_config,
               redundancy_weight=0.005,
               epsilon=1e-6):
    img = batch["img"]
    rng = jax.random.split(rng, num=img.shape[0]+1)
    augmentation_keys = rng[1:img.shape[0]+1]
    rng = rng[0]
    augmentation_keys, view_a = mapped_augment_imgs(augmentation_keys,
                                                    img,
                                                    augmentation_config)
    augmentation_keys, view_b = mapped_augment_imgs(augmentation_keys,
                                                    img,
                                                    augmentation_config)
    combined_views = jnp.concatenate([view_a, view_b], axis=0)

    def loss_fn(params):
        (h, z), updated_state = state.apply_fn(
            {"params": params,
             "batch_stats": state.batch_stats},
            combined_views,
            mutable=["batch_stats"],
            train=True
        )

        z_a, z_b = jnp.split(z, 2, axis=0)

        z_a_mean = jnp.mean(z_a, axis=0)
        z_a_std = jnp.std(z_a, axis=0)
        z_b_mean = jnp.mean(z_b, axis=0)
        z_b_std = jnp.std(z_b, axis=0)

        z_a_norm = (z_a - z_a_mean) / (z_a_std + epsilon)
        z_b_norm = (z_b - z_b_mean) / (z_b_std + epsilon)

        cross_correlation_matrix = jnp.matmul(z_a_norm.T, z_b_norm) / img.shape[0]
        cc_matrix_len = cross_correlation_matrix.shape[0]
        identity_matrix = jnp.identity(cc_matrix_len,
                                       dtype=cross_correlation_matrix.dtype)
        invariance_loss = jnp.sum(jnp.square(1-cross_correlation_matrix) * identity_matrix) / cc_matrix_len
        redundancy_loss = jnp.sum(jnp.square(cross_correlation_matrix * ( 1 - identity_matrix))) / (jnp.square(cc_matrix_len) - cc_matrix_len)
        loss = invariance_loss + redundancy_weight * redundancy_loss

        embedding_std = jnp.mean(jnp.std(h, axis=0))

        metrics = {"loss": loss,
                   "invariance_loss": invariance_loss,
                   "redundancy_loss": redundancy_loss,
                   "embedding_std": embedding_std}

        return loss, (rng, updated_state, metrics)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (rng, updated_state, metrics)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updated_state["batch_stats"])
    return rng, state, metrics


def train():
    train_file = Path(__file__).resolve().parent.parent / "cifar-10-python" / "cifar-10-batches-py/"
    batch_size = 512
    epochs = 300
    learning_rate = 4e-4
    weight_decay = 5e-4

    train_source = CIFARSource(train_file, "data_batch_")
    train_sampler = grain.IndexSampler(
        num_records=len(train_source),
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
        num_epochs=1,
        shuffle=True,
        seed=0,
    )

    train_loader = grain.DataLoader(
        data_source=train_source,
        sampler=train_sampler,
        worker_count=4,
        operations=[
            grain.Batch(batch_size=batch_size, drop_remainder=True)
        ]
    )

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
    total_steps = (len(train_sampler) // batch_size) * epochs
    ckpt_path = os.path.abspath("barlow_twins_checkpoints")
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True,
                                           step_prefix="barlow_twins")

    model = ResNet(stem_channels=[32],
                   stem_strides=[1],
                   res_channels=[32, 32,
                                 64, 64,
                                 128, 128,
                                 256, 256],
                   res_strides=[1, 1,
                                2, 1,
                                2, 1,
                                2, 1])

    state = create_train_state(init_rng, model, learning_rate, weight_decay,
                               total_steps, dummy_input.shape)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Parameter Count: {param_count/1e6:.2f}M")

    flop_analysis = jax.jit(
        state.apply_fn,
        static_argnames=["train"],
    ).lower(
        {
            "params": state.params,
            "batch_stats": state.batch_stats,
        },
        dummy_input,
        train=False,
    ).cost_analysis()

    if flop_analysis is None:
        flops = 0
    elif isinstance(flop_analysis, list):
        flops = flop_analysis[0].get("flops", 0)
    else:
        flops = flop_analysis.get("flops", 0)

    print(f"FLOPs (per forward pass): {flops / 1e6:.4f}M")

    os.makedirs('logs', exist_ok=True)
    log_path = "logs/barlow_train_metrics.csv"
    df = pd.DataFrame(columns=[
        "epoch",
        "train_loss",
        "invariance_loss",
        "redundancy_loss",
        "embedding_std"
    ])
    df.to_csv(log_path, index=False)

    steps_per_epoch = len(train_source) // batch_size

    augment_config = AugmentConfig(
        random_crop_resize_crop_size=(24, 24),
        random_crop_resize_new_size=(32, 32, 3),

        random_horizontal_flip_p=0.5,

        brightness_jitter_min_scale=0.5,
        brightness_jitter_max_scale=1.5,

        contrast_jitter_min_scale=0.5,
        contrast_jitter_max_scale=1.5,

        saturation_jitter_min_scale=0.5,
        saturation_jitter_max_scale=1.5,

        random_grayscale_p=0.1,

        normalize_means=(0.4914, 0.4822, 0.4465),
        normalize_stds=(0.2470, 0.2435, 0.2616),
    )

    for epoch in range(epochs):

        running_loss = 0.0
        running_invariance_loss = 0.0
        running_redundancy_loss = 0.0
        running_embedding_std = 0.0

        pbar = tqdm(
                train_loader,
                total=steps_per_epoch,
                desc=f"epoch {epoch+1}/{epochs}",
        )

        for step, batch in enumerate(pbar):
            rng, state, metrics = train_step(rng, state, batch, augment_config)

            loss = float(metrics["loss"])
            invariance_loss = float(metrics["invariance_loss"])
            redundancy_loss = float(metrics["redundancy_loss"])
            embedding_std = float(metrics["embedding_std"])

            running_loss += loss
            running_invariance_loss += invariance_loss
            running_redundancy_loss += redundancy_loss
            running_embedding_std += embedding_std

            pbar.set_postfix({
                "loss": loss,
            })

        # epoch averages
        avg_loss = running_loss / steps_per_epoch
        avg_invariance_loss = running_invariance_loss / steps_per_epoch
        avg_redundancy_loss = running_redundancy_loss / steps_per_epoch
        avg_embedding_std = running_embedding_std / steps_per_epoch

        print(
            f"epoch {epoch+1}/{epochs} | "
            f"loss: {avg_loss:.4f} | "
            f"invariance_loss: {avg_invariance_loss:.4f} | "
            f"redundancy_loss: {avg_redundancy_loss:.4f} | "
            f"embedding_std: {avg_embedding_std}"
        )

        with ocp.CheckpointManager(ckpt_path, options=options) as mngr:
            mngr.save(
                epoch+1,
                args=ocp.args.StandardSave(state)
            )

        row = pd.DataFrame([{
            "epoch": epoch+1,
            "train_loss": avg_loss,
            "invariance_loss": avg_invariance_loss,
            "redundancy_loss": avg_redundancy_loss,
            "embedding_std": avg_embedding_std
        }])
        row.to_csv(
            log_path,
            mode="a",
            header=False,
            index=False
        )


if __name__ == "__main__":
    train()
