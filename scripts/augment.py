from dataclasses import dataclass
import jax
import jax.numpy as jnp

def random_crop_resize(rng, img, crop_size, new_size):
    if (crop_size[0] > img.shape[0] or crop_size[1] > img.shape[1]):
        raise ValueError("Crop size cannot be more than image size!")
    rng, top_rng, left_rng = jax.random.split(rng, num=3)
    crop_top_idx = jax.random.randint(top_rng,
                                      shape=(),
                                      minval=0,
                                      maxval=img.shape[0] - crop_size[0] + 1)
    crop_left_idx = jax.random.randint(left_rng,
                                       shape=(),
                                       minval=0,
                                       maxval=img.shape[1] - crop_size[1] + 1)

    cropped_img = jax.lax.dynamic_slice(img,
                                        (crop_top_idx, crop_left_idx, 0),
                                        (crop_size[0], crop_size[1], 3))
    resized_img = jax.image.resize(cropped_img,
                                   shape=new_size,
                                   method="bilinear")

    return rng, resized_img


def random_horizontal_flip(rng, img, p=0.5):
    rng, flip_rng = jax.random.split(rng)
    out = jax.lax.cond(
        jax.random.bernoulli(flip_rng, p, shape=()),
        lambda x: jnp.flip(x, axis=1),
        lambda x: x,
        img
    )

    return rng, out

def brightness_jitter(rng, img, min_scale=0.5, max_scale=1.5):
    rng, brightness_rng = jax.random.split(rng)
    scale = jax.random.uniform(brightness_rng,
                               shape=(),
                               minval=min_scale,
                               maxval=max_scale)

    jittered_img = img * scale
    jittered_img = jnp.clip(jittered_img, 0, 1)

    return rng, jittered_img

def contrast_jitter(rng, img, min_scale=0.5, max_scale=1.5):
    rng, contrast_rng = jax.random.split(rng)
    scale = jax.random.uniform(contrast_rng,
                               shape=(),
                               minval=min_scale,
                               maxval=max_scale)

    img_mean = jnp.mean(img)
    jittered_img = (img - img_mean) * scale + img_mean
    jittered_img = jnp.clip(jittered_img, 0, 1)

    return rng, jittered_img


def to_grayscale(img):
    rgb_weights = jnp.asarray([0.299, 0.587, 0.114])[None, None, :]
    greyed_img = jnp.sum(img * rgb_weights, axis=-1, keepdims=True)
    greyed_img = jnp.clip(greyed_img, 0, 1)
    greyed_img = jnp.repeat(greyed_img, repeats=3, axis=-1)
    return greyed_img


def saturation_jitter(rng, img, min_scale=0.5, max_scale=1.5):
    rng, saturation_rng = jax.random.split(rng)
    scale = jax.random.uniform(saturation_rng,
                               shape=(),
                               minval=min_scale,
                               maxval=max_scale)
    grey_img = to_grayscale(img)

    jittered_img = scale * (img - grey_img) + grey_img
    jittered_img = jnp.clip(jittered_img, 0, 1)

    return rng, jittered_img


def random_grayscale(rng, img, p=0.1):
    rng, grey_rng = jax.random.split(rng)
    out = jax.lax.cond(
        jax.random.bernoulli(grey_rng, p=p, shape=()),
        lambda x: to_grayscale(x),
        lambda x: x,
        img
    )

    return rng, out

def normalize(img, means, stds):
    means = jnp.asarray(means)[None, None, :]
    stds = jnp.asarray(stds)[None, None, :]
    normalized_img = (img - means) / stds

    return normalized_img


@dataclass(frozen=True)
class AugmentConfig:
    random_crop_resize_crop_size: tuple[int, int]
    random_crop_resize_new_size: tuple[int, int, int]

    random_horizontal_flip_p: float

    brightness_jitter_min_scale: float
    brightness_jitter_max_scale: float

    contrast_jitter_min_scale: float
    contrast_jitter_max_scale: float

    saturation_jitter_min_scale: float
    saturation_jitter_max_scale: float

    random_grayscale_p: float

    normalize_means: tuple[float, float, float]
    normalize_stds: tuple[float, float, float]


def augment(rng, img, config):
    rng, img = random_crop_resize(
        rng,
        img,
        crop_size=config.random_crop_resize_crop_size,
        new_size=config.random_crop_resize_new_size
    )

    rng, img = random_horizontal_flip(
        rng,
        img,
        p=config.random_horizontal_flip_p
    )

    rng, img = brightness_jitter(
        rng,
        img,
        min_scale=config.brightness_jitter_min_scale,
        max_scale=config.brightness_jitter_max_scale
    )

    rng, img = contrast_jitter(
        rng,
        img,
        min_scale=config.contrast_jitter_min_scale,
        max_scale=config.contrast_jitter_max_scale
    )

    rng, img = saturation_jitter(
        rng,
        img,
        min_scale=config.saturation_jitter_min_scale,
        max_scale=config.saturation_jitter_max_scale
    )

    rng, img = random_grayscale(
        rng,
        img,
        p=config.random_grayscale_p
    )

    img = normalize(
        img,
        means=config.normalize_means,
        stds=config.normalize_stds
    )

    return rng, img
