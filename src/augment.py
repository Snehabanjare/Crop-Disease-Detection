"""Training-only image augmentation layers."""

import tensorflow as tf


def create_augmentation(seed: int = 42) -> tf.keras.Sequential:
    """Create reproducible augmentation layers that are inactive at inference."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                mode="horizontal_and_vertical",
                seed=seed,
            ),
            tf.keras.layers.RandomRotation(
                factor=0.1,
                seed=seed + 1,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=0.1,
                width_factor=0.1,
                seed=seed + 2,
            ),
            tf.keras.layers.RandomBrightness(
                factor=0.15,
                value_range=(0.0, 1.0),
                seed=seed + 3,
            ),
        ],
        name="training_augmentation",
    )
