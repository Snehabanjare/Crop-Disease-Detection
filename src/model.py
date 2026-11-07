"""CNN and MobileNetV2 model builders."""

import tensorflow as tf

from src.augment import create_augmentation


def _validate_model_args(
    num_classes: int,
    input_shape: tuple[int, int, int],
) -> None:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    if len(input_shape) != 3 or any(dimension <= 0 for dimension in input_shape):
        raise ValueError("input_shape must contain three positive dimensions.")


def build_baseline_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    seed: int = 42,
) -> tf.keras.Model:
    """Build a custom CNN with augmentation applied only during training."""
    _validate_model_args(num_classes, input_shape)
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the range [0, 1).")

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = create_augmentation(seed)(inputs)

    for index, filters in enumerate((32, 64, 128, 256), start=1):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=f"conv_{index}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{index}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{index}")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"max_pool_{index}")(x)
        if index >= 3:
            x = tf.keras.layers.Dropout(
                rate=dropout_rate / 2,
                name=f"dropout_{index}",
            )(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_features")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def build_transfer_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3,
    seed: int = 42,
    fine_tune: bool = False,
    unfreeze_from_layer: int = 100,
) -> tf.keras.Model:
    """Build an ImageNet-pretrained MobileNetV2 classifier.

    Input images must be RGB floats in [0, 1]. Set ``fine_tune`` to unfreeze
    the later backbone layers while keeping BatchNormalization layers frozen.
    """
    _validate_model_args(num_classes, input_shape)
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the range [0, 1).")

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = create_augmentation(seed)(inputs)
    x = tf.keras.layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenet_input_scaling",
    )(x)
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        name="mobilenetv2_backbone",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_classifier")

    if fine_tune:
        set_fine_tuning(model, unfreeze_from_layer)
    return model


def set_fine_tuning(
    model: tf.keras.Model,
    unfreeze_from_layer: int = 100,
) -> None:
    """Unfreeze later MobileNetV2 layers, leaving BatchNormalization frozen."""
    try:
        backbone = model.get_layer("mobilenetv2_backbone")
    except ValueError as error:
        raise ValueError("model does not contain a MobileNetV2 backbone.") from error

    if not 0 <= unfreeze_from_layer < len(backbone.layers):
        raise ValueError(
            f"unfreeze_from_layer must be between 0 and {len(backbone.layers) - 1}."
        )

    backbone.trainable = True
    for index, layer in enumerate(backbone.layers):
        layer.trainable = (
            index >= unfreeze_from_layer
            and not isinstance(layer, tf.keras.layers.BatchNormalization)
        )
