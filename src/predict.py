"""Run inference and optionally save a Grad-CAM visualization."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.preprocess import preprocess_image

IMAGE_SIZE = (224, 224)


def load_classifier(
    model_path: Path,
    class_names_path: Path,
) -> tuple[tf.keras.Model, list[str]]:
    """Load a Keras model and its matching ordered class-name list."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if not class_names_path.is_file():
        raise FileNotFoundError(f"Class names file does not exist: {class_names_path}")

    with class_names_path.open(encoding="utf-8") as file:
        class_names = json.load(file)
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("class_names.json must contain a non-empty JSON list.")
    if not all(isinstance(name, str) for name in class_names):
        raise ValueError("Every class name must be a string.")

    model = tf.keras.models.load_model(model_path, compile=False)
    output_shape = model.output_shape
    if len(output_shape) != 2 or output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model output shape {output_shape} does not match "
            f"{len(class_names)} class names."
        )
    return model, class_names


def predict_image(
    model: tf.keras.Model,
    class_names: list[str],
    image_path: Path,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Return the top-k classes and confidence scores for one image."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    image = preprocess_image(image_path, image_size=IMAGE_SIZE)
    probabilities = model.predict(image[np.newaxis, ...], verbose=0)[0]
    if probabilities.ndim != 1 or len(probabilities) != len(class_names):
        raise ValueError(
            f"Model returned {len(probabilities)} scores for "
            f"{len(class_names)} classes."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned non-finite prediction scores.")

    top_indices = np.argsort(probabilities)[::-1][: min(top_k, len(class_names))]
    return [
        (class_names[int(index)], float(probabilities[index]))
        for index in top_indices
    ]


def _gradcam_feature_model(model: tf.keras.Model) -> tf.keras.Model:
    """Return a model exposing the final convolutional feature map."""
    try:
        backbone = model.get_layer("mobilenetv2_backbone")
    except ValueError:
        backbone = None

    if backbone is not None:
        image = model.inputs[0]
        x = model.get_layer("training_augmentation")(image, training=False)
        x = model.get_layer("mobilenet_input_scaling")(x)
        feature_maps = backbone(x, training=False)
        predictions = model.get_layer("global_average_pooling")(feature_maps)
        predictions = model.get_layer("classifier_dropout")(
            predictions,
            training=False,
        )
        predictions = model.get_layer("predictions")(predictions)
        return tf.keras.Model(image, [feature_maps, predictions])
    else:
        convolution_layers = [
            layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        if not convolution_layers:
            raise ValueError("No top-level Conv2D layer found for Grad-CAM.")
        feature_maps = convolution_layers[-1].output
        return tf.keras.Model(model.inputs, [feature_maps, model.output])


def save_gradcam(
    model: tf.keras.Model,
    image_path: Path,
    class_index: int,
    output_path: Path,
) -> Path:
    """Save a Grad-CAM heatmap blended with the original input image."""
    image = preprocess_image(image_path, image_size=IMAGE_SIZE)
    overlay = create_gradcam_overlay(model, image, class_index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save Grad-CAM visualization: {output_path}")
    return output_path


def create_gradcam_overlay(
    model: tf.keras.Model,
    image: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """Return an RGB Grad-CAM overlay for a preprocessed RGB image."""
    if image.shape != (*IMAGE_SIZE, 3):
        raise ValueError(f"Expected a preprocessed image of shape {(*IMAGE_SIZE, 3)}.")

    input_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    feature_model = _gradcam_feature_model(model)

    with tf.GradientTape() as tape:
        feature_maps, predictions = feature_model(input_tensor, training=False)
        if not 0 <= class_index < int(predictions.shape[-1]):
            raise ValueError("class_index is outside the model output range.")
        class_score = predictions[:, class_index]
    gradients = tape.gradient(class_score, feature_maps)
    if gradients is None:
        raise ValueError("Could not compute Grad-CAM gradients for this model.")

    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)
    heatmap_array = (heatmap.numpy() * 255).astype(np.uint8)
    heatmap_array = cv2.resize(
        heatmap_array,
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    color_map = cv2.applyColorMap(heatmap_array, cv2.COLORMAP_JET)
    original = (image * 255).clip(0, 255).astype(np.uint8)
    overlay = cv2.addWeighted(
        cv2.cvtColor(original, cv2.COLOR_RGB2BGR),
        0.6,
        color_map,
        0.4,
        0,
    )
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a crop disease from a leaf image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a leaf image.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/baseline_model.keras"))
    parser.add_argument("--class-names", type=Path, default=Path("artifacts/class_names.json"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--grad-cam-output",
        type=Path,
        help="Optional path for a Grad-CAM overlay of the top prediction.",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    return args


def main() -> None:
    args = parse_args()
    model, class_names = load_classifier(args.model, args.class_names)
    predictions = predict_image(model, class_names, args.image, args.top_k)

    print(f"Predicted class: {predictions[0][0]}")
    print(f"Confidence: {predictions[0][1] * 100:.2f}%")
    print("Top predictions:")
    for class_name, confidence in predictions:
        print(f"  {class_name}: {confidence * 100:.2f}%")

    if predictions[0][1] < 0.60:
        print("Warning: confidence is below 60%; treat this prediction with caution.")

    if args.grad_cam_output is not None:
        predicted_index = class_names.index(predictions[0][0])
        output_path = save_gradcam(
            model,
            args.image,
            predicted_index,
            args.grad_cam_output,
        )
        print(f"Saved Grad-CAM visualization: {output_path}")


if __name__ == "__main__":
    main()
