"""Split a class-folder image dataset and train a selected classifier."""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.model import build_baseline_model, build_transfer_model, set_fine_tuning
from src.preprocess import preprocess_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGE_SIZE = (224, 224)
MIN_DATASET_IMAGES = 4_000


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def collect_dataset(dataset_dir: Path) -> tuple[list[str], list[int], list[str]]:
    """Collect image paths and integer labels from class-named subdirectories."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    class_root = dataset_dir / "train"
    if not class_root.is_dir():
        class_root = dataset_dir

    class_dirs = sorted(path for path in class_root.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError(
            f"Expected at least two class subdirectories in {class_root}."
        )

    image_paths: list[str] = []
    labels: list[int] = []
    class_names = [path.name for path in class_dirs]

    for label, class_dir in enumerate(class_dirs):
        class_images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_images:
            raise ValueError(f"No supported images found in class: {class_dir}")
        image_paths.extend(str(path) for path in class_images)
        labels.extend([label] * len(class_images))

    if len(image_paths) < MIN_DATASET_IMAGES:
        raise ValueError(
            f"Dataset contains {len(image_paths)} images; at least "
            f"{MIN_DATASET_IMAGES} are required."
        )

    return image_paths, labels, class_names


def split_dataset(
    image_paths: list[str],
    labels: list[int],
    seed: int,
) -> tuple[tuple[list[str], list[int]], tuple[list[str], list[int]], tuple[list[str], list[int]]]:
    """Create stratified 70/15/15 train, validation, and test splits."""
    paths = np.asarray(image_paths)
    targets = np.asarray(labels)
    train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
        paths,
        targets,
        test_size=0.30,
        random_state=seed,
        stratify=targets,
    )
    validation_paths, test_paths, validation_labels, test_labels = train_test_split(
        remaining_paths,
        remaining_labels,
        test_size=0.50,
        random_state=seed,
        stratify=remaining_labels,
    )
    return (
        (train_paths.tolist(), train_labels.tolist()),
        (validation_paths.tolist(), validation_labels.tolist()),
        (test_paths.tolist(), test_labels.tolist()),
    )


def _read_preprocessed_image(path: bytes | np.ndarray) -> np.ndarray:
    if isinstance(path, bytes):
        image_path = path.decode("utf-8")
    else:
        image_path = path.item().decode("utf-8")
    return preprocess_image(image_path, image_size=IMAGE_SIZE)


def make_dataset(
    paths: list[str],
    labels: list[int],
    num_classes: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    """Build a tf.data pipeline that preprocesses images with OpenCV."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.numpy_function(_read_preprocessed_image, [path], tf.float32)
        image.set_shape((*IMAGE_SIZE, 3))
        one_hot_label = tf.one_hot(label, depth=num_classes)
        return image, one_hot_label

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def create_callbacks(checkpoint_path: Path) -> list[tf.keras.callbacks.Callback]:
    """Create training callbacks that monitor validation accuracy."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def plot_training_curves(histories: list[tf.keras.callbacks.History], output_path: Path) -> None:
    """Save combined accuracy and loss plots for the training stages."""
    metrics = ("accuracy", "loss")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    for axis, metric in zip(axes, metrics):
        offset = 0
        for index, history in enumerate(histories, start=1):
            values = history.history.get(metric, [])
            validation_values = history.history.get(f"val_{metric}", [])
            epochs = np.arange(offset + 1, offset + len(values) + 1)
            axis.plot(epochs, values, label=f"Train (stage {index})")
            axis.plot(
                epochs,
                validation_values,
                linestyle="--",
                label=f"Validation (stage {index})",
            )
            offset += len(values)
        axis.set_title(metric.capitalize())
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric.capitalize())
        axis.legend()
        axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def train(args: argparse.Namespace) -> Path:
    """Train the requested model and return the saved final model path."""
    set_random_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, labels, class_names = collect_dataset(args.dataset_dir)
    (train_data, validation_data, test_data) = split_dataset(
        image_paths,
        labels,
        args.seed,
    )
    del test_data  # The test split is reserved for final evaluation.

    with (output_dir / "class_names.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    print(
        "Split sizes — "
        f"train: {len(train_data[0])}, "
        f"validation: {len(validation_data[0])}, "
        f"test: {len(image_paths) - len(train_data[0]) - len(validation_data[0])}"
    )
    print(f"Classes: {len(class_names)}")

    train_dataset = make_dataset(
        *train_data,
        len(class_names),
        args.batch_size,
        training=True,
        seed=args.seed,
    )
    validation_dataset = make_dataset(
        *validation_data,
        len(class_names),
        args.batch_size,
        training=False,
        seed=args.seed,
    )

    if args.model == "baseline":
        model = build_baseline_model(len(class_names), seed=args.seed)
    else:
        model = build_transfer_model(len(class_names), seed=args.seed)

    histories: list[tf.keras.callbacks.History] = []
    checkpoint_paths: list[Path] = []
    stages = [(args.epochs, args.learning_rate, "initial")]
    if args.model == "transfer" and args.fine_tune:
        stages.append((args.fine_tune_epochs, args.fine_tune_learning_rate, "fine_tune"))

    for stage_index, (epochs, learning_rate, stage_name) in enumerate(stages):
        if stage_index == 1:
            model.load_weights(checkpoint_paths[0])
            set_fine_tuning(model, args.unfreeze_from_layer)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        checkpoint_path = output_dir / f"{args.model}_{stage_name}.weights.h5"
        checkpoint_paths.append(checkpoint_path)
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=create_callbacks(checkpoint_path),
            shuffle=False,
        )
        histories.append(history)

    best_stage_index = max(
        range(len(histories)),
        key=lambda index: max(histories[index].history["val_accuracy"]),
    )
    model.load_weights(checkpoint_paths[best_stage_index])
    model_path = output_dir / f"{args.model}_model.keras"
    model.save(model_path)
    plot_training_curves(histories, output_dir / f"{args.model}_training_curves.png")
    print(f"Saved model: {model_path}")
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PlantVillage image classifier from class folders."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per class.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", choices=("baseline", "transfer"), default="baseline")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-from-layer", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fine-tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fine-tune later MobileNetV2 layers after transfer learning.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.fine_tune_epochs < 1 or args.batch_size < 1:
        parser.error("epoch counts and batch size must be positive.")
    if args.learning_rate <= 0 or args.fine_tune_learning_rate <= 0:
        parser.error("learning rates must be positive.")
    return args


if __name__ == "__main__":
    train(parse_args())
