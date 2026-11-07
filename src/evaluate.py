"""Evaluate baseline and transfer-learning models on the held-out test split."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.train import collect_dataset, make_dataset, split_dataset


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    model_name: str,
) -> None:
    """Save a labeled confusion-matrix heatmap."""
    figure_size = max(10, len(class_names) * 0.35)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    sns.heatmap(
        matrix,
        cmap="Blues",
        annot=len(class_names) <= 15,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_title(f"{model_name} confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.tick_params(axis="x", labelrotation=90)
    axis.tick_params(axis="y", labelrotation=0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def evaluate_model(
    model_path: Path,
    model_name: str,
    test_dataset: tf.data.Dataset,
    true_labels: list[int],
    class_names: list[str],
    output_dir: Path,
) -> dict[str, float | str]:
    """Evaluate one saved model and write its report and confusion heatmap."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path, compile=False)
    probabilities = model.predict(test_dataset, verbose=1)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(class_names):
        raise ValueError(
            f"{model_name} outputs {probabilities.shape}; expected "
            f"(samples, {len(class_names)})."
        )

    predicted_labels = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(len(class_names)),
    )
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )

    report_path = output_dir / f"{model_name}_classification_report.txt"
    report_path.write_text(
        f"Accuracy: {accuracy:.4f}\n\n{report}",
        encoding="utf-8",
    )
    plot_confusion_matrix(
        matrix,
        class_names,
        output_dir / f"{model_name}_confusion_matrix.png",
        model_name,
    )
    print(f"\n{model_name} — test accuracy: {accuracy:.4f}")
    print(report)
    return {"model": model_name, "accuracy": float(accuracy)}


def evaluate(args: argparse.Namespace) -> Path:
    """Evaluate both models on the same stratified test split."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths, labels, discovered_class_names = collect_dataset(args.dataset_dir)

    with args.class_names.open(encoding="utf-8") as file:
        class_names = json.load(file)
    if class_names != discovered_class_names:
        raise ValueError(
            "class_names.json does not match the sorted dataset class directories."
        )

    _, _, test_data = split_dataset(image_paths, labels, args.seed)
    test_dataset = make_dataset(
        *test_data,
        len(class_names),
        args.batch_size,
        training=False,
        seed=args.seed,
    )

    results = [
        evaluate_model(
            model_path,
            model_name,
            test_dataset,
            test_data[1],
            class_names,
            args.output_dir,
        )
        for model_name, model_path in (
            ("baseline", args.baseline_model),
            ("transfer", args.transfer_model),
        )
    ]

    comparison_path = args.output_dir / "model_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("model", "accuracy"))
        writer.writeheader()
        writer.writerows(results)

    print("\nBaseline vs transfer-learning comparison")
    print(f"{'Model':<12} {'Test accuracy':>14}")
    print("-" * 27)
    for result in results:
        print(f"{result['model']:<12} {result['accuracy']:>14.4f}")
    print(f"\nSaved comparison table: {comparison_path}")
    return comparison_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and transfer models on a held-out test split."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--class-names", type=Path, default=Path("artifacts/class_names.json"))
    parser.add_argument("--baseline-model", type=Path, default=Path("artifacts/baseline_model.keras"))
    parser.add_argument("--transfer-model", type=Path, default=Path("artifacts/transfer_model.keras"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive.")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
