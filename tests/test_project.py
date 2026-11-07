"""Lightweight tests for preprocessing, splitting, and model explanations."""

import unittest

import numpy as np
import tensorflow as tf

from src.model import build_baseline_model
from src.predict import create_gradcam_overlay
from src.preprocess import preprocess_image
from src.train import split_dataset


class PreprocessingTests(unittest.TestCase):
    def test_preprocesses_rgb_array_to_normalized_target_shape(self) -> None:
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        image[:, :, 0] = 255

        result = preprocess_image(image, image_size=(64, 64))

        self.assertEqual(result.shape, (64, 64, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)
        self.assertGreater(float(result[:, :, 0].mean()), 0.9)

    def test_rejects_invalid_image_size(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_image(np.zeros((8, 8, 3), dtype=np.uint8), image_size=(0, 8))


class DatasetSplitTests(unittest.TestCase):
    def test_split_is_deterministic_stratified_and_70_15_15(self) -> None:
        class_count = 4
        samples_per_class = 100
        labels = [
            label
            for label in range(class_count)
            for _ in range(samples_per_class)
        ]
        paths = [f"class_{label}/image_{index}.jpg" for index, label in enumerate(labels)]

        first = split_dataset(paths, labels, seed=42)
        second = split_dataset(paths, labels, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(
            tuple(len(part[0]) for part in first),
            (280, 60, 60),
        )
        for split_paths, split_labels in first:
            self.assertEqual(len(split_paths), len(split_labels))
            self.assertEqual(set(split_labels), set(range(class_count)))


class GradCamTests(unittest.TestCase):
    def test_creates_rgb_overlay_for_baseline_classifier(self) -> None:
        tf.keras.utils.set_random_seed(42)
        model = build_baseline_model(num_classes=3, seed=42)
        image = np.full((224, 224, 3), 0.5, dtype=np.float32)

        overlay = create_gradcam_overlay(model, image, class_index=0)

        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertGreater(int(overlay.max()), 0)


if __name__ == "__main__":
    unittest.main()
