Crop Disease Detection from Leaf Images

Upload a photo of a crop leaf and get the predicted crop and disease class (or "healthy"), with a confidence score and the top three predictions. The project compares a custom CNN baseline against a fine-tuned MobileNetV2 transfer-learning model on the PlantVillage dataset.

**[Live Demo](https://crop-disease-detection-oki9bpycmdhegbgyqs4ruf.streamlit.app/)** (the free host may need a few seconds to wake up)

## Highlights

- **Two models compared:** a custom CNN baseline and a MobileNetV2 transfer-learning model with optional fine-tuning.
- **89.18% test accuracy** with MobileNetV2 in a short run on a 4,370-image, 38-class sample (see [Results](#results)).
- **Reproducible splits:** a fixed-seed, stratified 70/15/15 train/validation/test split.
- **Full evaluation:** per-class precision, recall, and F1, confusion-matrix heatmaps, and a baseline-versus-transfer accuracy table.
- **Explainable predictions:** optional Grad-CAM heatmap, and predictions below 60% confidence are flagged as uncertain.
- **Deployed Streamlit app** and a command-line predictor.

## Tech stack

Python, TensorFlow, Keras, MobileNetV2, NumPy, Streamlit

## Project structure

```
.
├── app/
│   └── app.py              # Streamlit app
├── src/
│   ├── preprocess.py       # Loading, resizing, stratified split
│   ├── augment.py          # Data augmentation
│   ├── model.py            # Custom CNN and MobileNetV2 models
│   ├── train.py            # Training CLI
│   ├── evaluate.py         # Metrics, confusion matrices, comparison table
│   └── predict.py          # CLI inference and Grad-CAM
├── tests/
│   └── test_project.py     # Preprocessing, split, and Grad-CAM checks
├── artifacts/              # Created by training and evaluation
├── requirements.txt
├── LICENSE
└── README.md
```

Training, preprocessing, augmentation, evaluation, inference, and the UI are kept in separate modules.

## Getting started

### Requirements

- Python 3.10 to 3.13
- A TensorFlow-supported CPU or GPU environment
- At least 4,000 supported image files (JPG, JPEG, PNG, BMP) in the selected dataset

On native Windows, TensorFlow 2.11 and newer runs on CPU only. Use WSL2 for GPU training.

### Install

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

### Dataset layout

Download the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset) and choose **one** image set (for example `raw/color`). Do not mix color, grayscale, and segmented versions. Pass the directory whose immediate subfolders are class names:

```
dataset/
├── Apple___Apple_scab/
│   ├── image_001.JPG
│   └── ...
├── Apple___healthy/
│   └── ...
└── ...
```

Class folders may contain nested directories. If your archive is laid out as `dataset/train/<class>/...`, pass the `dataset` directory and training will read its `train` subfolder.

The loader checks the 4,000-image minimum, then creates a fixed-seed stratified 70/15/15 split. Training saves the sorted label mapping to `artifacts/class_names.json`. Keep the dataset unchanged and reuse the same seed when evaluating so the test split and class order match training.

## Usage

Run all commands from the project root.

### Train

Recommended: MobileNetV2 with fine-tuning.

```
python -m src.train --dataset-dir "path\to\dataset" --model transfer
```

Custom CNN baseline for comparison:

```
python -m src.train --dataset-dir "path\to\dataset" --model baseline
```

Fine-tuning is on by default for the transfer model; add `--no-fine-tune` to use only the frozen feature extractor. Epochs, batch size, learning rates, output directory, and seed are configurable (`python -m src.train --help`).

Both models use Adam and categorical cross-entropy, with EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks. Checkpoints, the final model, class names, and a training-curves plot are saved to `artifacts/`.

### Evaluate

After training both models into the same artifacts directory:

```
python -m src.evaluate --dataset-dir "path\to\dataset"
```

This reports test accuracy and per-class precision, recall, and F1, and saves a confusion-matrix heatmap for each model plus a baseline-versus-transfer accuracy table under `artifacts/evaluation/`.

### Predict from the command line

```
python -m src.predict --image "path\to\leaf.jpg" --model artifacts\transfer_model.keras
```

Top three predictions are shown by default. To also save a Grad-CAM overlay for the top prediction:

```
python -m src.predict --image "path\to\leaf.jpg" --model artifacts\transfer_model.keras --grad-cam-output artifacts\gradcam.jpg
```

### Run the Streamlit app

```
python -m streamlit run app/app.py
```

The app loads `artifacts/transfer_model.keras` and `artifacts/class_names.json` by default; use the sidebar to pick a different model or class-name file. It shows the predicted class, confidence, and top three predictions, and flags anything below 60% confidence as uncertain. Tick **Show Grad-CAM explanation** before clicking Predict to see which image regions influenced the top class.

## Results

| Setup | Value |
|---|---|
| Dataset sample | 4,370 images, 38 classes |
| Model | MobileNetV2 (3 frozen-backbone epochs + 2 fine-tuning epochs) |
| Held-out test set | 656 images |
| Test accuracy | **89.18%** |

This is a short smoke-test run, not a performance guarantee. Longer training and evaluation on the full, representative PlantVillage dataset are recommended. Grad-CAM shows where the model looked; it does not confirm a prediction is correct.

## Tests

```
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover preprocessing, the stratified split, and Grad-CAM.

## Notes on publishing

The `.gitignore` excludes the local virtual environment, dataset, and generated artifacts, except `artifacts/transfer_model.keras` and `artifacts/class_names.json`, which the hosted app needs for predictions. Do not commit dataset archives or extracted images.

## License

The project code is released under the MIT License (see `LICENSE`). This does not change the terms of the PlantVillage dataset, TensorFlow, or ImageNet-pretrained weights; check those terms before redistributing them.

## Author

**Sneha Banjare** · [GitHub](https://github.com/Snehabanjare) · [LinkedIn](https://www.linkedin.com/in/sneha-banjare-6494b8327)
