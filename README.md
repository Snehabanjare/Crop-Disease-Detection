# Crop Disease Detection from Leaf Images

Classify a crop leaf image into a PlantVillage crop/disease class (or healthy)
using a custom CNN or a fine-tuned MobileNetV2 model. The Streamlit app shows
the predicted class, confidence, and top-three predictions.

## Project structure

```text
.
├── app/
│   └── app.py
├── src/
│   ├── augment.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── tests/
│   └── test_project.py
├── artifacts/                 # Created by training/evaluation
├── LICENSE
├── requirements.txt
├── .gitignore
└── README.md
```

The layout keeps model training, preprocessing, augmentation, evaluation,
inference, and the UI in separate modules. For additional structure examples,
see the [apple-leaf-classifier](https://github.com/boluwatife-py/apple-leaf-classifier)
repository's `models/` and `utils/` directories and
[plant-disease-detection](https://github.com/cadu-ers/plant-disease-detection)'s
modular `src/` layout. The [PlantVillage dataset repository](https://github.com/spMohanty/PlantVillage-Dataset)
separates its color, grayscale, and segmented image sets; this project expects
one selected image set arranged in class folders.

## Requirements

- Python 3.10–3.13 (the installed TensorFlow build was tested with Python 3.13)
- A TensorFlow-supported CPU or GPU environment
- At least **4,000 supported image files total** in the selected dataset

On native Windows, TensorFlow 2.11 and newer runs on CPU; use WSL2 for
TensorFlow's supported CUDA GPU setup.

Install dependencies from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset layout

Pass the directory whose immediate subdirectories are class names. Each class
folder may contain images in nested directories. Supported formats are JPG,
JPEG, PNG, and BMP.

```text
dataset/
├── Apple___Apple_scab/
│   ├── image_001.JPG
│   └── ...
├── Apple___healthy/
│   └── ...
└── ...
```

For archives laid out as `dataset/train/<class>/...`, pass the `dataset`
directory; training detects and reads its `train` subdirectory.

For the original PlantVillage repository, choose one image set (for example
`raw/color`) rather than mixing color, grayscale, and segmented versions.
The loader checks the **4,000-image minimum** before training and uses a fixed
seed to create a stratified 70/15/15 train/validation/test split. Training
persists the sorted label mapping to `artifacts/class_names.json`. Keep the
dataset unchanged and use the same seed when evaluating so the test split and
class order match training.

## Train

Run commands from the project root. For the recommended pretrained model with
fine-tuning:

```powershell
python -m src.train --dataset-dir "path\to\dataset" --model transfer
```

To train the custom CNN baseline for comparison:

```powershell
python -m src.train --dataset-dir "path\to\dataset" --model baseline
```

Fine-tuning is enabled by default for the transfer model. To use only its
frozen feature extractor, add `--no-fine-tune`. Epoch counts, batch size,
learning rates, output directory, and random seed can be changed through the
CLI options shown by:

```powershell
python -m src.train --help
```

Both training modes use Adam and categorical cross-entropy, with
EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau callbacks. Checkpoints,
the final Keras model, class names, and a training-curves PNG are written to
`artifacts/` by default.

## Evaluate

Train both models into the same artifacts directory, then run:

```powershell
python -m src.evaluate --dataset-dir "path\to\dataset"
```

Evaluation reports test accuracy and per-class precision, recall, and F1;
it saves a confusion-matrix heatmap for each model and a baseline-versus-
transfer accuracy table under `artifacts/evaluation/`. Override the dataset,
model, class-name, or output paths with `python -m src.evaluate --help`.

## Predict from the command line

```powershell
python -m src.predict --image "path\to\leaf.jpg" --model artifacts\transfer_model.keras
```

Top-three predictions are shown by default. To save an optional Grad-CAM
overlay for the top prediction:

```powershell
python -m src.predict --image "path\to\leaf.jpg" --model artifacts\transfer_model.keras --grad-cam-output artifacts\gradcam.jpg
```

## Run the Streamlit app

```powershell
python -m streamlit run app/app.py
```

By default, the app looks for `artifacts/transfer_model.keras` and
`artifacts/class_names.json`. Use the sidebar to select a different model and
class-name file. Predictions below 60% confidence are flagged as uncertain.
Enable **Show Grad-CAM explanation** before clicking Predict to see a heatmap
overlay for the top class. Grad-CAM indicates image regions that influenced
the model; it does not verify that a prediction is correct.

## Smoke-test result

The supplied 4,370-image, 38-class sample was used for a short MobileNetV2
run (3 frozen-backbone epochs plus 2 fine-tuning epochs). It reached **89.18%
accuracy** on the held-out 656-image test split. This is a smoke-test result,
not a performance guarantee; longer training and evaluation on a larger,
representative dataset are recommended. The corresponding model and class
mapping are saved as `artifacts/transfer_model.keras` and
`artifacts/class_names.json`.

## Tests

Run the lightweight preprocessing, split, and Grad-CAM checks from the project
root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Publishing to GitHub

The `.gitignore` excludes the local virtual environment, dataset, and generated
model artifacts except for `artifacts/transfer_model.keras` and
`artifacts/class_names.json`, which the hosted app needs to make predictions.
Do not commit uploaded dataset archives or extracted images. Before
redistributing any dataset or third-party model weights, check their applicable
terms. This repository includes an MIT license for its project code; it does
not change the terms for PlantVillage data, TensorFlow, or ImageNet-pretrained
weights.
