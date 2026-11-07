"""Streamlit interface for leaf disease classification."""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.preprocess import preprocess_image
from src.predict import create_gradcam_overlay

IMAGE_SIZE = (224, 224)
DEFAULT_MODEL_PATH = Path("artifacts/transfer_model.keras")
DEFAULT_CLASS_NAMES_PATH = Path("artifacts/class_names.json")


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")
    return tf.keras.models.load_model(path, compile=False)


@st.cache_data
def load_class_names(class_names_path: str) -> list[str]:
    path = Path(class_names_path)
    if not path.is_file():
        raise FileNotFoundError(f"Class names file does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        class_names = json.load(file)
    if (
        not isinstance(class_names, list)
        or not class_names
        or not all(isinstance(name, str) for name in class_names)
    ):
        raise ValueError("Class names file must contain a non-empty list of strings.")
    return class_names


def describe_class(class_name: str) -> tuple[str, str]:
    """Split PlantVillage-style labels into crop and disease/health status."""
    crop, separator, condition = class_name.partition("___")
    if not separator:
        return class_name.replace("_", " "), ""
    return crop.replace("_", " "), condition.replace("_", " ")


def main() -> None:
    st.set_page_config(
        page_title="Crop Disease Detection",
        page_icon="🌿",
        layout="centered",
    )
    st.title("Crop Disease Detection")
    st.write("Upload a leaf image to predict its crop and health condition.")
    st.info(
        "This is a research/demo aid, not a medical or agricultural diagnosis. "
        "It was trained on PlantVillage images, which may differ from field photos. "
        "Confidence is the model's score, not a guarantee of correctness."
    )

    with st.sidebar:
        st.header("Model settings")
        model_path = st.text_input("Model file", str(DEFAULT_MODEL_PATH))
        class_names_path = st.text_input(
            "Class names file",
            str(DEFAULT_CLASS_NAMES_PATH),
        )

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=("jpg", "jpeg", "png", "bmp"),
    )
    if uploaded_file is None:
        st.info("Upload a JPG, PNG, or BMP leaf image to get started.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"The uploaded file could not be decoded as an image: {error}")
        return

    st.image(image, caption="Uploaded leaf image", use_container_width=True)
    show_gradcam = st.checkbox(
        "Show Grad-CAM explanation",
        help="Highlights image regions that influenced the top prediction.",
    )
    predict_button = st.button("Predict", type="primary")
    if not predict_button:
        return

    try:
        model = load_model(model_path)
        class_names = load_class_names(class_names_path)
        if model.output_shape[-1] != len(class_names):
            raise ValueError(
                f"Model has {model.output_shape[-1]} outputs but "
                f"class_names.json has {len(class_names)} labels."
            )

        processed_image = preprocess_image(
            np.asarray(image),
            image_size=IMAGE_SIZE,
        )
        probabilities = model.predict(processed_image[np.newaxis, ...], verbose=0)[0]
        if len(probabilities) != len(class_names) or not np.all(
            np.isfinite(probabilities)
        ):
            raise ValueError("Model returned invalid prediction scores.")
    except (FileNotFoundError, OSError, ValueError, tf.errors.OpError) as error:
        st.error(f"Could not run prediction: {error}")
        return

    top_indices = np.argsort(probabilities)[::-1][: min(3, len(class_names))]
    predicted_index = int(top_indices[0])
    predicted_class = class_names[predicted_index]
    crop, condition = describe_class(predicted_class)
    confidence = float(probabilities[predicted_index])

    st.subheader("Prediction")
    st.markdown(f"**Crop:** {crop}")
    if condition:
        st.markdown(f"**Disease / status:** {condition}")
    st.metric("Confidence", f"{confidence:.1%}")

    if confidence < 0.60:
        st.warning(
            "Confidence is below 60%. Treat this result as uncertain and "
            "consider taking a clearer photo or checking another leaf."
        )
    else:
        st.caption(
            "A high confidence score can still be wrong. Check the image quality "
            "and compare with other evidence."
        )

    st.subheader("Top 3 predictions")
    for rank, index in enumerate(top_indices, start=1):
        st.write(f"{rank}. {class_names[int(index)]}: {probabilities[index]:.1%}")

    if show_gradcam:
        try:
            overlay = create_gradcam_overlay(
                model,
                processed_image,
                predicted_index,
            )
        except (ValueError, tf.errors.OpError) as error:
            st.error(f"Could not create Grad-CAM visualization: {error}")
        else:
            st.subheader("Grad-CAM explanation")
            st.image(
                overlay,
                caption=(
                    "Highlighted regions show where the model focused; "
                    "they are not proof that the prediction is correct."
                ),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
