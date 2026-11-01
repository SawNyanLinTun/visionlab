
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw

from model import ShapeCNN


st.set_page_config(
    page_title="AI Vision Lab",
    page_icon="🔬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CHECKPOINT = MODEL_DIR / "m4_shapes_cnn.pt"

st.title("🔬 AI Vision Lab")
st.subheader("M4 — Image Classification with CNN")

st.write(
    "Explore how a Convolutional Neural Network "
    "recognizes circles, squares, and triangles."
)


@st.cache_resource
def load_model():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    model = ShapeCNN()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint["classes"]


def generate_shape(shape, seed):
    rng = np.random.default_rng(seed)

    background = tuple(
        int(x) for x in rng.integers(180, 256, size=3)
    )
    color = tuple(
        int(x) for x in rng.integers(0, 151, size=3)
    )

    image = Image.new("RGB", (64, 64), background)
    draw = ImageDraw.Draw(image)

    width = int(rng.integers(20, 39))
    x = int(rng.integers(5, 64 - width - 4))
    y = int(rng.integers(5, 64 - width - 4))

    if shape == "circle":
        draw.ellipse(
            (x, y, x + width, y + width),
            fill=color
        )
    elif shape == "square":
        draw.rectangle(
            (x, y, x + width, y + width),
            fill=color
        )
    else:
        draw.polygon(
            [
                (x + width // 2, y),
                (x + width, y + width),
                (x, y + width)
            ],
            fill=color
        )

    return image


def predict(model, image):
    image = image.convert("RGB").resize((64, 64))

    array = np.asarray(
        image, dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1).copy()
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(
            logits, dim=1
        )[0].numpy()

    return probabilities


if not CHECKPOINT.exists():
    st.error(
        "Model checkpoint missing. Upload "
        "models/m4_shapes_cnn.pt to GitHub."
    )
    st.stop()

try:
    model, classes = load_model()
except Exception as error:
    st.error("The model could not be loaded.")
    st.exception(error)
    st.stop()


tab1, tab2, tab3 = st.tabs(
    ["Try the AI", "Training Results", "Learn CNN"]
)

with tab1:
    st.header("Test the Shape Classifier")

    source = st.radio(
        "Choose image source",
        ["Generate a shape", "Upload an image"],
        horizontal=True
    )

    image = None

    if source == "Generate a shape":
        shape = st.selectbox("Shape", classes)
        seed = st.slider("Image variation", 0, 999, 42)

        image = generate_shape(shape, seed)

    else:
        uploaded = st.file_uploader(
            "Upload a PNG or JPG image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    if image is not None:
        left, right = st.columns(2)

        with left:
            st.image(
                image,
                caption="Input image",
                width=280
            )

        probabilities = predict(model, image)
        predicted_index = int(np.argmax(probabilities))

        with right:
            st.subheader("AI Prediction")
            st.success(classes[predicted_index].capitalize())

            for label, probability in zip(
                classes, probabilities
            ):
                st.progress(
                    float(probability),
                    text=f"{label}: {probability:.1%}"
                )

            st.caption(
                "Confidence scores represent the model's "
                "outputs, not guaranteed correctness."
            )

    st.info(
        "Try changing the shape, colour, or position. "
        "How do these changes influence the prediction?"
    )


with tab2:
    st.header("Model Performance")

    history_file = MODEL_DIR / "m4_training_history.csv"
    metrics_file = MODEL_DIR / "m4_test_metrics.json"

    if history_file.exists():
        history = pd.read_csv(history_file)

        if "epoch" not in history.columns:
            history["epoch"] = range(
                1, len(history) + 1
            )

        st.subheader("Training and Validation Accuracy")
        st.line_chart(
            history.set_index("epoch")[
                ["train_accuracy", "val_accuracy"]
            ]
        )

        st.subheader("Training and Validation Loss")
        st.line_chart(
            history.set_index("epoch")[
                ["train_loss", "val_loss"]
            ]
        )

    if metrics_file.exists():
        with open(metrics_file) as file:
            metrics = json.load(file)

        st.metric(
            "Test Accuracy",
            f"{metrics['test_accuracy']:.2%}"
        )

        st.subheader("Confusion Matrix")
        matrix = metrics[
            "confusion_matrix_rows_true_cols_pred"
        ]

        matrix_df = pd.DataFrame(
            matrix,
            index=classes,
            columns=classes
        )

        st.dataframe(matrix_df)
        st.caption(
            "Rows represent actual labels; "
            "columns represent predicted labels."
        )

    st.warning(
        "This CNN was trained on synthetic shapes. "
        "High test accuracy does not guarantee "
        "performance on real-world images."
    )


with tab3:
    st.header("How Does a CNN Work?")

    st.markdown("""
    **1. Input image**

    An image is converted into numerical pixel values.

    **2. Convolutional layers**

    The network learns visual patterns from the images.

    **3. ReLU activation**

    Introduces non-linearity so the model can learn
    complex relationships.

    **4. Max pooling**

    Reduces spatial dimensions while retaining
    important features.

    **5. Fully connected layers**

    Combine learned features to classify the image.

    **6. Output**

    The model produces scores for three classes:
    circle, square, and triangle.
    """)

    st.subheader("Student Reflection Questions")

    st.markdown("""
    - Why do we separate training and testing images?
    - Does a high confidence score guarantee correctness?
    - What happens if we test the model on real photographs?
    - Why should we examine incorrect predictions?
    """)
