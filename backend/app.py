import html
import sys
from gradcam import make_gradcam_heatmap, create_gradcam_overlay
from huggingface_hub import hf_hub_download
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
if not (ROOT / "backend").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from backend.preprocessing import load_data 

MODEL_PATH = r"models\model_knee_02.h5"

CLASSES = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
ALLOWED_TYPES = ["png", "jpg", "jpeg"]

st.set_page_config(
    page_title="Knee X-Ray Classifier",
    page_icon="🦴",
    layout="centered",
)


@st.cache_resource
def get_model():
    from tensorflow.keras.models import load_model  # type: ignore

    return load_model(str(MODEL_PATH))


def predict(image: Image.Image) -> dict:

    model = get_model()

    processed_image = load_data(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    index = int(np.argmax(predictions))

    heatmap = make_gradcam_heatmap(
        processed_image,
        model,
        index
    )

    original = image.convert("L")
    original = original.resize((200, 200))
    original = np.array(original)

    overlay = create_gradcam_overlay(
        original,
        heatmap
    )

    return {
        "prediction": CLASSES[index],
        "confidence": float(predictions[index]),
        "probabilities": {
            c: float(predictions[i])
            for i, c in enumerate(CLASSES)
        },
        "gradcam": overlay,
    }

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {background: #ffffff; color: #18181b;}
    .block-container {max-width: 520px; padding-top: 72px;}
    h1 {font-size: 1.5rem !important; font-weight: 600 !important;
        letter-spacing: -0.01em; padding: 0 !important; margin-bottom: 6px;}
    .subtitle {color: #6b6b72; font-size: 0.9rem; margin: 0 0 32px;}

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #f7f7f8; border: 1px solid #e2e2e5 !important;
        border-radius: 6px; padding: 8px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent; border: 1px dashed #e2e2e5; border-radius: 6px;
    }
    [data-testid="stFileUploader"] label p {font-weight: 500;}
    [data-testid="stFileUploader"] small {color: #6b6b72;}
    [data-testid="stFileUploader"] button {
        background: #18181b; color: #ffffff; border: 1px solid #18181b;
        border-radius: 6px; font-weight: 500;
    }

    .stButton > button {
        width: 100%; border-radius: 6px; font-size: 0.9rem;
        background: transparent; color: #18181b; border: 1px solid #e2e2e5;
    }
    .stButton > button:hover {background: #eeeeef; border-color: #c8c8cc; color: #18181b;}
    .stButton > button[kind="primary"] {
        background: #18181b; color: #ffffff; border-color: #18181b; font-weight: 500;
    }
    .stButton > button[kind="primary"]:hover {
        background: #3f3f46; border-color: #3f3f46; color: #ffffff;
    }

    [data-testid="stImage"] img {
        border: 1px solid #e2e2e5; border-radius: 6px; background: #000;
        max-height: 320px; object-fit: contain;
    }
    .filename {color: #6b6b72; font-size: 0.85rem; overflow: hidden;
        text-overflow: ellipsis; white-space: nowrap; margin-top: 6px;}

    .summary {display: flex; gap: 40px; margin: 8px 0 20px;}
    .label {color: #6b6b72; font-size: 0.8rem; margin: 0;}
    .value {font-size: 1.6rem; font-weight: 600; letter-spacing: -0.01em; margin: 2px 0 0;}
    .bar {display: grid; grid-template-columns: 76px 1fr 52px; align-items: center;
        gap: 12px; font-size: 0.85rem; color: #6b6b72; margin-bottom: 10px;}
    .bar.active {color: #18181b;}
    .track {height: 6px; background: #e8e8ea; border-radius: 3px; overflow: hidden;}
    .fill {height: 100%; background: #b5b5bb; border-radius: 3px;}
    .bar.active .fill {background: #2563eb;}
    .pct {text-align: right; font-variant-numeric: tabular-nums;}
    </style>
    """,
    unsafe_allow_html=True,
)


if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "result" not in st.session_state:
    st.session_state.result = None
if "result_for" not in st.session_state:
    st.session_state.result_for = None


def reset():
    st.session_state.uploader_key += 1
    st.session_state.result = None
    st.session_state.result_for = None


def run_prediction(uploaded_file):
    """Returns (result, error_message)."""
    try:
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError):
        return None, "The selected file is not a valid JPG or PNG image."

    try:
        return predict(image), None
    except Exception as e:
        return None, f"Prediction failed: {type(e).__name__}: {e}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_result(result: dict):
    prediction = result["prediction"]
    probabilities = result["probabilities"]

    st.markdown(
        f"""
        <div class="summary">
          <div><p class="label">Prediction</p><p class="value">{html.escape(prediction)}</p></div>
          <div><p class="label">Confidence</p><p class="value">{pct(result["confidence"])}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bars = ""
    for name in CLASSES:
        p = probabilities.get(name, 0.0)
        active = " active" if name == prediction else ""
        bars += (
            f'<div class="bar{active}"><span>{name}</span>'
            f'<span class="track"><div class="fill" style="width:{p * 100:.1f}%"></div></span>'
            f'<span class="pct">{pct(p)}</span></div>'
        )
    st.markdown(bars, unsafe_allow_html=True)
    st.markdown("### Grad-CAM")

    st.image(
        result["gradcam"],
        caption="Regions influencing the prediction",
        use_container_width=True
    )


st.title("Knee X-Ray Classifier")
st.markdown(
    '<p class="subtitle">Upload a knee X-ray to estimate osteoarthritis severity.</p>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    uploaded = st.file_uploader(
        "Upload X-ray",
        type=ALLOWED_TYPES,
        help="PNG or JPG",
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded is None:
        st.caption("PNG or JPG")

    if uploaded is not None:
        file_id = (uploaded.name, uploaded.size)

        # A different file was chosen -> drop the old result.
        if st.session_state.result_for != file_id:
            st.session_state.result = None

        st.image(uploaded, use_container_width=True)
        st.markdown(
            f'<div class="filename">{html.escape(uploaded.name)}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.result is None:
            if st.button("Predict", type="primary"):
                with st.spinner("Analyzing..."):
                    result, error = run_prediction(uploaded)
                if error:
                    st.error(error)
                else:
                    st.session_state.result = result
                    st.session_state.result_for = file_id
                    st.rerun()
        else:
            render_result(st.session_state.result)
            st.button("Analyze Another Image", on_click=reset)