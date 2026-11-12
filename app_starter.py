"""Cwtch Rescue intake photo preparation app (starter).

Run from the folder that holds outputs/prep_settings.json:
    streamlit run app_starter.py

The app already runs. Work through TODO 1 to TODO 6 in order and refresh the browser after each one.
"""
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from skimage import exposure, filters, transform, util

SETTINGS_FILE = Path("outputs/prep_settings.json")


# ---- Helpers given in the notebook (complete) ----
def sp_level(grey):
    smooth = filters.median(grey, np.ones((3, 3)))
    return float((np.abs(grey - smooth) > 0.4).mean())


def noise_level(grey):
    h, w = grey.shape
    g = grey[: h // 2 * 2, : w // 2 * 2]
    diagonal = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    return float(np.median(np.abs(diagonal)) / 0.6745)


# TODO 1: replace these two placeholders with YOUR clean_image and prepare_image from the notebook
# (Tasks 3 and 4). Copy them exactly. The app must do what the pipeline does.
def clean_image(grey, s):
    return grey, []


def prepare_image(img, s, clean=True):
    grey = util.img_as_float(np.asarray(img.convert("L")))
    small = transform.resize(grey, (s["size"], s["size"]))
    return small, ["placeholder: resized only"]
# ---------------------------------------------------


st.set_page_config(page_title="Cwtch Rescue intake photos")
st.title("Cwtch Rescue: prepare an intake photo")

if not SETTINGS_FILE.exists():
    st.error(f"{SETTINGS_FILE} not found. Run the notebook first, then start the app from the same folder.")
    st.stop()
settings = json.loads(SETTINGS_FILE.read_text())
st.caption(f"Settings read from {SETTINGS_FILE}")

uploaded = st.file_uploader("Upload an intake photo", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a JPG or PNG to start.")
    st.stop()

# TODO 2: open the upload with Image.open(io.BytesIO(uploaded.getvalue())) and call img.load().
# Wrap both in try/except. On failure, show st.error(...) with a helpful message and call st.stop().
img = Image.open(io.BytesIO(uploaded.getvalue()))

# TODO 3: quality report before preparation. Convert to a 0 to 1 greyscale array, then show size,
# colour mode, brightness, contrast, sp_level and noise_level in st.table(...).
# If contrast is below settings["blank_contrast"], show st.error and st.stop().
# Show st.warning(...) for salt and pepper, grainy noise, darkness, or a side shorter than settings["size"].

# TODO 4: prepare the image with prepare_image(img, settings). Then standardise it with
# settings["train_mean"] and settings["train_std"] (read from the file, never recomputed here).
prepared, steps = prepare_image(img, settings)

# TODO 5: show the original and the prepared image side by side with st.columns(2),
# then list every step with its values, and any step that was skipped and why.
st.image(np.kron(prepared, np.ones((4, 4))), caption="Prepared", clamp=True, width=280)
st.write(steps)

# TODO 6: add st.download_button(...) for the prepared image as PNG and for a text summary.
# Hint: save with Image.fromarray((prepared * 255).round().astype(np.uint8)).save(buffer, format="PNG")
