import base64
import os
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
import tensorflow as tf

from handwritten_digit_recognizer import DEFAULT_MODEL_PATH, MODEL_INPUT_SIZE, train_and_save


app = Flask(__name__)
MODEL_PATH = Path(DEFAULT_MODEL_PATH)
MODEL = None


def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        train_and_save(MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_data: str) -> np.ndarray:
    header, encoded = image_data.split(",", 1)
    if not header.startswith("data:image"):
        raise ValueError("Invalid image data.")

    decoded = base64.b64decode(encoded)
    image = Image.open(BytesIO(decoded)).convert("L")
    image = image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    sample = np.array(image).astype("float32") / 255.0
    return sample.reshape(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)


@app.route("/")
def index() -> str:
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict() -> tuple[str, int] | tuple[dict, int]:
    global MODEL
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return {"error": "Missing image data."}, 400

    try:
        sample = preprocess_image(payload["image"])
    except (ValueError, OSError, base64.binascii.Error):
        return {"error": "Unable to decode image."}, 400

    if MODEL is None:
        MODEL = load_model()

    probabilities = MODEL.predict(sample, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities)) * 100.0

    return jsonify({"digit": predicted_digit, "confidence": round(confidence, 1)})


if __name__ == "__main__":
    MODEL = load_model()
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
