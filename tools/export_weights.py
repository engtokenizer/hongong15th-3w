import json
from pathlib import Path

import numpy as np
import tensorflow as tf


MODEL_PATH = Path("mnist_model.keras")
OUTPUT_PATH = Path("static/model/weights.json")


def serialize_dense(layer: tf.keras.layers.Dense) -> dict:
    kernel, bias = layer.get_weights()
    return {
        "kernel": kernel.astype("float32").reshape(-1).tolist(),
        "bias": bias.astype("float32").tolist(),
        "shape": list(kernel.shape),
    }


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model at {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    if len(dense_layers) < 2:
        raise ValueError("Expected at least two Dense layers.")

    payload = {
        "dense1": serialize_dense(dense_layers[0]),
        "dense2": serialize_dense(dense_layers[1]),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
