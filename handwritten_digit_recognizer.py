import argparse
from pathlib import Path
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf


DEFAULT_MODEL_PATH = Path("mnist_model.keras")
CANVAS_SIZE = 280
MODEL_INPUT_SIZE = 28
BRUSH_RADIUS = 10


def train_and_save(model_path: Path, epochs: int = 3) -> None:
    """Train a simple MNIST classifier and save it to disk."""
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    model.save(model_path)


class DigitCanvas:
    def __init__(self, root: tk.Tk, model: tf.keras.Model) -> None:
        self.root = root
        self.model = model
        self.root.title("Handwritten Digit Recognizer")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.result_label = tk.Label(root, text="Draw a digit and click Predict", font=("Helvetica", 14))
        self.result_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        predict_button = tk.Button(root, text="Predict", command=self.predict_digit)
        predict_button.grid(row=2, column=0, sticky="ew", padx=10, pady=5)

        clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        clear_button.grid(row=2, column=1, sticky="ew", padx=10, pady=5)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.drawer = ImageDraw.Draw(self.image)

    def draw(self, event: tk.Event) -> None:
        x1 = event.x - BRUSH_RADIUS
        y1 = event.y - BRUSH_RADIUS
        x2 = event.x + BRUSH_RADIUS
        y2 = event.y + BRUSH_RADIUS
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.drawer.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.drawer = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit and click Predict")

    def predict_digit(self) -> None:
        resized = self.image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        sample = np.array(resized).astype("float32") / 255.0
        sample = sample.reshape(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

        probabilities = self.model.predict(sample, verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100.0
        self.result_label.config(text=f"Prediction: {predicted_digit} ({confidence:.1f}%)")


def load_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        train_and_save(model_path)
    return tf.keras.models.load_model(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST and recognize handwritten digits.")
    parser.add_argument("--train", action="store_true", help="Train the model before launching the UI.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the saved model.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    args = parser.parse_args()

    if args.train or not args.model.exists():
        train_and_save(args.model, epochs=args.epochs)

    model = tf.keras.models.load_model(args.model)
    root = tk.Tk()
    DigitCanvas(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
