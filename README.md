# Handwritten Digit Recognizer (Static)

This app runs fully in the browser. The MNIST model weights are exported to
`static/model/weights.json` and loaded by the frontend.

## Run locally

You can open `index.html` directly or serve the folder with any static server.

## Update the model

If you retrain the model in `tools/mnist_model.keras`, re-export the weights:

```bash
python tools/export_weights.py
```

The export script requires TensorFlow, NumPy, and Pillow installed locally.

Then commit the updated `static/model/weights.json`.
