# Handwritten Digit Recognizer (Static)

This app runs fully in the browser. The MNIST model weights are exported to
`static/model/weights.json` and loaded by the frontend.

## Run locally

You can open `index.html` directly or serve the folder with any static server.

## Update the model

If you retrain `mnist_model.keras`, re-export the weights:

```bash
venv/bin/python tools/export_weights.py
```

Then commit the updated `static/model/weights.json`.
