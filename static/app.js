const canvas = document.getElementById("digit-canvas");
const ctx = canvas.getContext("2d");
const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-btn");
const result = document.getElementById("result");

let drawing = false;
const MODEL_URL = "/static/model/weights.json";
let modelWeights = null;

const setBackground = () => {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
};

const drawStroke = (x, y) => {
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.fill();
};

const getCanvasPos = (event) => {
  const rect = canvas.getBoundingClientRect();
  const clientX = event.touches ? event.touches[0].clientX : event.clientX;
  const clientY = event.touches ? event.touches[0].clientY : event.clientY;
  return {
    x: clientX - rect.left,
    y: clientY - rect.top,
  };
};

const startDrawing = (event) => {
  drawing = true;
  const pos = getCanvasPos(event);
  drawStroke(pos.x, pos.y);
};

const stopDrawing = () => {
  drawing = false;
};

const draw = (event) => {
  if (!drawing) return;
  const pos = getCanvasPos(event);
  drawStroke(pos.x, pos.y);
};

const clearCanvas = () => {
  setBackground();
  result.textContent = "Prediction: -";
};

const loadModel = async () => {
  if (modelWeights) return modelWeights;
  result.textContent = "Loading model...";
  const response = await fetch(MODEL_URL);
  if (!response.ok) {
    throw new Error("Model file not found.");
  }
  const data = await response.json();
  modelWeights = {
    dense1: {
      kernel: new Float32Array(data.dense1.kernel),
      bias: new Float32Array(data.dense1.bias),
      outSize: data.dense1.shape[1],
    },
    dense2: {
      kernel: new Float32Array(data.dense2.kernel),
      bias: new Float32Array(data.dense2.bias),
      outSize: data.dense2.shape[1],
    },
  };
  return modelWeights;
};

const getInputVector = () => {
  const smallCanvas = document.createElement("canvas");
  smallCanvas.width = 28;
  smallCanvas.height = 28;
  const smallCtx = smallCanvas.getContext("2d");
  smallCtx.drawImage(canvas, 0, 0, smallCanvas.width, smallCanvas.height);
  const imageData = smallCtx.getImageData(0, 0, 28, 28).data;
  const input = new Float32Array(28 * 28);
  for (let i = 0; i < 28 * 28; i += 1) {
    input[i] = imageData[i * 4] / 255;
  }
  return input;
};

const dense = (input, kernel, bias, outSize) => {
  const output = new Float32Array(outSize);
  for (let j = 0; j < outSize; j += 1) {
    let sum = bias[j];
    for (let i = 0; i < input.length; i += 1) {
      sum += input[i] * kernel[i * outSize + j];
    }
    output[j] = sum;
  }
  return output;
};

const relu = (input) => {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    output[i] = input[i] > 0 ? input[i] : 0;
  }
  return output;
};

const softmax = (logits) => {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
    }
  }
  let sum = 0;
  const exps = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) {
    const value = Math.exp(logits[i] - maxLogit);
    exps[i] = value;
    sum += value;
  }
  for (let i = 0; i < exps.length; i += 1) {
    exps[i] /= sum;
  }
  return exps;
};

const predictDigit = async () => {
  try {
    result.textContent = "Predicting...";
    const weights = await loadModel();
    const input = getInputVector();
    const hidden = relu(dense(input, weights.dense1.kernel, weights.dense1.bias, weights.dense1.outSize));
    const logits = dense(hidden, weights.dense2.kernel, weights.dense2.bias, weights.dense2.outSize);
    const probabilities = softmax(logits);

    let bestIndex = 0;
    for (let i = 1; i < probabilities.length; i += 1) {
      if (probabilities[i] > probabilities[bestIndex]) {
        bestIndex = i;
      }
    }
    const confidence = probabilities[bestIndex] * 100;
    result.textContent = `Prediction: ${bestIndex} (${confidence.toFixed(1)}%)`;
  } catch (error) {
    result.textContent = "Error: Model not loaded.";
  }
};

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

canvas.addEventListener("touchstart", (event) => {
  event.preventDefault();
  startDrawing(event);
});
canvas.addEventListener("touchmove", (event) => {
  event.preventDefault();
  draw(event);
});
canvas.addEventListener("touchend", stopDrawing);

clearBtn.addEventListener("click", clearCanvas);
predictBtn.addEventListener("click", predictDigit);

setBackground();
