const canvas = document.getElementById("digit-canvas");
const ctx = canvas.getContext("2d");
const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-btn");
const result = document.getElementById("result");

let drawing = false;

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

const predictDigit = async () => {
  const payload = { image: canvas.toDataURL("image/png") };
  result.textContent = "Predicting...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      result.textContent = `Error: ${data.error || "Unknown"}`;
      return;
    }

    result.textContent = `Prediction: ${data.digit} (${data.confidence}%)`;
  } catch (error) {
    result.textContent = "Error: Failed to reach server.";
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
