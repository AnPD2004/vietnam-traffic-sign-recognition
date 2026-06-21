const API = {
  flow1: "/predict/flow1",
  flow2: "/predict/flow2",
  health: "/health",
};

const state = {
  1: { file: null },
  2: { file: null },
};

function $(selector) {
  return document.querySelector(selector);
}

function $all(selector) {
  return document.querySelectorAll(selector);
}

function setupPipeline(id) {
  const input = $(`[data-input="${id}"]`);
  const uploadZone = $(`[data-upload="${id}"]`);
  const placeholder = $(`[data-placeholder="${id}"]`);
  const preview = $(`[data-preview="${id}"]`);
  const runBtn = $(`[data-run="${id}"]`);

  function setFile(file) {
    if (!file || !file.type.startsWith("image/")) return;

    state[id].file = file;
    runBtn.disabled = false;

    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
    placeholder.hidden = true;
  }

  input.addEventListener("change", (e) => {
    const file = e.target.files[0];
    setFile(file);
  });

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    setFile(file);
  });

  runBtn.addEventListener("click", () => runPrediction(id));
}

function showLoading(id, visible) {
  $(`[data-loading="${id}"]`).hidden = !visible;
  $(`[data-run="${id}"]`).disabled = visible || !state[id].file;
}

function showError(id, message) {
  const el = $(`[data-error="${id}"]`);
  if (message) {
    el.textContent = message;
    el.hidden = false;
  } else {
    el.hidden = true;
    el.textContent = "";
  }
}

function renderDetections(id, detections) {
  const list = $(`[data-detections="${id}"]`);
  list.innerHTML = "";

  if (!detections.length) {
    const li = document.createElement("li");
    li.textContent = "Không phát hiện biển báo nào.";
    list.appendChild(li);
    return;
  }

  detections.forEach((det, index) => {
    const li = document.createElement("li");
    const conf = (det.confidence * 100).toFixed(1);
    li.textContent = `${index + 1}. ${det.class_name} — ${conf}%`;
    list.appendChild(li);
  });
}

function renderResult(id, data) {
  const metrics = data.metrics || {};

  $(`[data-sign-count="${id}"]`).textContent = metrics.sign_count ?? 0;
  $(`[data-inference-ms="${id}"]`).textContent =
    metrics.inference_ms != null ? `${metrics.inference_ms} ms` : "—";
  $(`[data-fps="${id}"]`).textContent =
    metrics.fps != null ? metrics.fps : "—";

  $(`[data-metrics="${id}"]`).hidden = false;

  if (data.image_base64) {
    const img = $(`[data-result-img="${id}"]`);
    img.src = `data:image/jpeg;base64,${data.image_base64}`;
    $(`[data-result="${id}"]`).hidden = false;
  }

  renderDetections(id, data.detections || []);
}

async function runPrediction(id) {
  const file = state[id].file;
  if (!file) return;

  const endpoint = id === 1 ? API.flow1 : API.flow2;
  showError(id, null);
  showLoading(id, true);

  const formData = new FormData();
  formData.append("image", file);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }

    renderResult(id, data);
  } catch (err) {
    showError(id, err.message || "Lỗi khi chạy nhận dạng.");
  } finally {
    showLoading(id, false);
  }
}

async function loadModelInfo() {
  const el = $("#model-info");
  try {
    const response = await fetch(API.health);
    const data = await response.json();

    if (!response.ok) throw new Error("Health check failed");

    const p1 = data.models?.pipeline1;
    const p2 = data.models?.pipeline2;

    el.textContent =
      `P1 YOLO: ${p1?.yolo_run_id || "—"} | ` +
      `P2 YOLO: ${p2?.yolo_run_id || "—"} | ` +
      `P2 CNN: ${p2?.cnn_run_id || "—"}`;
  } catch {
    el.textContent = "Không thể tải thông tin model. Hãy chạy server Flask.";
  }
}

setupPipeline(1);
setupPipeline(2);
loadModelInfo();
