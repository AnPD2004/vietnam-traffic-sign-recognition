export const API = {
  flow1: "/predict/flow1",
  flow2: "/predict/flow2",
  videoFlow1: "/predict/video/flow1",
  videoFlow2: "/predict/video/flow2",
  health: "/health",
};

async function postForm(endpoint, fieldName, file) {
  const formData = new FormData();
  formData.append(fieldName, file);

  const response = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }

  return data;
}

export async function runPrediction(id, file) {
  const endpoint = id === 1 ? API.flow1 : API.flow2;
  return postForm(endpoint, "image", file);
}

export async function runVideoPrediction(id, file) {
  const endpoint = id === 1 ? API.videoFlow1 : API.videoFlow2;
  return postForm(endpoint, "video", file);
}

async function collectParallelResults(runners) {
  const outcomes = await Promise.allSettled(runners);

  const results = { 1: null, 2: null };
  const errors = { 1: null, 2: null };

  outcomes.forEach((outcome, index) => {
    const id = index + 1;
    if (outcome.status === "fulfilled") {
      results[id] = outcome.value;
    } else {
      errors[id] =
        outcome.reason?.message || "Lỗi khi chạy nhận dạng.";
    }
  });

  return { results, errors };
}

export async function runAllPredictions(file) {
  return collectParallelResults([
    runPrediction(1, file),
    runPrediction(2, file),
  ]);
}

export async function runAllVideoPredictions(file) {
  return collectParallelResults([
    runVideoPrediction(1, file),
    runVideoPrediction(2, file),
  ]);
}

export async function loadModelInfo() {
  const response = await fetch(API.health);
  const data = await response.json();

  if (!response.ok) {
    throw new Error("Health check failed");
  }

  return data;
}
