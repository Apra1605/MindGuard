import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm";
const MAX_BLENDSHAPES_TO_SHOW = 12;

const demosSection = document.getElementById("demos");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const webcamButton = document.getElementById("webcamButton");
const statusBadge = document.getElementById("statusBadge");
const fpsChip = document.getElementById("fpsChip");
const emotionCallout = document.getElementById("emotionCallout");
const emotionLabel = document.getElementById("emotionLabel");
const emotionConfidence = document.getElementById("emotionConfidence");
const emotionReason = document.getElementById("emotionReason");
const expressionSummary = document.getElementById("expressionSummary");
const signalSummary = document.getElementById("signalSummary");
const expressionTags = document.getElementById("expressionTags");
const videoBlendShapes = document.getElementById("video-blend-shapes");
const blendshapeCount = document.getElementById("blendshapeCount");
const smileMetric = document.getElementById("smileMetric");
const browMetric = document.getElementById("browMetric");
const eyeMetric = document.getElementById("eyeMetric");
const jawMetric = document.getElementById("jawMetric");

const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

let faceLandmarker;
let webcamRunning = false;
let mediaStream = null;
let lastVideoTime = -1;
let animationFrameId = null;
let lastAnimationTimestamp = 0;
let latestResults = null;

function updateStatus(type, text) {
  statusBadge.textContent = text;
  statusBadge.className = `status-badge ${type}`;
}

function formatPercent(value) {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function formatBlendshapeName(name) {
  return name
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (char) => char.toUpperCase());
}

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function averageScore(scores, keys) {
  if (!keys.length) {
    return 0;
  }

  const total = keys.reduce((sum, key) => sum + (scores[key] || 0), 0);
  return total / keys.length;
}

function toScoreMap(faceBlendshapes) {
  const categories = faceBlendshapes?.[0]?.categories || [];
  return categories.reduce((accumulator, category) => {
    accumulator[category.categoryName] = category.score;
    return accumulator;
  }, {});
}

function renderTags(tags) {
  expressionTags.innerHTML = tags.length
    ? tags.map((tag) => `<span class="tag">${tag}</span>`).join("")
    : '<span class="tag placeholder">No strong movement cues yet</span>';
}

function getTopBlendshape(faceBlendshapes) {
  const categories = faceBlendshapes?.[0]?.categories || [];
  if (!categories.length) {
    return null;
  }

  return categories
    .slice()
    .sort((a, b) => b.score - a.score)[0];
}

function setIdleState(message) {
  emotionCallout.dataset.tone = "neutral";
  emotionLabel.textContent = "Awaiting input";
  emotionConfidence.textContent = "0%";
  emotionReason.textContent = message;
  if (expressionSummary) {
    expressionSummary.textContent = "Waiting for webcam...";
  }
  if (signalSummary) {
    signalSummary.textContent = "No face signal yet.";
  }
  renderTags([]);
  smileMetric.textContent = "0%";
  browMetric.textContent = "0%";
  eyeMetric.textContent = "0%";
  jawMetric.textContent = "0%";
  blendshapeCount.textContent = "0 active";
  videoBlendShapes.innerHTML = "";
}

function buildCues(scores) {
  return {
    smile: averageScore(scores, [
      "mouthSmileLeft",
      "mouthSmileRight",
      "mouthDimpleLeft",
      "mouthDimpleRight",
    ]),
    cheekLift: averageScore(scores, ["cheekSquintLeft", "cheekSquintRight"]),
    frown: averageScore(scores, ["mouthFrownLeft", "mouthFrownRight"]),
    browRaise: averageScore(scores, [
      "browInnerUp",
      "browOuterUpLeft",
      "browOuterUpRight",
    ]),
    browDown: averageScore(scores, ["browDownLeft", "browDownRight"]),
    eyeWide: averageScore(scores, ["eyeWideLeft", "eyeWideRight"]),
    eyeSquint: averageScore(scores, ["eyeSquintLeft", "eyeSquintRight"]),
    jawOpen: scores.jawOpen || 0,
    mouthFunnel: scores.mouthFunnel || 0,
    mouthStretch: averageScore(scores, ["mouthStretchLeft", "mouthStretchRight"]),
    mouthPress: averageScore(scores, ["mouthPressLeft", "mouthPressRight"]),
    mouthUpperUp: averageScore(scores, ["mouthUpperUpLeft", "mouthUpperUpRight"]),
    mouthShrug: averageScore(scores, ["mouthShrugLower", "mouthShrugUpper"]),
    mouthPucker: scores.mouthPucker || 0,
    noseSneer: averageScore(scores, ["noseSneerLeft", "noseSneerRight"]),
    blink: averageScore(scores, ["eyeBlinkLeft", "eyeBlinkRight"]),
    jawForward: scores.jawForward || 0,
  };
}

function analyzeEmotion(scores) {
  const cues = buildCues(scores);

  const emotionCandidates = [
    {
      label: "Joy",
      tone: "joy",
      score: clamp(cues.smile * 0.7 + cues.cheekLift * 0.25 - cues.frown * 0.1),
      reason: "Strong smile activity with supporting cheek movement.",
    },
    {
      label: "Surprise",
      tone: "surprise",
      score: clamp(
        cues.jawOpen * 0.3 +
          cues.eyeWide * 0.25 +
          cues.browRaise * 0.25 +
          cues.mouthFunnel * 0.2
      ),
      reason: "Raised brows, widened eyes, and an open mouth are all elevated.",
    },
    {
      label: "Anger",
      tone: "anger",
      score: clamp(
        cues.browDown * 0.3 +
          cues.mouthPress * 0.2 +
          cues.noseSneer * 0.2 +
          cues.frown * 0.15 +
          cues.jawForward * 0.15
      ),
      reason: "Lowered brows and facial tension suggest a more forceful expression.",
    },
    {
      label: "Sadness",
      tone: "sadness",
      score: clamp(cues.frown * 0.45 + cues.browRaise * 0.3 + cues.mouthShrug * 0.2),
      reason: "Frown cues with inner brow lift lean toward a sad expression.",
    },
    {
      label: "Fear",
      tone: "fear",
      score: clamp(
        cues.eyeWide * 0.3 +
          cues.browRaise * 0.2 +
          cues.mouthStretch * 0.25 +
          cues.jawOpen * 0.2
      ),
      reason: "Wide eyes plus stretched mouth activity resemble a fearful reaction.",
    },
    {
      label: "Disgust",
      tone: "disgust",
      score: clamp(
        cues.noseSneer * 0.4 + cues.mouthUpperUp * 0.25 + cues.browDown * 0.2
      ),
      reason: "Nose sneer and upper-lip raise are the strongest cues here.",
    },
  ];

  const strongestEmotion = emotionCandidates.reduce((best, candidate) =>
    candidate.score > best.score ? candidate : best
  );

  const expressionEnergy = clamp(
    Math.max(
      strongestEmotion.score,
      cues.smile,
      cues.browRaise,
      cues.browDown,
      cues.eyeWide,
      cues.jawOpen
    )
  );

  const tags = [];

  if (cues.smile > 0.28) tags.push("Smiling");
  if (cues.frown > 0.22) tags.push("Frowning");
  if (cues.browRaise > 0.24) tags.push("Eyebrows raised");
  if (cues.browDown > 0.24) tags.push("Brows lowered");
  if (cues.eyeWide > 0.22) tags.push("Eyes widened");
  if (cues.eyeSquint > 0.24) tags.push("Eyes squinting");
  if (cues.jawOpen > 0.2) tags.push("Mouth open");
  if (cues.mouthPucker > 0.2) tags.push("Lips pursed");
  if (cues.noseSneer > 0.18) tags.push("Nose sneer");
  if (cues.blink > 0.58) tags.push("Blink detected");

  if (expressionEnergy < 0.18) {
    return {
      label: "Neutral / Resting",
      tone: "neutral",
      confidence: 0.62,
      reason: "The face appears relatively relaxed, with no single movement cue dominating.",
      tags: tags.length ? tags : ["Resting expression"],
      cues,
    };
  }

  return {
    label: strongestEmotion.label,
    tone: strongestEmotion.tone,
    confidence: clamp(strongestEmotion.score),
    reason: strongestEmotion.reason,
    tags: tags.length ? tags : ["Subtle expression"],
    cues,
  };
}

function drawBlendShapes(faceBlendshapes) {
  const categories = (faceBlendshapes?.[0]?.categories || [])
    .slice()
    .sort((a, b) => b.score - a.score)
    .filter((item) => item.score >= 0.01)
    .slice(0, MAX_BLENDSHAPES_TO_SHOW);

  blendshapeCount.textContent = `${categories.length} active`;

  if (!categories.length) {
    videoBlendShapes.innerHTML = "";
    return;
  }

  videoBlendShapes.innerHTML = categories
    .map(
      (shape) => `
        <li class="blend-shape-row">
          <div>
            <div class="blend-shape-name">${formatBlendshapeName(shape.categoryName)}</div>
            <div class="blend-shape-bar">
              <span class="blend-shape-fill" style="width: ${Math.max(
                shape.score * 100,
                2
              )}%"></span>
            </div>
          </div>
          <div class="blend-shape-score">${shape.score.toFixed(3)}</div>
        </li>
      `
    )
    .join("");
}

function renderAnalysis(faceBlendshapes) {
  const scores = toScoreMap(faceBlendshapes);
  const analysis = analyzeEmotion(scores);
  const topBlendshape = getTopBlendshape(faceBlendshapes);

  emotionCallout.dataset.tone = analysis.tone;
  emotionLabel.textContent = analysis.label;
  emotionConfidence.textContent = formatPercent(analysis.confidence);
  emotionReason.textContent = analysis.reason;
  if (expressionSummary) {
    expressionSummary.textContent = analysis.tags.join(", ");
  }
  if (signalSummary) {
    signalSummary.textContent = topBlendshape
      ? `${formatBlendshapeName(topBlendshape.categoryName)} at ${formatPercent(
          topBlendshape.score
        )}`
      : "No dominant signal available.";
  }

  smileMetric.textContent = formatPercent(analysis.cues.smile);
  browMetric.textContent = formatPercent(analysis.cues.browRaise);
  eyeMetric.textContent = formatPercent(analysis.cues.eyeWide);
  jawMetric.textContent = formatPercent(analysis.cues.jawOpen);

  renderTags(analysis.tags);
  drawBlendShapes(faceBlendshapes);
}

function resizeCanvasToVideo() {
  if (!video.videoWidth || !video.videoHeight) {
    return;
  }

  if (
    canvasElement.width !== video.videoWidth ||
    canvasElement.height !== video.videoHeight
  ) {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }
}

function drawLandmarks(results) {
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (!results?.faceLandmarks?.length) {
    return;
  }

  for (const landmarks of results.faceLandmarks) {
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      { color: "#f7f3ee88", lineWidth: 1 }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
      { color: "#ffd9ac", lineWidth: 1.5 }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LIPS,
      { color: "#ff9c5b", lineWidth: 1.8 }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
      { color: "#77e6d7", lineWidth: 1.5 }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
      { color: "#77e6d7", lineWidth: 1.5 }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
      { color: "#fff0c2", lineWidth: 1.5 }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
      { color: "#fff0c2", lineWidth: 1.5 }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
      { color: "#ffffff", lineWidth: 1.2 }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
      { color: "#ffffff", lineWidth: 1.2 }
    );
  }
}

async function createFaceLandmarker() {
  updateStatus("loading", "Loading model");

  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(WASM_PATH);

    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: MODEL_ASSET_PATH,
          delegate: "GPU",
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: 1,
      });
      updateStatus("ready", "Model ready (GPU)");
    } catch (gpuError) {
      console.warn("GPU delegate unavailable, falling back to CPU.", gpuError);
      faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: MODEL_ASSET_PATH,
          delegate: "CPU",
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: 1,
      });
      updateStatus("ready", "Model ready (CPU)");
    }

    webcamButton.disabled = false;
    webcamButton.textContent = "Start Webcam Analysis";
    demosSection.classList.remove("invisible");
  } catch (error) {
    console.error("Failed to create the face landmarker.", error);
    updateStatus("error", "Model failed to load");
    emotionReason.textContent =
      "The MediaPipe model could not be initialized. Check the console for details.";
  }
}

function stopStream() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  webcamRunning = false;
  lastVideoTime = -1;
  lastAnimationTimestamp = 0;
  latestResults = null;
  video.srcObject = null;
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  webcamButton.textContent = "Start Webcam Analysis";
  updateStatus("ready", "Model ready");
  fpsChip.textContent = "Webcam stopped";
  setIdleState("The webcam is off. Start it again to resume live analysis.");
}

async function startWebcam() {
  if (!faceLandmarker) {
    return;
  }

  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
    updateStatus("error", "Webcam unsupported");
    emotionReason.textContent =
      "Your browser does not support getUserMedia, which is required for webcam access.";
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    video.srcObject = mediaStream;

    if (video.readyState < 2) {
      await new Promise((resolve) => {
        video.addEventListener("loadeddata", resolve, { once: true });
      });
    }

    await video.play();

    webcamRunning = true;
    webcamButton.textContent = "Stop Webcam Analysis";
    updateStatus("active", "Webcam running");
    fpsChip.textContent = "Analyzing live video";
    animationFrameId = requestAnimationFrame(predictWebcam);
  } catch (error) {
    console.error("Unable to access webcam.", error);
    updateStatus("error", "Camera permission denied");
    emotionReason.textContent =
      "The webcam could not be started. Check browser permissions and try again.";
  }
}

async function toggleWebcam() {
  if (webcamRunning) {
    stopStream();
    return;
  }

  await startWebcam();
}

function predictWebcam(timestamp) {
  if (!webcamRunning || !faceLandmarker) {
    return;
  }

  resizeCanvasToVideo();

  if (lastAnimationTimestamp) {
    const fps = 1000 / Math.max(1, timestamp - lastAnimationTimestamp);
    fpsChip.textContent = `${fps.toFixed(1)} FPS`;
  }
  lastAnimationTimestamp = timestamp;

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    latestResults = faceLandmarker.detectForVideo(video, performance.now());
  }

  if (latestResults?.faceLandmarks?.length) {
    drawLandmarks(latestResults);
    renderAnalysis(latestResults.faceBlendshapes);
  } else {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    emotionCallout.dataset.tone = "neutral";
    emotionLabel.textContent = "No face found";
    emotionConfidence.textContent = "0%";
    emotionReason.textContent =
      "The camera is live, but no face is clearly visible to the model right now.";
    if (expressionSummary) {
      expressionSummary.textContent = "No readable facial expression yet.";
    }
    if (signalSummary) {
      signalSummary.textContent = "Move closer to the webcam for stronger signals.";
    }
    renderTags(["Move closer to the camera", "Face the lens directly"]);
    smileMetric.textContent = "0%";
    browMetric.textContent = "0%";
    eyeMetric.textContent = "0%";
    jawMetric.textContent = "0%";
    blendshapeCount.textContent = "0 active";
    videoBlendShapes.innerHTML = "";
  }

  animationFrameId = requestAnimationFrame(predictWebcam);
}

webcamButton.addEventListener("click", toggleWebcam);
window.addEventListener("beforeunload", stopStream);

setIdleState("Start the webcam to begin live facial analysis.");
createFaceLandmarker();
