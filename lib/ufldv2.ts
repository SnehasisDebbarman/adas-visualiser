import type { LanePoint, LaneResult } from "@/lib/laneDetection";

type OrtModule = typeof import("onnxruntime-web");
type Session = import("onnxruntime-web").InferenceSession;
type Tensor = import("onnxruntime-web").Tensor;

const INPUT = 640;
const MODEL_URL = "https://raw.githubusercontent.com/hustvl/YOLOP/main/weights/yolop-640-640.onnx";
const MODEL_CACHE = "adas-models-yolop-v2";
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const CURVE_POINTS = 18;

function clamp(value: number, min: number, max: number) { return Math.max(min, Math.min(max, value)); }
function isIOSLike() { return typeof navigator !== "undefined" && /iPad|iPhone|iPod/.test(navigator.userAgent); }

function solve3(m: number[][], v: number[]) {
  const a = m.map((row, i) => [...row, v[i]]);
  for (let i = 0; i < 3; i++) {
    let pivot = i;
    for (let r = i + 1; r < 3; r++) if (Math.abs(a[r][i]) > Math.abs(a[pivot][i])) pivot = r;
    [a[i], a[pivot]] = [a[pivot], a[i]];
    if (Math.abs(a[i][i]) < 1e-9) return null;
    const div = a[i][i];
    for (let c = i; c < 4; c++) a[i][c] /= div;
    for (let r = 0; r < 3; r++) {
      if (r === i) continue;
      const factor = a[r][i];
      for (let c = i; c < 4; c++) a[r][c] -= factor * a[i][c];
    }
  }
  return [a[0][3], a[1][3], a[2][3]] as const;
}

function fitCurve(points: LanePoint[], width: number, height: number) {
  if (points.length < 7) return null;
  const sorted = [...points].sort((a, b) => a.y - b.y);
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, tx0 = 0, tx1 = 0, tx2 = 0;
  for (const point of sorted) {
    const y = point.y / height, x = point.x / width, y2 = y * y;
    s0 += 1; s1 += y; s2 += y2; s3 += y2 * y; s4 += y2 * y2;
    tx0 += x; tx1 += x * y; tx2 += x * y2;
  }
  const coeff = solve3([[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]], [tx2, tx1, tx0]);
  if (!coeff) return null;
  const [a, b, c] = coeff;
  const farY = clamp(sorted[0].y, height * 0.40, height * 0.72);
  const nearY = clamp(sorted[sorted.length - 1].y, height * 0.78, height * 0.985);
  const curve = Array.from({ length: CURVE_POINTS }, (_, i) => {
    const y = farY + ((nearY - farY) * i) / (CURVE_POINTS - 1);
    const yn = y / height;
    return { x: clamp((a * yn * yn + b * yn + c) * width, 0, width), y };
  });
  return { curve, segment: [curve[0], curve[curve.length - 1]] as [LanePoint, LanePoint] };
}

type Preprocessed = { input: Float32Array; dx: number; dy: number; contentWidth: number; contentHeight: number };
function createPreprocessor() {
  const canvas = document.createElement("canvas"); canvas.width = INPUT; canvas.height = INPUT;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Could not create YOLOP preprocessing canvas");
  return (video: HTMLVideoElement): Preprocessed => {
    const vw = video.videoWidth, vh = video.videoHeight;
    const scale = Math.min(INPUT / vw, INPUT / vh);
    const contentWidth = Math.round(vw * scale), contentHeight = Math.round(vh * scale);
    const dx = Math.floor((INPUT - contentWidth) / 2), dy = Math.floor((INPUT - contentHeight) / 2);
    ctx.fillStyle = "rgb(114,114,114)"; ctx.fillRect(0, 0, INPUT, INPUT); ctx.drawImage(video, dx, dy, contentWidth, contentHeight);
    const image = ctx.getImageData(0, 0, INPUT, INPUT).data;
    const plane = INPUT * INPUT, input = new Float32Array(plane * 3);
    for (let i = 0, p = 0; i < image.length; i += 4, p++) {
      input[p] = (image[i] / 255 - MEAN[0]) / STD[0];
      input[plane + p] = (image[i + 1] / 255 - MEAN[1]) / STD[1];
      input[plane * 2 + p] = (image[i + 2] / 255 - MEAN[2]) / STD[2];
    }
    return { input, dx, dy, contentWidth, contentHeight };
  };
}

function clusterCenters(mask: Float32Array, y: number) {
  const centers: number[] = []; let start = -1; const plane = INPUT * INPUT;
  for (let x = 0; x < INPUT; x++) {
    const idx = y * INPUT + x, foreground = mask[plane + idx] > mask[idx];
    if (foreground && start < 0) start = x;
    if ((!foreground || x === INPUT - 1) && start >= 0) {
      const end = foreground && x === INPUT - 1 ? x : x - 1;
      if (end - start >= 1) centers.push((start + end) / 2);
      start = -1;
    }
  }
  return centers;
}

function trackBoundary(mask: Float32Array, prep: Preprocessed, side: "left" | "right", width: number, height: number) {
  const result: LanePoint[] = [];
  const center = prep.dx + prep.contentWidth / 2;
  const yBottom = prep.dy + Math.round(prep.contentHeight * 0.985);
  const yTop = prep.dy + Math.round(prep.contentHeight * 0.40);
  const steps = 34;
  let previousX: number | null = null, missed = 0;
  const initialExpected = center + (side === "left" ? -1 : 1) * prep.contentWidth * 0.20;

  for (let i = 0; i < steps; i++) {
    const y = Math.round(yBottom - ((yBottom - yTop) * i) / (steps - 1));
    const centers = clusterCenters(mask, clamp(y, 0, INPUT - 1)).filter((x) =>
      side === "left" ? x < center + prep.contentWidth * 0.02 : x > center - prep.contentWidth * 0.02
    );
    if (!centers.length) { missed++; if (missed > 5) previousX = null; continue; }

    const progress = i / (steps - 1);
    const expectedTowardHorizon = center + (side === "left" ? -1 : 1) * prep.contentWidth * (0.20 - progress * 0.13);
    const target = previousX ?? initialExpected;
    let best: number | null = null, bestCost = Infinity;
    for (const x of centers) {
      const continuity = Math.abs(x - target);
      const geometry = Math.abs(x - expectedTowardHorizon) * 0.35;
      const maxJump = previousX === null ? prep.contentWidth * 0.24 : prep.contentWidth * (0.085 + missed * 0.025);
      const cost = continuity + geometry;
      if (continuity <= maxJump && cost < bestCost) { best = x; bestCost = cost; }
    }
    if (best === null) { missed++; continue; }
    previousX = best; missed = 0;
    const sourceX = ((best - prep.dx) / prep.contentWidth) * width;
    const sourceY = ((y - prep.dy) / prep.contentHeight) * height;
    if (sourceX >= 0 && sourceX <= width && sourceY >= 0 && sourceY <= height) result.push({ x: sourceX, y: sourceY });
  }
  return result;
}

function decodeLaneMask(tensor: Tensor, prep: Preprocessed, width: number, height: number): LaneResult {
  const data = tensor.data as Float32Array;
  const leftPoints = trackBoundary(data, prep, "left", width, height);
  const rightPoints = trackBoundary(data, prep, "right", width, height);
  let leftFit = fitCurve(leftPoints, width, height), rightFit = fitCurve(rightPoints, width, height);

  if (leftFit && rightFit) {
    const nearWidth = rightFit.segment[1].x - leftFit.segment[1].x;
    const farWidth = rightFit.segment[0].x - leftFit.segment[0].x;
    const plausible = nearWidth > width * 0.16 && nearWidth < width * 0.85 && farWidth > width * 0.025 && farWidth < nearWidth * 1.08;
    if (!plausible) {
      if (leftPoints.length >= rightPoints.length) rightFit = null;
      else leftFit = null;
    }
  }

  const lc = clamp(leftPoints.length / 26, 0, 1), rc = clamp(rightPoints.length / 26, 0, 1);
  const confidence = leftFit && rightFit ? (lc + rc) / 2 : Math.max(lc, rc) * 0.45;
  let centerOffset = 0; let departure: LaneResult["departure"] = "unknown";
  if (leftFit && rightFit) {
    const laneCenter = (leftFit.segment[1].x + rightFit.segment[1].x) / 2;
    const laneWidth = Math.max(1, rightFit.segment[1].x - leftFit.segment[1].x);
    centerOffset = clamp((width / 2 - laneCenter) / laneWidth, -1, 1);
    departure = centerOffset > 0.14 ? "right" : centerOffset < -0.14 ? "left" : "centered";
  }
  return {
    left: leftFit?.segment ?? null,
    right: rightFit?.segment ?? null,
    leftCurve: leftFit?.curve,
    rightCurve: rightFit?.curve,
    confidence,
    centerOffset,
    departure
  };
}

async function loadModelBytes() {
  const request = new Request(MODEL_URL, { cache: "force-cache", mode: "cors" });
  if (typeof caches !== "undefined") {
    const cache = await caches.open(MODEL_CACHE), cached = await cache.match(request);
    if (cached) return new Uint8Array(await cached.arrayBuffer());
    const response = await fetch(request); if (!response.ok) throw new Error(`YOLOP model download failed: ${response.status}`);
    try { await cache.put(request, response.clone()); } catch (error) { console.warn("Could not persist YOLOP model cache", error); }
    return new Uint8Array(await response.arrayBuffer());
  }
  const response = await fetch(request); if (!response.ok) throw new Error(`YOLOP model download failed: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

export type Ufldv2Detector = { backend: "webgpu" | "wasm"; precision: "fp32"; detect(video: HTMLVideoElement): Promise<LaneResult>; dispose(): Promise<void> };
export async function createUfldv2Detector(): Promise<Ufldv2Detector> {
  const ort: OrtModule = await import("onnxruntime-web");
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
  ort.env.wasm.numThreads = 1;

  const ios = isIOSLike();
  const hasWebGpu = !ios && typeof navigator !== "undefined" && Boolean((navigator as Navigator & { gpu?: unknown }).gpu);
  let backend: "webgpu" | "wasm" = hasWebGpu ? "webgpu" : "wasm";
  let session: Session;

  if (ios) {
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    });
  } else {
    const modelBytes = await loadModelBytes();
    try {
      session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: hasWebGpu ? ["webgpu", "wasm"] : ["wasm"],
        graphOptimizationLevel: "all"
      });
    } catch (error) {
      if (!hasWebGpu) throw error;
      console.warn("YOLOP WebGPU init failed; using WASM", error);
      backend = "wasm";
      session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
      });
    }
  }

  const preprocess = createPreprocessor();
  const laneOutputName = session.outputNames.find((name) => name.includes("lane_line")) ?? "lane_line_seg";
  return {
    backend, precision: "fp32",
    async detect(video) {
      if (!video.videoWidth || !video.videoHeight || video.readyState < 2) return { left: null, right: null, confidence: 0, centerOffset: 0, departure: "unknown" };
      const prep = preprocess(video); const input = new ort.Tensor("float32", prep.input, [1, 3, INPUT, INPUT]);
      const outputs = await session.run({ [session.inputNames[0]]: input }); const laneTensor = outputs[laneOutputName];
      if (!laneTensor) throw new Error("YOLOP lane segmentation output missing");
      return decodeLaneMask(laneTensor, prep, video.videoWidth, video.videoHeight);
    },
    async dispose() { await session.release(); }
  };
}
