import type { LanePoint, LaneResult } from "@/lib/laneDetection";

type OrtModule = typeof import("onnxruntime-web");
type Session = import("onnxruntime-web").InferenceSession;
type Tensor = import("onnxruntime-web").Tensor;

const INPUT_WIDTH = 1600;
const INPUT_HEIGHT = 320;
const RESIZED_HEIGHT = Math.round(INPUT_HEIGHT / 0.6);
const NUM_GRID_ROW = 200;
const NUM_ROW = 72;
const NUM_LANES = 4;
const ROW_ANCHORS = Array.from({ length: NUM_ROW }, (_, i) => 0.42 + (i * (1 - 0.42)) / (NUM_ROW - 1));
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function index4(a: number, b: number, c: number, d: number, B: number, C: number, D: number) {
  return ((a * B + b) * C + c) * D + d;
}

function argmaxGrid(data: Float32Array, grid: number, row: number, lane: number) {
  let best = 0;
  let bestValue = -Infinity;
  for (let g = 0; g < grid; g++) {
    const value = data[index4(0, g, row, lane, grid, NUM_ROW, NUM_LANES)];
    if (value > bestValue) { bestValue = value; best = g; }
  }
  return best;
}

function existsAt(data: Float32Array, row: number, lane: number) {
  const no = data[index4(0, 0, row, lane, 2, NUM_ROW, NUM_LANES)];
  const yes = data[index4(0, 1, row, lane, 2, NUM_ROW, NUM_LANES)];
  return yes > no;
}

function localExpectation(data: Float32Array, maxIndex: number, row: number, lane: number) {
  const from = Math.max(0, maxIndex - 1);
  const to = Math.min(NUM_GRID_ROW - 1, maxIndex + 1);
  let maxLogit = -Infinity;
  for (let g = from; g <= to; g++) maxLogit = Math.max(maxLogit, data[index4(0, g, row, lane, NUM_GRID_ROW, NUM_ROW, NUM_LANES)]);
  let sum = 0;
  let weighted = 0;
  for (let g = from; g <= to; g++) {
    const probability = Math.exp(data[index4(0, g, row, lane, NUM_GRID_ROW, NUM_ROW, NUM_LANES)] - maxLogit);
    sum += probability;
    weighted += probability * g;
  }
  return sum > 0 ? weighted / sum + 0.5 : maxIndex + 0.5;
}

function fitSegment(points: LanePoint[], width: number, height: number): [LanePoint, LanePoint] | null {
  if (points.length < 12) return null;
  const sorted = [...points].sort((a, b) => a.y - b.y);
  const trimmed = sorted.slice(Math.floor(sorted.length * 0.08), Math.ceil(sorted.length * 0.94));
  let sy = 0, sx = 0, syy = 0, syx = 0;
  for (const point of trimmed) {
    sy += point.y;
    sx += point.x;
    syy += point.y * point.y;
    syx += point.y * point.x;
  }
  const n = trimmed.length;
  const denominator = n * syy - sy * sy;
  if (Math.abs(denominator) < 1e-6) return null;
  const a = (n * syx - sy * sx) / denominator;
  const b = (sx - a * sy) / n;
  const farY = clamp(trimmed[0].y, height * 0.38, height * 0.75);
  const nearY = clamp(trimmed[trimmed.length - 1].y, height * 0.72, height * 0.99);
  return [
    { x: clamp(a * farY + b, 0, width), y: farY },
    { x: clamp(a * nearY + b, 0, width), y: nearY }
  ];
}

function decodeLane(locRow: Float32Array, existRow: Float32Array, laneIndex: number, width: number, height: number) {
  const points: LanePoint[] = [];
  let valid = 0;
  for (let row = 0; row < NUM_ROW; row++) {
    if (!existsAt(existRow, row, laneIndex)) continue;
    valid++;
    const maxGrid = argmaxGrid(locRow, NUM_GRID_ROW, row, laneIndex);
    const gridPosition = localExpectation(locRow, maxGrid, row, laneIndex);
    const x = (gridPosition / (NUM_GRID_ROW - 1)) * width;
    const y = ROW_ANCHORS[row] * height;
    points.push({ x, y });
  }
  return { points, confidence: clamp(valid / NUM_ROW, 0, 1) };
}

function makeResult(locRowTensor: Tensor, existRowTensor: Tensor, width: number, height: number): LaneResult {
  const locRow = locRowTensor.data as Float32Array;
  const existRow = existRowTensor.data as Float32Array;
  const leftDecoded = decodeLane(locRow, existRow, 1, width, height);
  const rightDecoded = decodeLane(locRow, existRow, 2, width, height);
  const left = fitSegment(leftDecoded.points, width, height);
  const right = fitSegment(rightDecoded.points, width, height);
  const confidence = left && right
    ? (leftDecoded.confidence + rightDecoded.confidence) / 2
    : Math.max(leftDecoded.confidence, rightDecoded.confidence) * 0.55;

  let centerOffset = 0;
  let departure: LaneResult["departure"] = "unknown";
  if (left && right) {
    const laneCenter = (left[1].x + right[1].x) / 2;
    const laneWidth = Math.max(1, right[1].x - left[1].x);
    centerOffset = clamp((width / 2 - laneCenter) / laneWidth, -1, 1);
    departure = centerOffset > 0.14 ? "right" : centerOffset < -0.14 ? "left" : "centered";
  }

  return { left, right, confidence, centerOffset, departure };
}

function createPreprocessor() {
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_WIDTH;
  canvas.height = RESIZED_HEIGHT;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Could not create UFLDv2 preprocessing canvas");

  return (video: HTMLVideoElement) => {
    ctx.drawImage(video, 0, 0, INPUT_WIDTH, RESIZED_HEIGHT);
    const image = ctx.getImageData(0, RESIZED_HEIGHT - INPUT_HEIGHT, INPUT_WIDTH, INPUT_HEIGHT).data;
    const plane = INPUT_WIDTH * INPUT_HEIGHT;
    const input = new Float32Array(plane * 3);
    for (let i = 0, p = 0; i < image.length; i += 4, p++) {
      input[p] = (image[i] / 255 - MEAN[0]) / STD[0];
      input[plane + p] = (image[i + 1] / 255 - MEAN[1]) / STD[1];
      input[plane * 2 + p] = (image[i + 2] / 255 - MEAN[2]) / STD[2];
    }
    return input;
  };
}

export type Ufldv2Detector = {
  backend: "webgpu" | "wasm";
  detect(video: HTMLVideoElement): Promise<LaneResult>;
  dispose(): Promise<void>;
};

export async function createUfldv2Detector(): Promise<Ufldv2Detector> {
  const ort: OrtModule = await import("onnxruntime-web");
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
  ort.env.wasm.numThreads = 1;

  const hasWebGpu = typeof navigator !== "undefined" && Boolean((navigator as Navigator & { gpu?: unknown }).gpu);
  let backend: "webgpu" | "wasm" = hasWebGpu ? "webgpu" : "wasm";
  let session: Session;
  try {
    session = await ort.InferenceSession.create("/models/ufldv2.onnx", {
      executionProviders: hasWebGpu ? ["webgpu", "wasm"] : ["wasm"],
      graphOptimizationLevel: "all"
    });
  } catch (error) {
    if (!hasWebGpu) throw error;
    backend = "wasm";
    session = await ort.InferenceSession.create("/models/ufldv2.onnx", {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    });
  }

  const preprocess = createPreprocessor();
  return {
    backend,
    async detect(video) {
      if (!video.videoWidth || !video.videoHeight || video.readyState < 2) {
        return { left: null, right: null, confidence: 0, centerOffset: 0, departure: "unknown" };
      }
      const input = preprocess(video);
      const tensor = new ort.Tensor("float32", input, [1, 3, INPUT_HEIGHT, INPUT_WIDTH]);
      const result = await session.run({ [session.inputNames[0]]: tensor });
      const locRow = result.loc_row ?? result[session.outputNames.find((name) => name.includes("loc_row")) ?? ""];
      const existRow = result.exist_row ?? result[session.outputNames.find((name) => name.includes("exist_row")) ?? ""];
      if (!locRow || !existRow) throw new Error("Unexpected UFLDv2 output tensors");
      return makeResult(locRow, existRow, video.videoWidth, video.videoHeight);
    },
    async dispose() {
      await session.release();
    }
  };
}
