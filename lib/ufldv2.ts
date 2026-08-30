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

function fitCurve(points: LanePoint[], width: number, height: number): [LanePoint, LanePoint] | null {
  if (points.length < 14) return null;
  const sorted = [...points].sort((a, b) => a.y - b.y);
  const start = Math.floor(sorted.length * 0.06);
  const end = Math.ceil(sorted.length * 0.96);
  const trimmed = sorted.slice(start, end);
  if (trimmed.length < 12) return null;

  // Quadratic fit in normalized image coordinates: x = a*y^2 + b*y + c.
  // Normalizing keeps the matrix stable and preserves bends that a straight fit loses.
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  let tx0 = 0, tx1 = 0, tx2 = 0;
  for (const point of trimmed) {
    const y = point.y / height;
    const x = point.x / width;
    const y2 = y * y;
    s0 += 1; s1 += y; s2 += y2; s3 += y2 * y; s4 += y2 * y2;
    tx0 += x; tx1 += x * y; tx2 += x * y2;
  }
  const coeff = solve3(
    [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]],
    [tx2, tx1, tx0]
  );
  if (!coeff) return null;
  const [qa, qb, qc] = coeff;
  const xAt = (yPx: number) => clamp((qa * Math.pow(yPx / height, 2) + qb * (yPx / height) + qc) * width, 0, width);

  const farY = clamp(trimmed[0].y, height * 0.42, height * 0.72);
  const nearY = clamp(trimmed[trimmed.length - 1].y, height * 0.80, height * 0.985);
  return [
    { x: xAt(farY), y: farY },
    { x: xAt(nearY), y: nearY }
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
  const left = fitCurve(leftDecoded.points, width, height);
  const right = fitCurve(rightDecoded.points, width, height);
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
