export type LanePoint = { x: number; y: number };

export type LaneResult = {
  left: [LanePoint, LanePoint] | null;
  right: [LanePoint, LanePoint] | null;
  leftCurve?: LanePoint[];
  rightCurve?: LanePoint[];
  confidence: number;
  centerOffset: number;
  departure: "left" | "right" | "centered" | "unknown";
};

type Sample = { x: number; y: number; weight: number };
type QuadraticFit = { a: number; b: number; c: number; confidence: number; minY: number; maxY: number };

const PROCESS_WIDTH = 360;
const HORIZON_RATIO = 0.53;
const MIN_ROWS = 9;
const CURVE_POINTS = 18;

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function solve3(matrix: number[][], vector: number[]) {
  const a = matrix.map((row, index) => [...row, vector[index]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];
    if (Math.abs(a[col][col]) < 1e-8) return null;
    const divisor = a[col][col];
    for (let j = col; j < 4; j++) a[col][j] /= divisor;
    for (let row = 0; row < 3; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j < 4; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return [a[0][3], a[1][3], a[2][3]] as const;
}

function fitQuadratic(samples: Sample[], width: number, height: number): QuadraticFit | null {
  if (samples.length < MIN_ROWS) return null;
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
  let tx0 = 0, tx1 = 0, tx2 = 0, strength = 0;
  let minY = height, maxY = 0;

  for (const sample of samples) {
    const y = sample.y / height;
    const x = sample.x / width;
    const y2 = y * y;
    const w = sample.weight;
    s0 += w;
    s1 += y * w;
    s2 += y2 * w;
    s3 += y2 * y * w;
    s4 += y2 * y2 * w;
    tx0 += x * w;
    tx1 += x * y * w;
    tx2 += x * y2 * w;
    strength += w;
    minY = Math.min(minY, sample.y);
    maxY = Math.max(maxY, sample.y);
  }

  const coeff = solve3(
    [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]],
    [tx2, tx1, tx0]
  );
  if (!coeff) return null;

  const [a, b, c] = coeff;
  const density = clamp(samples.length / 36, 0, 1);
  const score = clamp(strength / (samples.length * 1.45), 0, 1);
  const span = clamp((maxY - minY) / Math.max(1, height * (1 - HORIZON_RATIO)), 0, 1);
  return { a, b, c, confidence: density * 0.45 + score * 0.25 + span * 0.30, minY, maxY };
}

function curveX(fit: QuadraticFit, y: number, width: number, height: number) {
  const yn = y / height;
  return clamp((fit.a * yn * yn + fit.b * yn + fit.c) * width, 0, width);
}

function pixelScore(data: Uint8ClampedArray, width: number, x: number, y: number) {
  const index = (y * width + x) * 4;
  const leftIndex = (y * width + Math.max(0, x - 2)) * 4;
  const rightIndex = (y * width + Math.min(width - 1, x + 2)) * 4;

  const r = data[index];
  const g = data[index + 1];
  const b = data[index + 2];
  const leftGray = data[leftIndex] * 0.299 + data[leftIndex + 1] * 0.587 + data[leftIndex + 2] * 0.114;
  const rightGray = data[rightIndex] * 0.299 + data[rightIndex + 1] * 0.587 + data[rightIndex + 2] * 0.114;
  const gray = r * 0.299 + g * 0.587 + b * 0.114;
  const edge = Math.abs(rightGray - leftGray);
  const maxChannel = Math.max(r, g, b);
  const minChannel = Math.min(r, g, b);
  const saturation = maxChannel - minChannel;
  const white = gray > 148 && saturation < 58;
  const yellow = r > 145 && g > 118 && b < 150 && r - b > 34;

  if (!white && !yellow && edge < 42) return 0;
  return edge * 1.4 + (white || yellow ? 52 : 0) + Math.max(0, gray - 120) * 0.24;
}

function sampleSide(data: Uint8ClampedArray, width: number, height: number, side: "left" | "right") {
  const samples: Sample[] = [];
  const horizon = Math.round(height * HORIZON_RATIO);
  const centre = width / 2;
  const sign = side === "left" ? -1 : 1;
  let previousX: number | null = null;
  let misses = 0;

  for (let y = height - 5; y >= horizon; y -= 3) {
    const t = (y - horizon) / Math.max(1, height - horizon);
    const perspectiveExpected = centre + sign * width * (0.065 + 0.33 * Math.pow(t, 1.08));
    const target = previousX ?? perspectiveExpected;
    const searchRadius = width * (previousX === null ? 0.115 : 0.07 + misses * 0.014);
    const minX = Math.round(clamp(target - searchRadius, 3, width - 4));
    const maxX = Math.round(clamp(target + searchRadius, 3, width - 4));

    let bestX = -1;
    let bestScore = 0;
    for (let x = minX; x <= maxX; x += 2) {
      if (side === "left" && x > centre + width * 0.04) continue;
      if (side === "right" && x < centre - width * 0.04) continue;
      const raw = pixelScore(data, width, x, y);
      if (!raw) continue;
      const continuityPenalty = previousX === null ? 0 : Math.abs(x - previousX) * 1.6;
      const geometryPenalty = Math.abs(x - perspectiveExpected) * 0.24;
      const score = raw - continuityPenalty - geometryPenalty;
      if (score > bestScore) {
        bestScore = score;
        bestX = x;
      }
    }

    if (bestX >= 0 && bestScore > 48) {
      previousX = bestX;
      misses = 0;
      samples.push({ x: bestX, y, weight: clamp(bestScore / 105, 0.5, 2.25) });
    } else {
      misses++;
      if (misses > 5) previousX = null;
    }
  }
  return samples;
}

function plausiblePair(left: QuadraticFit | null, right: QuadraticFit | null, width: number, height: number) {
  if (!left || !right) return true;
  const nearY = height * 0.95;
  const midY = height * 0.72;
  const nearWidth = curveX(right, nearY, width, height) - curveX(left, nearY, width, height);
  const midWidth = curveX(right, midY, width, height) - curveX(left, midY, width, height);
  return nearWidth > width * 0.18 && nearWidth < width * 0.88 && midWidth > width * 0.07 && midWidth < nearWidth * 1.03;
}

export function createLaneDetector() {
  let workCanvas: HTMLCanvasElement | null = null;
  let workCtx: CanvasRenderingContext2D | null = null;

  return (video: HTMLVideoElement): LaneResult => {
    if (!video.videoWidth || !video.videoHeight || video.readyState < 2) {
      return { left: null, right: null, confidence: 0, centerOffset: 0, departure: "unknown" };
    }

    if (!workCanvas) {
      workCanvas = document.createElement("canvas");
      workCtx = workCanvas.getContext("2d", { willReadFrequently: true });
    }
    if (!workCtx || !workCanvas) {
      return { left: null, right: null, confidence: 0, centerOffset: 0, departure: "unknown" };
    }

    const processHeight = Math.max(180, Math.round(PROCESS_WIDTH * (video.videoHeight / video.videoWidth)));
    if (workCanvas.width !== PROCESS_WIDTH || workCanvas.height !== processHeight) {
      workCanvas.width = PROCESS_WIDTH;
      workCanvas.height = processHeight;
    }

    workCtx.drawImage(video, 0, 0, PROCESS_WIDTH, processHeight);
    const image = workCtx.getImageData(0, 0, PROCESS_WIDTH, processHeight);
    const leftSamples = sampleSide(image.data, PROCESS_WIDTH, processHeight, "left");
    const rightSamples = sampleSide(image.data, PROCESS_WIDTH, processHeight, "right");
    let leftFit = fitQuadratic(leftSamples, PROCESS_WIDTH, processHeight);
    let rightFit = fitQuadratic(rightSamples, PROCESS_WIDTH, processHeight);

    if (!plausiblePair(leftFit, rightFit, PROCESS_WIDTH, processHeight)) {
      if ((leftFit?.confidence ?? 0) >= (rightFit?.confidence ?? 0)) rightFit = null;
      else leftFit = null;
    }

    const scaleX = video.videoWidth / PROCESS_WIDTH;
    const scaleY = video.videoHeight / processHeight;
    const makeCurve = (fit: QuadraticFit | null) => {
      if (!fit) return undefined;
      const farY = clamp(fit.minY, processHeight * HORIZON_RATIO, processHeight * 0.72);
      const nearY = clamp(fit.maxY, processHeight * 0.82, processHeight * 0.975);
      return Array.from({ length: CURVE_POINTS }, (_, i) => {
        const y = farY + ((nearY - farY) * i) / (CURVE_POINTS - 1);
        return { x: curveX(fit, y, PROCESS_WIDTH, processHeight) * scaleX, y: y * scaleY };
      });
    };

    const leftCurve = makeCurve(leftFit);
    const rightCurve = makeCurve(rightFit);
    const leftSegment = leftCurve ? [leftCurve[0], leftCurve[leftCurve.length - 1]] as [LanePoint, LanePoint] : null;
    const rightSegment = rightCurve ? [rightCurve[0], rightCurve[rightCurve.length - 1]] as [LanePoint, LanePoint] : null;
    const confidence = leftFit && rightFit
      ? (leftFit.confidence + rightFit.confidence) / 2
      : Math.max(leftFit?.confidence ?? 0, rightFit?.confidence ?? 0) * 0.52;

    let centerOffset = 0;
    let departure: LaneResult["departure"] = "unknown";
    if (leftSegment && rightSegment) {
      const laneCentre = (leftSegment[1].x + rightSegment[1].x) / 2;
      const laneWidth = Math.max(1, rightSegment[1].x - leftSegment[1].x);
      centerOffset = clamp((video.videoWidth / 2 - laneCentre) / laneWidth, -1, 1);
      departure = centerOffset > 0.14 ? "right" : centerOffset < -0.14 ? "left" : "centered";
    }

    return {
      left: leftSegment,
      right: rightSegment,
      leftCurve,
      rightCurve,
      confidence: clamp(confidence, 0, 1),
      centerOffset,
      departure
    };
  };
}
