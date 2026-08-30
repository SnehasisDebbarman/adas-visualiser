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
type FittedLine = { a: number; b: number; confidence: number };

const PROCESS_WIDTH = 360;
const HORIZON_RATIO = 0.56;
const MIN_ROWS = 8;

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function fitLine(samples: Sample[]): FittedLine | null {
  if (samples.length < MIN_ROWS) return null;
  let sw = 0, sy = 0, sx = 0, syy = 0, syx = 0, score = 0;
  for (const sample of samples) {
    const w = sample.weight;
    sw += w; sy += sample.y * w; sx += sample.x * w;
    syy += sample.y * sample.y * w; syx += sample.y * sample.x * w; score += w;
  }
  const denominator = sw * syy - sy * sy;
  if (Math.abs(denominator) < 1e-6) return null;
  const a = (sw * syx - sy * sx) / denominator;
  const b = (sx - a * sy) / sw;
  const density = clamp(samples.length / 34, 0, 1);
  const strength = clamp(score / (samples.length * 150), 0, 1);
  return { a, b, confidence: density * 0.65 + strength * 0.35 };
}

function pixelScore(data: Uint8ClampedArray, width: number, x: number, y: number) {
  const index = (y * width + x) * 4;
  const leftIndex = (y * width + Math.max(0, x - 2)) * 4;
  const rightIndex = (y * width + Math.min(width - 1, x + 2)) * 4;
  const r = data[index], g = data[index + 1], b = data[index + 2];
  const leftGray = data[leftIndex] * 0.299 + data[leftIndex + 1] * 0.587 + data[leftIndex + 2] * 0.114;
  const rightGray = data[rightIndex] * 0.299 + data[rightIndex + 1] * 0.587 + data[rightIndex + 2] * 0.114;
  const gray = r * 0.299 + g * 0.587 + b * 0.114;
  const edge = Math.abs(rightGray - leftGray);
  const maxChannel = Math.max(r, g, b), minChannel = Math.min(r, g, b), saturation = maxChannel - minChannel;
  const white = gray > 145 && saturation < 62;
  const yellow = r > 145 && g > 120 && b < 155 && r - b > 35;
  if (!white && !yellow && edge < 38) return 0;
  return edge * 1.35 + (white || yellow ? 48 : 0) + Math.max(0, gray - 120) * 0.25;
}

function sampleSide(data: Uint8ClampedArray, width: number, height: number, side: "left" | "right") {
  const samples: Sample[] = [];
  const horizon = Math.round(height * HORIZON_RATIO), centre = width / 2, farHalf = width * 0.07, nearHalf = width * 0.38;
  const sign = side === "left" ? -1 : 1;
  for (let y = height - 5; y >= horizon; y -= 4) {
    const t = (y - horizon) / Math.max(1, height - horizon);
    const expected = centre + sign * (farHalf + (nearHalf - farHalf) * t);
    const searchRadius = width * (0.10 + 0.05 * t);
    const minX = Math.round(clamp(expected - searchRadius, 3, width - 4));
    const maxX = Math.round(clamp(expected + searchRadius, 3, width - 4));
    let bestX = -1, bestScore = 0;
    for (let x = minX; x <= maxX; x += 2) {
      const score = pixelScore(data, width, x, y);
      if (score > bestScore) { bestScore = score; bestX = x; }
    }
    if (bestX >= 0 && bestScore > 58) samples.push({ x: bestX, y, weight: clamp(bestScore / 90, 0.5, 2.5) });
  }
  return samples;
}

function validSlope(line: FittedLine | null, side: "left" | "right") {
  if (!line) return null;
  if (side === "left" && !(line.a < -0.12 && line.a > -2.8)) return null;
  if (side === "right" && !(line.a > 0.12 && line.a < 2.8)) return null;
  return line;
}

export function createLaneDetector() {
  let workCanvas: HTMLCanvasElement | null = null;
  let workCtx: CanvasRenderingContext2D | null = null;
  return (video: HTMLVideoElement): LaneResult => {
    if (!video.videoWidth || !video.videoHeight || video.readyState < 2) return { left: null, right: null, confidence: 0, centerOffset: 0, departure: "unknown" };
    if (!workCanvas) { workCanvas = document.createElement("canvas"); workCtx = workCanvas.getContext("2d", { willReadFrequently: true }); }
    if (!workCtx || !workCanvas) return { left: null, right: null, confidence: 0, centerOffset: 0, departure: "unknown" };
    const processHeight = Math.max(180, Math.round(PROCESS_WIDTH * (video.videoHeight / video.videoWidth)));
    if (workCanvas.width !== PROCESS_WIDTH || workCanvas.height !== processHeight) { workCanvas.width = PROCESS_WIDTH; workCanvas.height = processHeight; }
    workCtx.drawImage(video, 0, 0, PROCESS_WIDTH, processHeight);
    const image = workCtx.getImageData(0, 0, PROCESS_WIDTH, processHeight);
    const left = validSlope(fitLine(sampleSide(image.data, PROCESS_WIDTH, processHeight, "left")), "left");
    const right = validSlope(fitLine(sampleSide(image.data, PROCESS_WIDTH, processHeight, "right")), "right");
    const horizonY = processHeight * HORIZON_RATIO, bottomY = processHeight * 0.96;
    const scaleX = video.videoWidth / PROCESS_WIDTH, scaleY = video.videoHeight / processHeight;
    const toSegment = (line: FittedLine | null): [LanePoint, LanePoint] | null => line ? [
      { x: clamp(line.a * horizonY + line.b, 0, PROCESS_WIDTH) * scaleX, y: horizonY * scaleY },
      { x: clamp(line.a * bottomY + line.b, 0, PROCESS_WIDTH) * scaleX, y: bottomY * scaleY }
    ] : null;
    const leftSegment = toSegment(left), rightSegment = toSegment(right);
    const confidence = left && right ? (left.confidence + right.confidence) / 2 : (left?.confidence ?? right?.confidence ?? 0) * 0.55;
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
      leftCurve: leftSegment ? [...leftSegment] : undefined,
      rightCurve: rightSegment ? [...rightSegment] : undefined,
      confidence: clamp(confidence, 0, 1),
      centerOffset,
      departure
    };
  };
}
