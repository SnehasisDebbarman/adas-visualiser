"use client";

import { ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import type { ObjectDetection } from "@tensorflow-models/coco-ssd";
import { createLaneDetector, type LanePoint, type LaneResult } from "@/lib/laneDetection";

type Prediction = {
  id: number;
  bbox: [number, number, number, number];
  class: string;
  score: number;
  distanceHint: number;
};

type RawPrediction = Omit<Prediction, "id" | "distanceHint">;
type SourceMode = "idle" | "camera" | "video";

const ROAD_CLASSES = new Set([
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "bus",
  "truck",
  "traffic light",
  "stop sign"
]);

const VEHICLE_CLASSES = new Set(["bicycle", "car", "motorcycle", "bus", "truck"]);
const EMPTY_LANE: LaneResult = { left: null, right: null, confidence: 0, centerOffset: 0, departure: "unknown" };

function colourForClass(name: string) {
  if (name === "person") return "#ffcc66";
  if (name === "traffic light" || name === "stop sign") return "#ff6b6b";
  return "#6ee7ff";
}

function centre(bbox: [number, number, number, number]) {
  return [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2] as const;
}

function lerpPoint(previous: LanePoint, next: LanePoint, alpha: number): LanePoint {
  return {
    x: previous.x + (next.x - previous.x) * alpha,
    y: previous.y + (next.y - previous.y) * alpha
  };
}

function smoothSegment(
  previous: [LanePoint, LanePoint] | null,
  next: [LanePoint, LanePoint] | null,
  alpha: number
): [LanePoint, LanePoint] | null {
  if (!next) return previous;
  if (!previous) return next;
  return [lerpPoint(previous[0], next[0], alpha), lerpPoint(previous[1], next[1], alpha)];
}

function smoothLane(previous: LaneResult, next: LaneResult): LaneResult {
  const confidence = next.confidence > 0
    ? previous.confidence + (next.confidence - previous.confidence) * 0.28
    : previous.confidence * 0.82;

  const left = next.left ? smoothSegment(previous.left, next.left, 0.24) : confidence > 0.18 ? previous.left : null;
  const right = next.right ? smoothSegment(previous.right, next.right, 0.24) : confidence > 0.18 ? previous.right : null;

  return {
    left,
    right,
    confidence,
    centerOffset: previous.centerOffset + (next.centerOffset - previous.centerOffset) * 0.2,
    departure: next.departure === "unknown" && confidence > 0.18 ? previous.departure : next.departure
  };
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<ObjectDetection | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastInferenceRef = useRef(0);
  const lastLaneInferenceRef = useRef(0);
  const inferenceBusyRef = useRef(false);
  const fileUrlRef = useRef<string | null>(null);
  const predictionsRef = useRef<Prediction[]>([]);
  const laneResultRef = useRef<LaneResult>(EMPTY_LANE);
  const laneDetectorRef = useRef<ReturnType<typeof createLaneDetector> | null>(null);
  const nextTrackIdRef = useRef(1);

  const [mode, setMode] = useState<SourceMode>("idle");
  const [status, setStatus] = useState("Loading perception model…");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [laneResult, setLaneResult] = useState<LaneResult>(EMPTY_LANE);
  const [fps, setFps] = useState(0);
  const [modelReady, setModelReady] = useState(false);
  const [showLanes, setShowLanes] = useState(true);

  const publishPredictions = useCallback((items: Prediction[]) => {
    predictionsRef.current = items;
    setPredictions(items);
  }, []);

  const publishLaneResult = useCallback((next: LaneResult) => {
    const smoothed = smoothLane(laneResultRef.current, next);
    laneResultRef.current = smoothed;
    setLaneResult(smoothed);
  }, []);

  useEffect(() => {
    laneDetectorRef.current = createLaneDetector();
    let cancelled = false;

    async function loadModel() {
      try {
        const tf = await import("@tensorflow/tfjs");
        const cocoSsd = await import("@tensorflow-models/coco-ssd");
        await tf.ready();
        const model = await cocoSsd.load({ base: "lite_mobilenet_v2" });
        if (!cancelled) {
          modelRef.current = model;
          setModelReady(true);
          setStatus((current) => current.startsWith("Analysing") || current.startsWith("Live") ? current : "Model ready — choose a source");
        }
      } catch (error) {
        console.error(error);
        if (!cancelled) setStatus("Lane detection is ready; object model could not load");
      }
    }

    loadModel();
    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (fileUrlRef.current) URL.revokeObjectURL(fileUrlRef.current);
    };
  }, []);

  const stopCurrentSource = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (fileUrlRef.current) {
      URL.revokeObjectURL(fileUrlRef.current);
      fileUrlRef.current = null;
    }

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
      video.removeAttribute("src");
      video.load();
    }
    publishPredictions([]);
    laneResultRef.current = EMPTY_LANE;
    setLaneResult(EMPTY_LANE);
  }, [publishPredictions]);

  const drawLaneOverlay = useCallback((ctx: CanvasRenderingContext2D, lane: LaneResult) => {
    if (!showLanes || (!lane.left && !lane.right)) return;

    ctx.save();
    ctx.setLineDash([]);
    ctx.lineCap = "round";

    if (lane.left && lane.right) {
      ctx.beginPath();
      ctx.moveTo(lane.left[0].x, lane.left[0].y);
      ctx.lineTo(lane.right[0].x, lane.right[0].y);
      ctx.lineTo(lane.right[1].x, lane.right[1].y);
      ctx.lineTo(lane.left[1].x, lane.left[1].y);
      ctx.closePath();
      ctx.fillStyle = lane.departure === "centered"
        ? "rgba(73, 225, 174, 0.13)"
        : "rgba(255, 190, 92, 0.14)";
      ctx.fill();

      const nearCentreX = (lane.left[1].x + lane.right[1].x) / 2;
      const farCentreX = (lane.left[0].x + lane.right[0].x) / 2;
      ctx.strokeStyle = "rgba(255,255,255,.55)";
      ctx.lineWidth = 2;
      ctx.setLineDash([12, 12]);
      ctx.beginPath();
      ctx.moveTo(farCentreX, lane.left[0].y);
      ctx.lineTo(nearCentreX, lane.left[1].y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const drawBoundary = (segment: [LanePoint, LanePoint] | null) => {
      if (!segment) return;
      ctx.strokeStyle = "#49e1ae";
      ctx.shadowColor = "rgba(73,225,174,.65)";
      ctx.shadowBlur = 10;
      ctx.lineWidth = Math.max(3, ctx.canvas.width / 420);
      ctx.beginPath();
      ctx.moveTo(segment[0].x, segment[0].y);
      ctx.lineTo(segment[1].x, segment[1].y);
      ctx.stroke();
      ctx.shadowBlur = 0;
    };

    drawBoundary(lane.left);
    drawBoundary(lane.right);

    const confidenceText = `LANE ${Math.round(lane.confidence * 100)}%`;
    ctx.font = "700 13px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "rgba(3,10,13,.78)";
    ctx.fillRect(16, ctx.canvas.height - 48, 112, 28);
    ctx.fillStyle = "#8fffd6";
    ctx.fillText(confidenceText, 25, ctx.canvas.height - 29);
    ctx.restore();
  }, [showLanes]);

  const drawPredictions = useCallback((ctx: CanvasRenderingContext2D, items: Prediction[]) => {
    for (const item of items) {
      const [x, y, width, height] = item.bbox;
      const colour = colourForClass(item.class);
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([]);
      ctx.strokeRect(x, y, width, height);

      const label = `#${item.id} ${item.class.toUpperCase()} ${Math.round(item.score * 100)}%`;
      ctx.font = "600 13px ui-monospace, SFMono-Regular, Menlo, monospace";
      const metrics = ctx.measureText(label);
      const labelWidth = metrics.width + 14;
      const labelY = Math.max(0, y - 24);
      ctx.fillStyle = colour;
      ctx.fillRect(x, labelY, labelWidth, 22);
      ctx.fillStyle = "#071015";
      ctx.fillText(label, x + 7, labelY + 15);

      ctx.fillStyle = "rgba(7, 16, 21, .78)";
      ctx.fillRect(x, y + height - 20, 58, 20);
      ctx.fillStyle = "#fff";
      ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillText(`~${item.distanceHint}m`, x + 6, y + height - 6);
    }
  }, []);

  const trackDetections = useCallback((raw: RawPrediction[], videoWidth: number, videoHeight: number) => {
    const previous = predictionsRef.current;
    const used = new Set<number>();

    return raw.map((item) => {
      const [cx, cy] = centre(item.bbox);
      let best: Prediction | undefined;
      let bestDistance = Number.POSITIVE_INFINITY;

      for (const old of previous) {
        if (old.class !== item.class || used.has(old.id)) continue;
        const [ox, oy] = centre(old.bbox);
        const delta = Math.hypot(cx - ox, cy - oy) / Math.hypot(videoWidth, videoHeight);
        if (delta < bestDistance && delta < 0.12) {
          best = old;
          bestDistance = delta;
        }
      }

      const id = best?.id ?? nextTrackIdRef.current++;
      used.add(id);
      const distanceHint = Math.max(1, Math.round(24 * (1 - Math.min(0.96, item.bbox[3] / videoHeight))));
      return { ...item, id, distanceHint };
    });
  }, []);

  const runLoop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frameCount = 0;
    let fpsStartedAt = performance.now();

    const frame = (now: number) => {
      if (!video.videoWidth || !video.videoHeight) {
        rafRef.current = requestAnimationFrame(frame);
        return;
      }

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      if (video.readyState >= 2 && laneDetectorRef.current && now - lastLaneInferenceRef.current > 110) {
        lastLaneInferenceRef.current = now;
        try {
          publishLaneResult(laneDetectorRef.current(video));
        } catch (error) {
          console.error("Lane detection error", error);
        }
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawLaneOverlay(ctx, laneResultRef.current);
      drawPredictions(ctx, predictionsRef.current);

      frameCount += 1;
      if (now - fpsStartedAt >= 1000) {
        setFps(Math.round((frameCount * 1000) / (now - fpsStartedAt)));
        frameCount = 0;
        fpsStartedAt = now;
      }

      if (modelRef.current && !inferenceBusyRef.current && now - lastInferenceRef.current > 180 && video.readyState >= 2) {
        lastInferenceRef.current = now;
        inferenceBusyRef.current = true;
        modelRef.current.detect(video, 20, 0.34)
          .then((detected) => {
            const raw = detected
              .filter((item) => ROAD_CLASSES.has(item.class))
              .map((item) => ({
                bbox: item.bbox as [number, number, number, number],
                class: item.class,
                score: item.score
              }));
            publishPredictions(trackDetections(raw, video.videoWidth, video.videoHeight));
          })
          .catch((error) => console.error("Detection error", error))
          .finally(() => { inferenceBusyRef.current = false; });
      }

      rafRef.current = requestAnimationFrame(frame);
    };

    rafRef.current = requestAnimationFrame(frame);
  }, [drawLaneOverlay, drawPredictions, publishLaneResult, publishPredictions, trackDetections]);

  useEffect(() => {
    if (mode === "idle") return;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    runLoop();
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [mode, runLoop]);

  const startCamera = async () => {
    stopCurrentSource();
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("Camera access is not supported in this browser");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
      setMode("camera");
      setStatus(modelReady ? "Live lane + object perception active" : "Live lane detection active — object model loading");
    } catch (error) {
      console.error(error);
      setStatus("Camera permission was denied or unavailable");
    }
  };

  const waitForVideo = (video: HTMLVideoElement) => new Promise<void>((resolve, reject) => {
    if (video.readyState >= 2) return resolve();
    const onReady = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error("Browser could not decode this video")); };
    const cleanup = () => {
      video.removeEventListener("loadeddata", onReady);
      video.removeEventListener("error", onError);
    };
    video.addEventListener("loadeddata", onReady, { once: true });
    video.addEventListener("error", onError, { once: true });
  });

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    stopCurrentSource();
    const video = videoRef.current;
    if (!video) return;

    try {
      setStatus(`Loading ${file.name}…`);
      const url = URL.createObjectURL(file);
      fileUrlRef.current = url;
      video.srcObject = null;
      video.src = url;
      video.loop = true;
      video.muted = true;
      video.playsInline = true;
      video.load();
      await waitForVideo(video);
      setMode("video");
      await video.play().catch(() => undefined);
      setStatus(modelReady ? `Lane + object analysis: ${file.name}` : `Lane analysis: ${file.name} — object model loading`);
    } catch (error) {
      console.error(error);
      setMode("idle");
      setStatus("Could not open that video. Try MP4/H.264 or WebM.");
    }
  };

  const vehicleCount = predictions.filter((item) => VEHICLE_CLASSES.has(item.class)).length;
  const peopleCount = predictions.filter((item) => item.class === "person").length;
  const lanePercent = Math.round(laneResult.confidence * 100);
  const laneState = laneResult.departure === "unknown"
    ? "Searching"
    : laneResult.departure === "centered"
      ? "Centered"
      : `Drift ${laneResult.departure}`;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">ADAS LAB / LANE + OBJECT PERCEPTION</div>
          <h1>Road intelligence, in the browser.</h1>
        </div>
        <div className={`model-pill ${modelReady ? "ready" : ""}`}>
          <span className="pulse" />
          {modelReady ? "OBJECT MODEL READY" : "LANES READY · MODEL LOADING"}
        </div>
      </header>

      <section className="workspace">
        <div className="viewer-card">
          <div className="viewer-toolbar">
            <div className="source-actions">
              <button onClick={startCamera} className="primary-button">Live camera</button>
              <label className="upload-button">
                Upload video
                <input type="file" accept="video/mp4,video/webm,video/quicktime,video/*" onChange={handleUpload} />
              </label>
            </div>
            <label className="toggle">
              <input type="checkbox" checked={showLanes} onChange={(event) => setShowLanes(event.target.checked)} />
              <span>Lane detection</span>
            </label>
          </div>

          <div className="viewport">
            {mode === "idle" && (
              <div className="empty-state">
                <div className="reticle" />
                <strong>Connect a driving feed</strong>
                <span>Lane detection starts immediately from camera or video.</span>
              </div>
            )}
            <video ref={videoRef} className="video" muted playsInline controls={mode === "video"} />
            <canvas ref={canvasRef} className="overlay" />
            <div className="hud hud-left">{mode === "camera" ? "LIVE" : mode === "video" ? "FILE" : "STANDBY"}</div>
            <div className="hud hud-center">{laneState.toUpperCase()}</div>
            <div className="hud hud-right">{fps} FPS</div>
          </div>

          <div className="statusbar">
            <span>{status}</span>
            <span>Lane CV + client-side AI · no frame upload</span>
          </div>
        </div>

        <aside className="side-panel">
          <div className="panel-card metric-grid">
            <div><span>Lane confidence</span><strong>{lanePercent}%</strong></div>
            <div><span>Lane position</span><strong className="metric-text">{laneState}</strong></div>
            <div><span>Road users</span><strong>{vehicleCount}</strong></div>
            <div><span>Pedestrians</span><strong>{peopleCount}</strong></div>
            <div><span>Signals/signs</span><strong>{predictions.filter((p) => ["traffic light", "stop sign"].includes(p.class)).length}</strong></div>
            <div><span>Tracking</span><strong>{predictions.length}</strong></div>
          </div>

          <div className="panel-card">
            <div className="panel-heading"><span>Lane perception</span><span>{lanePercent}%</span></div>
            <div className="lane-readout">
              <div className={`lane-state ${laneResult.departure}`}>{laneState}</div>
              <div className="confidence-track"><span style={{ width: `${lanePercent}%` }} /></div>
              <p>Edge + colour ROI fitting with temporal smoothing. Best on forward-facing dashcam footage with visible road markings.</p>
            </div>
          </div>

          <div className="panel-card">
            <div className="panel-heading"><span>Tracked objects</span><span>{predictions.length}</span></div>
            <div className="detection-list">
              {predictions.length === 0 ? (
                <p>No tracked road objects yet.</p>
              ) : predictions.map((item) => (
                <div className="detection-row" key={item.id}>
                  <span className="dot" style={{ background: colourForClass(item.class) }} />
                  <span>#{item.id} {item.class}</span>
                  <strong>~{item.distanceHint}m</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="panel-card future-card">
            <div className="panel-heading"><span>Bird&apos;s-eye view</span><span className="tag">PHASE 2</span></div>
            <div className="mini-scene">
              <div className="road-plane" />
              <div className="ego-car">EGO</div>
              {predictions.filter((p) => p.class !== "traffic light" && p.class !== "stop sign").map((item) => {
                const videoWidth = videoRef.current?.videoWidth || 1;
                const [cx] = centre(item.bbox);
                const lateral = Math.max(10, Math.min(90, (cx / videoWidth) * 100));
                const depth = Math.max(18, Math.min(78, 82 - (item.distanceHint / 24) * 64));
                return (
                  <span
                    key={item.id}
                    className={`bev-object ${item.class === "person" ? "person" : "vehicle"}`}
                    style={{ left: `${lateral}%`, top: `${depth}%` }}
                    title={`#${item.id} ${item.class}`}
                  >
                    {item.id}
                  </span>
                );
              })}
            </div>
            <p>Lane geometry now drives the forward perception overlay. Perspective-calibrated BEV lanes are next.</p>
          </div>
        </aside>
      </section>

      <footer>
        Prototype only — lane, distance and BEV outputs are experimental and not safety-grade measurements.
      </footer>
    </main>
  );
}
