"use client";

import { ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import type { ObjectDetection } from "@tensorflow-models/coco-ssd";

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

function colourForClass(name: string) {
  if (name === "person") return "#ffcc66";
  if (name === "traffic light" || name === "stop sign") return "#ff6b6b";
  return "#6ee7ff";
}

function centre(bbox: [number, number, number, number]) {
  return [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2] as const;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<ObjectDetection | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastInferenceRef = useRef(0);
  const inferenceBusyRef = useRef(false);
  const fileUrlRef = useRef<string | null>(null);
  const predictionsRef = useRef<Prediction[]>([]);
  const nextTrackIdRef = useRef(1);

  const [mode, setMode] = useState<SourceMode>("idle");
  const [status, setStatus] = useState("Loading perception model…");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [fps, setFps] = useState(0);
  const [modelReady, setModelReady] = useState(false);
  const [showRoad, setShowRoad] = useState(true);

  const publishPredictions = useCallback((items: Prediction[]) => {
    predictionsRef.current = items;
    setPredictions(items);
  }, []);

  useEffect(() => {
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
        if (!cancelled) setStatus("Video works, but the perception model could not load");
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
  }, [publishPredictions]);

  const drawRoadEstimate = useCallback((ctx: CanvasRenderingContext2D, width: number, height: number) => {
    if (!showRoad) return;

    const horizon = height * 0.55;
    const centreX = width / 2;
    const nearHalf = width * 0.34;
    const farHalf = width * 0.08;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(centreX - farHalf, horizon);
    ctx.lineTo(centreX + farHalf, horizon);
    ctx.lineTo(centreX + nearHalf, height);
    ctx.lineTo(centreX - nearHalf, height);
    ctx.closePath();
    ctx.fillStyle = "rgba(73, 225, 174, 0.10)";
    ctx.fill();
    ctx.strokeStyle = "rgba(73, 225, 174, 0.85)";
    ctx.lineWidth = 2;
    ctx.setLineDash([12, 10]);
    ctx.stroke();

    ctx.setLineDash([8, 12]);
    ctx.strokeStyle = "rgba(255,255,255,0.72)";
    ctx.beginPath();
    ctx.moveTo(centreX, horizon + height * 0.02);
    ctx.lineTo(centreX, height);
    ctx.stroke();
    ctx.restore();
  }, [showRoad]);

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

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawRoadEstimate(ctx, canvas.width, canvas.height);
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
  }, [drawPredictions, drawRoadEstimate, publishPredictions, trackDetections]);

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
      setStatus(modelReady ? "Live camera perception active" : "Live camera active — waiting for perception model");
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
      setStatus(modelReady ? `Analysing ${file.name}` : `${file.name} loaded — waiting for perception model`);
    } catch (error) {
      console.error(error);
      setMode("idle");
      setStatus("Could not open that video. Try MP4/H.264 or WebM.");
    }
  };

  const vehicleCount = predictions.filter((item) => VEHICLE_CLASSES.has(item.class)).length;
  const peopleCount = predictions.filter((item) => item.class === "person").length;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">ADAS LAB / PERCEPTION</div>
          <h1>Road intelligence, in the browser.</h1>
        </div>
        <div className={`model-pill ${modelReady ? "ready" : ""}`}>
          <span className="pulse" />
          {modelReady ? "MODEL READY" : "LOADING MODEL"}
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
              <input type="checkbox" checked={showRoad} onChange={(event) => setShowRoad(event.target.checked)} />
              <span>Road guide</span>
            </label>
          </div>

          <div className="viewport">
            {mode === "idle" && (
              <div className="empty-state">
                <div className="reticle" />
                <strong>Connect a driving feed</strong>
                <span>Camera and video input work independently from model loading.</span>
              </div>
            )}
            <video ref={videoRef} className="video" muted playsInline controls={mode === "video"} />
            <canvas ref={canvasRef} className="overlay" />
            <div className="hud hud-left">{mode === "camera" ? "LIVE" : mode === "video" ? "FILE" : "STANDBY"}</div>
            <div className="hud hud-right">{fps} FPS</div>
          </div>

          <div className="statusbar">
            <span>{status}</span>
            <span>Client-side inference · no frame upload</span>
          </div>
        </div>

        <aside className="side-panel">
          <div className="panel-card metric-grid">
            <div><span>Road users</span><strong>{vehicleCount}</strong></div>
            <div><span>Pedestrians</span><strong>{peopleCount}</strong></div>
            <div><span>Signals/signs</span><strong>{predictions.filter((p) => ["traffic light", "stop sign"].includes(p.class)).length}</strong></div>
            <div><span>Tracking</span><strong>{predictions.length}</strong></div>
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
                const videoHeight = videoRef.current?.videoHeight || 1;
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
            <p>Tracked 2D detections are now projected into a live BEV preview. True depth and lane geometry are the next model upgrades.</p>
          </div>
        </aside>
      </section>

      <footer>
        Prototype only — distance and BEV positions are estimates, not safety-grade measurements.
      </footer>
    </main>
  );
}
