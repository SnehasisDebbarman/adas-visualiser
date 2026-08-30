"use client";

import dynamic from "next/dynamic";
import { ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import type { ObjectDetection } from "@tensorflow-models/coco-ssd";
import { createLaneDetector, type LanePoint, type LaneResult } from "@/lib/laneDetection";
import { createUfldv2Detector, type Ufldv2Detector } from "@/lib/ufldv2";

const AdasScene = dynamic(() => import("@/components/AdasScene"), { ssr: false });

type Prediction = { id: number; bbox: [number, number, number, number]; class: string; score: number; distanceHint: number };
type RawPrediction = Omit<Prediction, "id" | "distanceHint">;
type SourceMode = "idle" | "camera" | "video";
type LaneEngine = "idle" | "loading" | "yolop-webgpu" | "yolop-wasm" | "fallback";

const ROAD_CLASSES = new Set(["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]);
const VEHICLE_CLASSES = new Set(["bicycle", "car", "motorcycle", "bus", "truck"]);
const EMPTY_LANE: LaneResult = { left: null, right: null, confidence: 0, centerOffset: 0, departure: "unknown" };

function colourForClass(name: string) {
  if (name === "person") return "#ffcc66";
  if (name === "traffic light" || name === "stop sign") return "#ff6b6b";
  return "#6ee7ff";
}
function centre(bbox: [number, number, number, number]) { return [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2] as const; }
function lerpPoint(a: LanePoint, b: LanePoint, alpha: number): LanePoint { return { x: a.x + (b.x - a.x) * alpha, y: a.y + (b.y - a.y) * alpha }; }
function smoothSegment(a: [LanePoint, LanePoint] | null, b: [LanePoint, LanePoint] | null, alpha: number): [LanePoint, LanePoint] | null {
  if (!b) return a;
  if (!a) return b;
  return [lerpPoint(a[0], b[0], alpha), lerpPoint(a[1], b[1], alpha)];
}
function smoothCurve(a: LanePoint[] | undefined, b: LanePoint[] | undefined, alpha: number) {
  if (!b?.length) return a;
  if (!a?.length || a.length !== b.length) return b;
  return b.map((point, index) => lerpPoint(a[index], point, alpha));
}
function segmentFromCurve(curve?: LanePoint[]): [LanePoint, LanePoint] | null {
  return curve && curve.length >= 2 ? [curve[0], curve[curve.length - 1]] : null;
}
function smoothLane(a: LaneResult, b: LaneResult): LaneResult {
  const confidence = b.confidence > 0 ? a.confidence + (b.confidence - a.confidence) * 0.38 : a.confidence * 0.72;
  const leftCurve = b.leftCurve ? smoothCurve(a.leftCurve, b.leftCurve, 0.42) : confidence > 0.14 ? a.leftCurve : undefined;
  const rightCurve = b.rightCurve ? smoothCurve(a.rightCurve, b.rightCurve, 0.42) : confidence > 0.14 ? a.rightCurve : undefined;
  const left = segmentFromCurve(leftCurve) ?? (b.left ? smoothSegment(a.left, b.left, 0.38) : confidence > 0.14 ? a.left : null);
  const right = segmentFromCurve(rightCurve) ?? (b.right ? smoothSegment(a.right, b.right, 0.38) : confidence > 0.14 ? a.right : null);
  return {
    left,
    right,
    leftCurve,
    rightCurve,
    confidence,
    centerOffset: a.centerOffset + (b.centerOffset - a.centerOffset) * 0.32,
    departure: b.departure === "unknown" && confidence > 0.14 ? a.departure : b.departure
  };
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<ObjectDetection | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const fileUrlRef = useRef<string | null>(null);
  const predictionsRef = useRef<Prediction[]>([]);
  const laneResultRef = useRef<LaneResult>(EMPTY_LANE);
  const fallbackLaneRef = useRef<ReturnType<typeof createLaneDetector> | null>(null);
  const neuralLaneRef = useRef<Ufldv2Detector | null>(null);
  const laneLoadRef = useRef<Promise<void> | null>(null);
  const objectLoadRef = useRef<Promise<void> | null>(null);
  const nextTrackIdRef = useRef(1);
  const lastInferenceRef = useRef(0);
  const lastLaneInferenceRef = useRef(0);
  const inferenceBusyRef = useRef(false);
  const laneInferenceBusyRef = useRef(false);
  const laneEngineRef = useRef<LaneEngine>("idle");

  const [mode, setMode] = useState<SourceMode>("idle");
  const [status, setStatus] = useState("Choose a source — perception models load on demand");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [laneResult, setLaneResult] = useState<LaneResult>(EMPTY_LANE);
  const [fps, setFps] = useState(0);
  const [modelReady, setModelReady] = useState(false);
  const [laneEngine, setLaneEngine] = useState<LaneEngine>("idle");
  const [showLanes, setShowLanes] = useState(true);
  const [show3D, setShow3D] = useState(false);

  const setLaneEngineState = useCallback((engine: LaneEngine) => { laneEngineRef.current = engine; setLaneEngine(engine); }, []);
  const publishPredictions = useCallback((items: Prediction[]) => { predictionsRef.current = items; setPredictions(items); }, []);
  const publishLaneResult = useCallback((next: LaneResult) => {
    const smoothed = smoothLane(laneResultRef.current, next);
    laneResultRef.current = smoothed;
    setLaneResult(smoothed);
  }, []);

  useEffect(() => {
    fallbackLaneRef.current = createLaneDetector();
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (fileUrlRef.current) URL.revokeObjectURL(fileUrlRef.current);
      void neuralLaneRef.current?.dispose();
    };
  }, []);

  const ensureLaneModel = useCallback(() => {
    if (neuralLaneRef.current || laneLoadRef.current) return laneLoadRef.current ?? Promise.resolve();
    setLaneEngineState("loading");
    setStatus("Loading YOLOP lane model…");
    laneLoadRef.current = (async () => {
      try {
        const detector = await createUfldv2Detector();
        neuralLaneRef.current = detector;
        setLaneEngineState(detector.backend === "webgpu" ? "yolop-webgpu" : "yolop-wasm");
        setStatus(`YOLOP curved-lane model ready (${detector.backend.toUpperCase()})`);
      } catch (error) {
        console.error("YOLOP load failed", error);
        setLaneEngineState("fallback");
        setStatus("YOLOP unavailable — using lightweight lane fallback");
      } finally {
        laneLoadRef.current = null;
      }
    })();
    return laneLoadRef.current;
  }, [setLaneEngineState]);

  const ensureObjectModel = useCallback(() => {
    if (modelRef.current || objectLoadRef.current) return objectLoadRef.current ?? Promise.resolve();
    objectLoadRef.current = (async () => {
      try {
        const tf = await import("@tensorflow/tfjs");
        const coco = await import("@tensorflow-models/coco-ssd");
        await tf.ready();
        modelRef.current = await coco.load({ base: "lite_mobilenet_v2" });
        setModelReady(true);
      } catch (error) {
        console.error("Object model load failed", error);
      } finally {
        objectLoadRef.current = null;
      }
    })();
    return objectLoadRef.current;
  }, []);

  const startPerceptionModels = useCallback(async () => {
    await ensureLaneModel();
    setShow3D(true);
    window.setTimeout(() => { void ensureObjectModel(); }, 1400);
  }, [ensureLaneModel, ensureObjectModel]);

  const stopCurrentSource = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (fileUrlRef.current) { URL.revokeObjectURL(fileUrlRef.current); fileUrlRef.current = null; }
    const video = videoRef.current;
    if (video) { video.pause(); video.srcObject = null; video.removeAttribute("src"); video.load(); }
    publishPredictions([]);
    laneResultRef.current = EMPTY_LANE;
    setLaneResult(EMPTY_LANE);
    setShow3D(false);
  }, [publishPredictions]);

  const drawLaneOverlay = useCallback((ctx: CanvasRenderingContext2D, lane: LaneResult) => {
    if (!showLanes) return;
    const left = lane.leftCurve?.length ? lane.leftCurve : lane.left ? [...lane.left] : [];
    const right = lane.rightCurve?.length ? lane.rightCurve : lane.right ? [...lane.right] : [];
    if (!left.length && !right.length) return;

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const neural = laneEngineRef.current.startsWith("yolop");

    if (left.length >= 2 && right.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(left[0].x, left[0].y);
      for (let i = 1; i < left.length; i++) ctx.lineTo(left[i].x, left[i].y);
      for (let i = right.length - 1; i >= 0; i--) ctx.lineTo(right[i].x, right[i].y);
      ctx.closePath();
      ctx.fillStyle = lane.departure === "centered" ? "rgba(64,151,255,.13)" : "rgba(255,190,92,.14)";
      ctx.fill();
    }

    const drawCurve = (curve: LanePoint[]) => {
      if (curve.length < 2) return;
      ctx.strokeStyle = neural ? "#4f9cff" : "#49e1ae";
      ctx.shadowColor = neural ? "rgba(79,156,255,.82)" : "rgba(73,225,174,.65)";
      ctx.shadowBlur = 9;
      ctx.lineWidth = Math.max(3, ctx.canvas.width / 360);
      ctx.beginPath();
      ctx.moveTo(curve[0].x, curve[0].y);
      for (let i = 1; i < curve.length; i++) ctx.lineTo(curve[i].x, curve[i].y);
      ctx.stroke();
    };

    drawCurve(left);
    drawCurve(right);
    ctx.restore();
  }, [showLanes]);

  const drawPredictions = useCallback((ctx: CanvasRenderingContext2D, items: Prediction[]) => {
    for (const item of items) {
      const [x, y, w, h] = item.bbox;
      const colour = colourForClass(item.class);
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);
      const label = `#${item.id} ${item.class.toUpperCase()} ${Math.round(item.score * 100)}%`;
      ctx.font = "600 13px ui-monospace,monospace";
      const lw = ctx.measureText(label).width + 14;
      const ly = Math.max(0, y - 24);
      ctx.fillStyle = colour;
      ctx.fillRect(x, ly, lw, 22);
      ctx.fillStyle = "#071015";
      ctx.fillText(label, x + 7, ly + 15);
    }
  }, []);

  const trackDetections = useCallback((raw: RawPrediction[], vw: number, vh: number) => {
    const previous = predictionsRef.current;
    const used = new Set<number>();
    return raw.map((item) => {
      const [cx, cy] = centre(item.bbox);
      let best: Prediction | undefined;
      let bestDistance = Infinity;
      for (const old of previous) {
        if (old.class !== item.class || used.has(old.id)) continue;
        const [ox, oy] = centre(old.bbox);
        const distance = Math.hypot(cx - ox, cy - oy) / Math.hypot(vw, vh);
        if (distance < bestDistance && distance < 0.12) { best = old; bestDistance = distance; }
      }
      const id = best?.id ?? nextTrackIdRef.current++;
      used.add(id);
      return { ...item, id, distanceHint: Math.max(1, Math.round(24 * (1 - Math.min(0.96, item.bbox[3] / vh)))) };
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
    const mobile = typeof navigator !== "undefined" && /iPhone|iPad|Android/i.test(navigator.userAgent);

    const frame = (now: number) => {
      if (!video.videoWidth || !video.videoHeight) { rafRef.current = requestAnimationFrame(frame); return; }
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }

      const neural = neuralLaneRef.current;
      const neuralInterval = mobile ? 720 : neural?.backend === "webgpu" ? 240 : 500;
      if (video.readyState >= 2 && neural && !laneInferenceBusyRef.current && now - lastLaneInferenceRef.current > neuralInterval) {
        lastLaneInferenceRef.current = now;
        laneInferenceBusyRef.current = true;
        neural.detect(video)
          .then(publishLaneResult)
          .catch((error) => {
            console.error("YOLOP inference failed", error);
            void neural.dispose();
            neuralLaneRef.current = null;
            setLaneEngineState("fallback");
            setStatus("YOLOP inference failed — using lane fallback");
          })
          .finally(() => { laneInferenceBusyRef.current = false; });
      } else if (video.readyState >= 2 && laneEngineRef.current === "fallback" && fallbackLaneRef.current && now - lastLaneInferenceRef.current > 180) {
        lastLaneInferenceRef.current = now;
        try { publishLaneResult(fallbackLaneRef.current(video)); } catch (error) { console.error(error); }
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawLaneOverlay(ctx, laneResultRef.current);
      drawPredictions(ctx, predictionsRef.current);

      frameCount++;
      if (now - fpsStartedAt >= 1000) {
        setFps(Math.round(frameCount * 1000 / (now - fpsStartedAt)));
        frameCount = 0;
        fpsStartedAt = now;
      }

      const objectInterval = mobile ? 700 : 280;
      if (modelRef.current && !inferenceBusyRef.current && now - lastInferenceRef.current > objectInterval && video.readyState >= 2) {
        lastInferenceRef.current = now;
        inferenceBusyRef.current = true;
        modelRef.current.detect(video, mobile ? 10 : 16, 0.38)
          .then((detected) => {
            const raw = detected.filter((item) => ROAD_CLASSES.has(item.class)).map((item) => ({ bbox: item.bbox as [number, number, number, number], class: item.class, score: item.score }));
            publishPredictions(trackDetections(raw, video.videoWidth, video.videoHeight));
          })
          .catch(console.error)
          .finally(() => { inferenceBusyRef.current = false; });
      }
      rafRef.current = requestAnimationFrame(frame);
    };
    rafRef.current = requestAnimationFrame(frame);
  }, [drawLaneOverlay, drawPredictions, publishLaneResult, publishPredictions, setLaneEngineState, trackDetections]);

  useEffect(() => {
    if (mode === "idle") return;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    runLoop();
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [mode, runLoop]);

  const startCamera = async () => {
    stopCurrentSource();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
      setMode("camera");
      setStatus("Camera ready — loading curved-lane perception…");
      void startPerceptionModels();
    } catch (error) {
      console.error(error);
      setStatus("Camera permission was denied or unavailable");
    }
  };

  const waitForVideo = (video: HTMLVideoElement) => new Promise<void>((resolve, reject) => {
    if (video.readyState >= 2) return resolve();
    const ready = () => { clean(); resolve(); };
    const error = () => { clean(); reject(new Error("decode")); };
    const clean = () => { video.removeEventListener("loadeddata", ready); video.removeEventListener("error", error); };
    video.addEventListener("loadeddata", ready, { once: true });
    video.addEventListener("error", error, { once: true });
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
      video.src = url;
      video.loop = true;
      video.muted = true;
      video.playsInline = true;
      video.load();
      await waitForVideo(video);
      setMode("video");
      await video.play().catch(() => undefined);
      setStatus("Video ready — loading curved-lane perception…");
      void startPerceptionModels();
    } catch (error) {
      console.error(error);
      setMode("idle");
      setStatus("Could not open that video. Try MP4/H.264 or WebM.");
    }
  };

  const vehicleCount = predictions.filter((item) => VEHICLE_CLASSES.has(item.class)).length;
  const peopleCount = predictions.filter((item) => item.class === "person").length;
  const lanePercent = Math.round(laneResult.confidence * 100);
  const laneState = laneResult.departure === "unknown" ? "Searching" : laneResult.departure === "centered" ? "Centered" : `Drift ${laneResult.departure}`;
  const laneEngineLabel = laneEngine === "yolop-webgpu" ? "YOLOP GPU" : laneEngine === "yolop-wasm" ? "YOLOP CPU" : laneEngine === "fallback" ? "CV fallback" : laneEngine === "loading" ? "Loading YOLOP" : "Model idle";
  const vw = videoRef.current?.videoWidth || 1;
  const sceneObjects = predictions.map((p) => { const [cx] = centre(p.bbox); return { id: p.id, class: p.class, lateral: (cx / vw - 0.5) * 2, distance: p.distanceHint }; });

  const leftCurve = laneResult.leftCurve;
  const rightCurve = laneResult.rightCurve;
  let laneCurve = 0;
  if (leftCurve?.length && rightCurve?.length) {
    const midIndex = Math.floor(Math.min(leftCurve.length, rightCurve.length) / 2);
    const far = (leftCurve[0].x + rightCurve[0].x) / 2;
    const mid = (leftCurve[midIndex].x + rightCurve[midIndex].x) / 2;
    const near = (leftCurve[leftCurve.length - 1].x + rightCurve[rightCurve.length - 1].x) / 2;
    const heading = (far - near) / vw;
    const bend = (far - 2 * mid + near) / vw;
    laneCurve = heading * 4.4 + bend * 9.5;
  } else if (laneResult.left && laneResult.right) {
    const far = (laneResult.left[0].x + laneResult.right[0].x) / 2;
    const near = (laneResult.left[1].x + laneResult.right[1].x) / 2;
    laneCurve = ((far - near) / vw) * 5;
  }

  return <main className="app-shell">
    <header className="topbar"><div><div className="eyebrow">ADAS LAB / YOLOP CURVED LANES + 3D</div><h1>See the road as a live spatial model.</h1></div><div className={`model-pill ${laneEngine.startsWith("yolop") ? "ready" : ""}`}><span className="pulse"/>{laneEngineLabel.toUpperCase()}</div></header>
    <section className="dual-view">
      <div className="viewer-card">
        <div className="viewer-toolbar"><div className="source-actions"><button onClick={startCamera} className="primary-button">Live camera</button><label className="upload-button">Upload video<input type="file" accept="video/mp4,video/webm,video/quicktime,video/*" onChange={handleUpload}/></label></div><label className="toggle"><input type="checkbox" checked={showLanes} onChange={(e) => setShowLanes(e.target.checked)}/><span>Lane overlay</span></label></div>
        <div className="viewport"><video ref={videoRef} className="video" muted playsInline controls={mode === "video"}/><canvas ref={canvasRef} className="overlay"/><div className="hud hud-left">{mode === "camera" ? "LIVE" : mode === "video" ? "FILE" : "STANDBY"}</div><div className="hud hud-center">{laneState.toUpperCase()}</div><div className="hud hud-right">{fps} FPS</div>{mode === "idle" && <div className="empty-state"><div className="reticle"/><strong>Connect a driving feed</strong><span>Models stay unloaded until you choose a source.</span></div>}</div>
        <div className="statusbar"><span>{status}</span><span>{laneEngineLabel} · {modelReady ? "objects ready" : mode === "idle" ? "objects idle" : "objects deferred"}</span></div>
      </div>
      <div className="scene-card">{show3D ? <AdasScene objects={sceneObjects} lane={{ centerOffset: laneResult.centerOffset, confidence: laneResult.confidence, visible: !!(laneResult.left || laneResult.right), curve: laneCurve }}/> : <div className="adas-scene" style={{ display: "grid", placeItems: "center" }}><div className="scene-badge">3D LOADS AFTER LANE MODEL</div></div>}<div className="scene-status"><span>LIVE SPATIAL VIEW</span><span>{predictions.length} tracked · lane {lanePercent}%</span></div></div>
    </section>
    <section className="telemetry"><div><span>Lane model</span><strong>{laneEngineLabel}</strong></div><div><span>Lane confidence</span><strong>{lanePercent}%</strong></div><div><span>Lane position</span><strong>{laneState}</strong></div><div><span>Vehicles</span><strong>{vehicleCount}</strong></div><div><span>Pedestrians</span><strong>{peopleCount}</strong></div></section>
    <footer>Experimental visualisation only — neural lane, object depth and 3D positions are not safety-grade measurements.</footer>
  </main>;
}
