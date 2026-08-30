"use client";

import { ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import type { ObjectDetection } from "@tensorflow-models/coco-ssd";
import { createLaneDetector, type LanePoint, type LaneResult } from "@/lib/laneDetection";
import AdasScene from "@/components/AdasScene";

type Prediction = { id: number; bbox: [number, number, number, number]; class: string; score: number; distanceHint: number; lastSeen: number; velocity: [number, number, number, number] };
type RawPrediction = { bbox: [number, number, number, number]; class: string; score: number };
type SourceMode = "idle" | "camera" | "video";
const ROAD_CLASSES = new Set(["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]);
const VEHICLE_CLASSES = new Set(["bicycle", "car", "motorcycle", "bus", "truck"]);
const TRACK_GRACE_MS = 2200;
const TRACK_PREDICT_MS = 900;
const EMPTY_LANE: LaneResult = { left: null, right: null, confidence: 0, centerOffset: 0, departure: "unknown" };
function colourForClass(name: string) { if (name === "person") return "#ffcc66"; if (name === "traffic light" || name === "stop sign") return "#ff6b6b"; return "#6ee7ff"; }
function centre(bbox: [number, number, number, number]) { return [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2] as const; }
function lerpPoint(a: LanePoint, b: LanePoint, alpha: number): LanePoint { return { x: a.x + (b.x-a.x)*alpha, y: a.y + (b.y-a.y)*alpha }; }
function smoothSegment(a: [LanePoint,LanePoint]|null,b:[LanePoint,LanePoint]|null,alpha:number):[LanePoint,LanePoint]|null { if(!b)return a;if(!a)return b;return[lerpPoint(a[0],b[0],alpha),lerpPoint(a[1],b[1],alpha)]; }
function smoothLane(a:LaneResult,b:LaneResult):LaneResult { const confidence=b.confidence>0?a.confidence+(b.confidence-a.confidence)*.28:a.confidence*.82;return{left:b.left?smoothSegment(a.left,b.left,.24):confidence>.18?a.left:null,right:b.right?smoothSegment(a.right,b.right,.24):confidence>.18?a.right:null,confidence,centerOffset:a.centerOffset+(b.centerOffset-a.centerOffset)*.2,departure:b.departure==="unknown"&&confidence>.18?a.departure:b.departure}; }
function distanceFromBox(bbox:[number,number,number,number],vh:number){return Math.max(1,Math.round(24*(1-Math.min(.96,bbox[3]/vh))))}
function boxVisible(bbox:[number,number,number,number],vw:number,vh:number){const[x,y,w,h]=bbox,margin=Math.max(vw,vh)*.035;return x+w>-margin&&y+h>-margin&&x<vw+margin&&y<vh+margin}
function bboxIou(a:[number,number,number,number],b:[number,number,number,number]){const ax2=a[0]+a[2],ay2=a[1]+a[3],bx2=b[0]+b[2],by2=b[1]+b[3],ix=Math.max(0,Math.min(ax2,bx2)-Math.max(a[0],b[0])),iy=Math.max(0,Math.min(ay2,by2)-Math.max(a[1],b[1])),inter=ix*iy,union=a[2]*a[3]+b[2]*b[3]-inter;return union>0?inter/union:0}
function compatibleTrackClass(a:string,b:string){return a===b||(VEHICLE_CLASSES.has(a)&&VEHICLE_CLASSES.has(b))}

export default function Home(){
 const videoRef=useRef<HTMLVideoElement>(null),canvasRef=useRef<HTMLCanvasElement>(null),modelRef=useRef<ObjectDetection|null>(null),streamRef=useRef<MediaStream|null>(null),rafRef=useRef<number|null>(null),fileUrlRef=useRef<string|null>(null),predictionsRef=useRef<Prediction[]>([]),laneResultRef=useRef<LaneResult>(EMPTY_LANE),laneDetectorRef=useRef<ReturnType<typeof createLaneDetector>|null>(null),nextTrackIdRef=useRef(1),lastInferenceRef=useRef(0),lastLaneInferenceRef=useRef(0),inferenceBusyRef=useRef(false);
 const[mode,setMode]=useState<SourceMode>("idle"),[status,setStatus]=useState("Loading perception model…"),[predictions,setPredictions]=useState<Prediction[]>([]),[laneResult,setLaneResult]=useState<LaneResult>(EMPTY_LANE),[fps,setFps]=useState(0),[modelReady,setModelReady]=useState(false),[showLanes,setShowLanes]=useState(true);
 const publishPredictions=useCallback((items:Prediction[])=>{predictionsRef.current=items;setPredictions(items)},[]);
 const publishLaneResult=useCallback((next:LaneResult)=>{const s=smoothLane(laneResultRef.current,next);laneResultRef.current=s;setLaneResult(s)},[]);
 useEffect(()=>{laneDetectorRef.current=createLaneDetector();let cancelled=false;(async()=>{try{const tf=await import("@tensorflow/tfjs"),coco=await import("@tensorflow-models/coco-ssd");await tf.ready();const model=await coco.load({base:"lite_mobilenet_v2"});if(!cancelled){modelRef.current=model;setModelReady(true);setStatus(c=>c.startsWith("Analysing")||c.startsWith("Live")?c:"Model ready — choose a source")}}catch(e){console.error(e);if(!cancelled)setStatus("Lane detection is ready; object model could not load")}})();return()=>{cancelled=true;if(rafRef.current)cancelAnimationFrame(rafRef.current);streamRef.current?.getTracks().forEach(t=>t.stop());if(fileUrlRef.current)URL.revokeObjectURL(fileUrlRef.current)}},[]);
 const stopCurrentSource=useCallback(()=>{if(rafRef.current)cancelAnimationFrame(rafRef.current);rafRef.current=null;streamRef.current?.getTracks().forEach(t=>t.stop());streamRef.current=null;if(fileUrlRef.current){URL.revokeObjectURL(fileUrlRef.current);fileUrlRef.current=null}const v=videoRef.current;if(v){v.pause();v.srcObject=null;v.removeAttribute("src");v.load()}publishPredictions([]);laneResultRef.current=EMPTY_LANE;setLaneResult(EMPTY_LANE)},[publishPredictions]);
 const drawLaneOverlay=useCallback((ctx:CanvasRenderingContext2D,lane:LaneResult)=>{if(!showLanes||(!lane.left&&!lane.right))return;ctx.save();ctx.lineCap="round";if(lane.left&&lane.right){ctx.beginPath();ctx.moveTo(lane.left[0].x,lane.left[0].y);ctx.lineTo(lane.right[0].x,lane.right[0].y);ctx.lineTo(lane.right[1].x,lane.right[1].y);ctx.lineTo(lane.left[1].x,lane.left[1].y);ctx.closePath();ctx.fillStyle=lane.departure==="centered"?"rgba(73,225,174,.13)":"rgba(255,190,92,.14)";ctx.fill()}for(const segment of[lane.left,lane.right]){if(!segment)continue;ctx.strokeStyle="#49e1ae";ctx.shadowColor="rgba(73,225,174,.65)";ctx.shadowBlur=10;ctx.lineWidth=Math.max(3,ctx.canvas.width/420);ctx.beginPath();ctx.moveTo(segment[0].x,segment[0].y);ctx.lineTo(segment[1].x,segment[1].y);ctx.stroke()}ctx.restore()},[showLanes]);
 const drawPredictions=useCallback((ctx:CanvasRenderingContext2D,items:Prediction[])=>{for(const item of items){const[x,y,w,h]=item.bbox,c=colourForClass(item.class);ctx.strokeStyle=c;ctx.lineWidth=2.5;ctx.strokeRect(x,y,w,h);const label=`#${item.id} ${item.class.toUpperCase()} ${Math.round(item.score*100)}%`;ctx.font="600 13px ui-monospace,monospace";const lw=ctx.measureText(label).width+14,ly=Math.max(0,y-24);ctx.fillStyle=c;ctx.fillRect(x,ly,lw,22);ctx.fillStyle="#071015";ctx.fillText(label,x+7,ly+15)}},[]);
 const trackDetections=useCallback((raw:RawPrediction[],vw:number,vh:number)=>{
  const now=performance.now(),previous=predictionsRef.current,used=new Set<number>(),matchedOld=new Set<number>(),next:Prediction[]=[];
  for(const item of raw){
   const[cx,cy]=centre(item.bbox);let best:Prediction|undefined,bd=Infinity;
   for(const old of previous){if(!compatibleTrackClass(old.class,item.class)||used.has(old.id))continue;const age=Math.min(now-old.lastSeen,TRACK_PREDICT_MS),predicted:[number,number,number,number]=[old.bbox[0]+old.velocity[0]*age,old.bbox[1]+old.velocity[1]*age,Math.max(3,old.bbox[2]+old.velocity[2]*age),Math.max(3,old.bbox[3]+old.velocity[3]*age)],[ox,oy]=centre(predicted),centerDistance=Math.hypot(cx-ox,cy-oy)/Math.hypot(vw,vh),overlap=bboxIou(item.bbox,predicted),cost=centerDistance*.68+(1-overlap)*.32;if((centerDistance<.22||overlap>.08)&&cost<bd){best=old;bd=cost}}
   const id=best?.id??nextTrackIdRef.current++;used.add(id);if(best)matchedOld.add(id);
   const dt=Math.max(50,Math.min(700,now-(best?.lastSeen??now)));
   const velocity:[number,number,number,number]=best?[
    best.velocity[0]*.55+THREEClamp((item.bbox[0]-best.bbox[0])/dt,-.8,.8)*.45,
    best.velocity[1]*.55+THREEClamp((item.bbox[1]-best.bbox[1])/dt,-.8,.8)*.45,
    best.velocity[2]*.55+THREEClamp((item.bbox[2]-best.bbox[2])/dt,-.45,.45)*.45,
    best.velocity[3]*.55+THREEClamp((item.bbox[3]-best.bbox[3])/dt,-.45,.45)*.45
   ]:[0,0,0,0];
   next.push({...item,id,distanceHint:distanceFromBox(item.bbox,vh),lastSeen:now,velocity});
  }
  for(const old of previous){
   if(matchedOld.has(old.id))continue;
   const age=now-old.lastSeen;if(age>TRACK_GRACE_MS)continue;
   const predict=Math.min(age,TRACK_PREDICT_MS);
   const bbox:[number,number,number,number]=[
    old.bbox[0]+old.velocity[0]*predict,
    old.bbox[1]+old.velocity[1]*predict,
    Math.max(3,old.bbox[2]+old.velocity[2]*predict),
    Math.max(3,old.bbox[3]+old.velocity[3]*predict)
   ];
   if(!boxVisible(bbox,vw,vh))continue;
   next.push({...old,bbox,score:Math.max(.18,old.score*(1-age/TRACK_GRACE_MS*.35)),distanceHint:distanceFromBox(bbox,vh)});
  }
  return next;
 },[]);
 const runLoop=useCallback(()=>{const video=videoRef.current,canvas=canvasRef.current;if(!video||!canvas)return;const ctx=canvas.getContext("2d");if(!ctx)return;let fc=0,fs=performance.now();const frame=(now:number)=>{if(!video.videoWidth||!video.videoHeight){rafRef.current=requestAnimationFrame(frame);return}if(canvas.width!==video.videoWidth||canvas.height!==video.videoHeight){canvas.width=video.videoWidth;canvas.height=video.videoHeight}if(video.readyState>=2&&laneDetectorRef.current&&now-lastLaneInferenceRef.current>110){lastLaneInferenceRef.current=now;try{publishLaneResult(laneDetectorRef.current(video))}catch(e){console.error(e)}}ctx.clearRect(0,0,canvas.width,canvas.height);drawLaneOverlay(ctx,laneResultRef.current);drawPredictions(ctx,predictionsRef.current);fc++;if(now-fs>=1000){setFps(Math.round(fc*1000/(now-fs)));fc=0;fs=now}if(modelRef.current&&!inferenceBusyRef.current&&now-lastInferenceRef.current>180&&video.readyState>=2){lastInferenceRef.current=now;inferenceBusyRef.current=true;modelRef.current.detect(video,20,.34).then(d=>{const raw=d.filter(i=>ROAD_CLASSES.has(i.class)).map(i=>({bbox:i.bbox as [number,number,number,number],class:i.class,score:i.score}));publishPredictions(trackDetections(raw,video.videoWidth,video.videoHeight))}).catch(console.error).finally(()=>{inferenceBusyRef.current=false})}rafRef.current=requestAnimationFrame(frame)};rafRef.current=requestAnimationFrame(frame)},[drawLaneOverlay,drawPredictions,publishLaneResult,publishPredictions,trackDetections]);
 useEffect(()=>{if(mode==="idle")return;if(rafRef.current)cancelAnimationFrame(rafRef.current);runLoop();return()=>{if(rafRef.current)cancelAnimationFrame(rafRef.current)}},[mode,runLoop]);
 const startCamera=async()=>{stopCurrentSource();try{const s=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:"environment"},width:{ideal:1920},height:{ideal:1080}},audio:false});streamRef.current=s;const v=videoRef.current;if(!v)return;v.srcObject=s;await v.play();setMode("camera");setStatus(modelReady?"Live lane + object perception active":"Live lane detection active — object model loading")}catch(e){console.error(e);setStatus("Camera permission was denied or unavailable")}};
 const waitForVideo=(v:HTMLVideoElement)=>new Promise<void>((resolve,reject)=>{if(v.readyState>=2)return resolve();const ready=()=>{clean();resolve()},error=()=>{clean();reject(new Error("decode"))},clean=()=>{v.removeEventListener("loadeddata",ready);v.removeEventListener("error",error)};v.addEventListener("loadeddata",ready,{once:true});v.addEventListener("error",error,{once:true})});
 const handleUpload=async(e:ChangeEvent<HTMLInputElement>)=>{const file=e.target.files?.[0];e.target.value="";if(!file)return;stopCurrentSource();const v=videoRef.current;if(!v)return;try{setStatus(`Loading ${file.name}…`);const url=URL.createObjectURL(file);fileUrlRef.current=url;v.src=url;v.loop=true;v.muted=true;v.playsInline=true;v.load();await waitForVideo(v);setMode("video");await v.play().catch(()=>undefined);setStatus(modelReady?`Lane + object analysis: ${file.name}`:`Lane analysis: ${file.name} — object model loading`)}catch(e){console.error(e);setMode("idle");setStatus("Could not open that video. Try MP4/H.264 or WebM.")}};
 const vehicleCount=predictions.filter(i=>VEHICLE_CLASSES.has(i.class)).length,peopleCount=predictions.filter(i=>i.class==="person").length,lanePercent=Math.round(laneResult.confidence*100),laneState=laneResult.departure==="unknown"?"Searching":laneResult.departure==="centered"?"Centered":`Drift ${laneResult.departure}`;
 const vw=videoRef.current?.videoWidth||1,vh=videoRef.current?.videoHeight||1;
 const sceneObjects=predictions.map(p=>{const[cx]=centre(p.bbox);return{id:p.id,class:p.class,lateral:(cx/vw-.5)*2,distance:p.distanceHint}});
 let laneCurve=0;if(laneResult.left&&laneResult.right){const far=(laneResult.left[0].x+laneResult.right[0].x)/2,near=(laneResult.left[1].x+laneResult.right[1].x)/2;laneCurve=((far-near)/vw)*5}else if(laneResult.left){laneCurve=((laneResult.left[0].x-laneResult.left[1].x)/vw)*3}else if(laneResult.right){laneCurve=((laneResult.right[0].x-laneResult.right[1].x)/vw)*3}
 const lanePoints=[...(laneResult.left??[]),...(laneResult.right??[])];
 const farLaneY=lanePoints.length?Math.min(...lanePoints.map(p=>p.y)):vh*.68;
 const visibleRoadRatio=Math.max(.08,Math.min(.92,1-farLaneY/vh));
 const roadDepth=Math.max(34,Math.min(88,34+visibleRoadRatio*66*(.72+laneResult.confidence*.28)));
 return <main className="app-shell"><header className="topbar"><div><div className="eyebrow">ADAS LAB / 3D PERCEPTION</div><h1>See the road as a live spatial model.</h1></div><div className={`model-pill ${modelReady?"ready":""}`}><span className="pulse"/>{modelReady?"OBJECT MODEL READY":"LANES READY · MODEL LOADING"}</div></header>
 <section className="dual-view"><div className="viewer-card"><div className="viewer-toolbar"><div className="source-actions"><button onClick={startCamera} className="primary-button">Live camera</button><label className="upload-button">Upload video<input type="file" accept="video/mp4,video/webm,video/quicktime,video/*" onChange={handleUpload}/></label></div><label className="toggle"><input type="checkbox" checked={showLanes} onChange={e=>setShowLanes(e.target.checked)}/><span>Lane overlay</span></label></div><div className="viewport"><video ref={videoRef} className="video" muted playsInline controls={mode==="video"}/><canvas ref={canvasRef} className="overlay"/><div className="hud hud-left">{mode==="camera"?"LIVE":mode==="video"?"FILE":"STANDBY"}</div><div className="hud hud-center">{laneState.toUpperCase()}</div><div className="hud hud-right">{fps} FPS</div>{mode==="idle"&&<div className="empty-state"><div className="reticle"/><strong>Connect a driving feed</strong><span>Camera/video perception feeds the 3D scene.</span></div>}</div><div className="statusbar"><span>{status}</span><span>Client-side perception</span></div></div>
 <div className="scene-card"><AdasScene objects={sceneObjects} lane={{centerOffset:laneResult.centerOffset,confidence:laneResult.confidence,visible:!!(laneResult.left||laneResult.right),curve:laneCurve,roadDepth}}/><div className="scene-status"><span>LIVE SPATIAL VIEW</span><span>{predictions.length} tracked · lane {lanePercent}%</span></div></div></section>
 <section className="telemetry"><div><span>Lane confidence</span><strong>{lanePercent}%</strong></div><div><span>Lane position</span><strong>{laneState}</strong></div><div><span>Vehicles</span><strong>{vehicleCount}</strong></div><div><span>Pedestrians</span><strong>{peopleCount}</strong></div><div><span>Tracked objects</span><strong>{predictions.length}</strong></div></section>
 <footer>Experimental visualisation only — object depth and 3D positions are estimates and are not safety-grade measurements.</footer></main>;
}

function THREEClamp(value:number,min:number,max:number){return Math.max(min,Math.min(max,value))}
