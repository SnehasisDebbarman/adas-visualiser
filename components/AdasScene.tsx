"use client";

import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";

export type SceneObject = { id: number; class: string; lateral: number; distance: number };
export type SceneLane = { centerOffset: number; confidence: number; visible: boolean; curve: number };
type Props = { objects: SceneObject[]; lane: SceneLane };

function Vehicle({ x, z, kind }: { x: number; z: number; kind: string }) {
  const truck = kind === "truck" || kind === "bus";
  const motorcycle = kind === "motorcycle" || kind === "bicycle";
  const size: [number, number, number] = truck ? [1.95, 1.75, 4.8] : motorcycle ? [0.7, 1.05, 2] : [1.7, 1.1, 3.7];
  return <group position={[x, size[1] / 2 + .04, -z]}>
    <mesh castShadow><boxGeometry args={size}/><meshStandardMaterial color={truck ? "#d5d9df" : "#eceef1"} roughness={.38} metalness={.16}/></mesh>
    {!motorcycle && <mesh position={[0,size[1]*.36,-.28]}><boxGeometry args={[size[0]*.76,size[1]*.46,size[2]*.48]}/><meshStandardMaterial color="#aeb7c4" roughness={.28} metalness={.1}/></mesh>}
  </group>;
}

function Pedestrian({ x, z }: { x:number; z:number }) {
  return <group position={[x,0,-z]}><mesh position={[0,.9,0]}><capsuleGeometry args={[.18,.75,5,10]}/><meshStandardMaterial color="#e3e5e8"/></mesh><mesh position={[0,1.55,0]}><sphereGeometry args={[.22,12,12]}/><meshStandardMaterial color="#f3f4f5"/></mesh></group>;
}

function TrafficSignal({ x, z }: { x:number; z:number }) {
  return <group position={[x,0,-z]}><mesh position={[0,2.3,0]}><cylinderGeometry args={[.045,.045,4.6,8]}/><meshStandardMaterial color="#c5c9d0"/></mesh><mesh position={[0,4.42,0]}><boxGeometry args={[.48,.95,.28]}/><meshStandardMaterial color="#323845"/></mesh><mesh position={[0,4.67,.16]}><sphereGeometry args={[.105,10,10]}/><meshBasicMaterial color="#ff5a64"/></mesh><mesh position={[0,4.42,.16]}><sphereGeometry args={[.105,10,10]}/><meshBasicMaterial color="#ffcf5a"/></mesh><mesh position={[0,4.17,.16]}><sphereGeometry args={[.105,10,10]}/><meshBasicMaterial color="#55e69c"/></mesh></group>;
}

function RoadScene({ objects, lane }: Props) {
  const laneShift = THREE.MathUtils.clamp(lane.centerOffset * 2.8, -1.25, 1.25);
  const curve = THREE.MathUtils.clamp(lane.curve, -1.4, 1.4) * 13;
  const laneHalfWidth = 1.8;
  const dashes = useMemo(()=>Array.from({length:20},(_,i)=>4+i*4.5),[]);
  const pathPieces = useMemo(()=>Array.from({length:23},(_,i)=>2.5+i*3.5),[]);
  const bend=(z:number)=>{
    const t=THREE.MathUtils.clamp(z/86,0,1);
    return curve*(t*t*(3-2*t));
  };
  const heading=(z:number)=>{
    const dz=.5;
    return Math.atan2(bend(z+dz)-bend(Math.max(0,z-dz)),dz*2);
  };
  const roadX=(z:number)=>laneShift+bend(z);

  return <>
    <color attach="background" args={["#77819b"]}/><fog attach="fog" args={["#77819b",34,98]}/>
    <ambientLight intensity={2.15}/><directionalLight position={[10,20,10]} intensity={2.25} castShadow/>

    <mesh rotation={[-Math.PI/2,0,0]} position={[0,-.04,-48]} receiveShadow><planeGeometry args={[100,150]}/><meshStandardMaterial color="#555e72" roughness={.98}/></mesh>

    {lane.visible && <>
      {[-laneHalfWidth, laneHalfWidth].map(boundary=><group key={boundary}>{dashes.map(z=>{const x=roadX(z)+boundary;return <mesh key={`${boundary}-${z}`} rotation={[-Math.PI/2,0,-heading(z)]} position={[x,.018,-z]}><planeGeometry args={[.13,2.9]}/><meshBasicMaterial color="#f6f7fa"/></mesh>})}</group>)}
      {pathPieces.map(z=>{const x=roadX(z);return <mesh key={`path-${z}`} rotation={[-Math.PI/2,0,-heading(z)]} position={[x,.014,-z]}><planeGeometry args={[3.15,3.72]}/><meshBasicMaterial color="#4c88ff" transparent opacity={.12 + lane.confidence*.13}/></mesh>})}
    </>}

    {!lane.visible && <mesh rotation={[-Math.PI/2,0,0]} position={[0,-.005,-42]}><planeGeometry args={[5.8,96]}/><meshStandardMaterial color="#454e61" roughness={.92}/></mesh>}

    <group position={[0,.05,1.6]}><mesh position={[0,.55,0]} castShadow><boxGeometry args={[1.85,1.05,4.2]}/><meshStandardMaterial color="#f6f7f9" metalness={.2} roughness={.3}/></mesh><mesh position={[0,1.12,-.2]}><boxGeometry args={[1.5,.55,2.16]}/><meshStandardMaterial color="#8fa8c5" metalness={.16} roughness={.18}/></mesh></group>

    {objects.slice(0,28).map(object=>{const z=THREE.MathUtils.clamp(object.distance*2.2+4,7,84);const x=roadX(z)+THREE.MathUtils.clamp(object.lateral*8,-14,14);if(object.class==="person")return <Pedestrian key={object.id} x={x} z={z}/>;if(object.class==="traffic light"||object.class==="stop sign")return <TrafficSignal key={object.id} x={x} z={z}/>;return <Vehicle key={object.id} x={x} z={z} kind={object.class}/>})}
  </>;
}

export default function AdasScene(props:Props){return <div className="adas-scene"><Canvas shadows dpr={[1,1.5]} camera={{position:[0,9.3,14.8],fov:51,near:.1,far:150}} onCreated={({camera})=>camera.lookAt(0,.15,-30)}><RoadScene {...props}/></Canvas><div className="scene-badge">3D PERCEPTION</div><div className="scene-horizon">CURVATURE-AWARE ROAD MODEL</div></div>}
