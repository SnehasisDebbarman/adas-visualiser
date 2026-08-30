"use client";

import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";

export type SceneObject = {
  id: number;
  class: string;
  lateral: number;
  distance: number;
};

export type SceneLane = {
  centerOffset: number;
  confidence: number;
  visible: boolean;
};

type Props = {
  objects: SceneObject[];
  lane: SceneLane;
};

function Vehicle({ x, z, kind }: { x: number; z: number; kind: string }) {
  const truck = kind === "truck" || kind === "bus";
  const motorcycle = kind === "motorcycle" || kind === "bicycle";
  const size: [number, number, number] = truck ? [1.9, 1.65, 4.5] : motorcycle ? [0.7, 1.15, 1.9] : [1.65, 1.15, 3.6];
  return (
    <group position={[x, size[1] / 2 + 0.04, -z]}>
      <mesh castShadow>
        <boxGeometry args={size} />
        <meshStandardMaterial color={truck ? "#cfd5dc" : "#e7eaee"} roughness={0.42} metalness={0.18} />
      </mesh>
      {!motorcycle && (
        <mesh position={[0, size[1] * 0.34, -0.25]}>
          <boxGeometry args={[size[0] * 0.76, size[1] * 0.48, size[2] * 0.48]} />
          <meshStandardMaterial color="#aeb8c5" roughness={0.32} metalness={0.12} />
        </mesh>
      )}
    </group>
  );
}

function Pedestrian({ x, z }: { x: number; z: number }) {
  return (
    <group position={[x, 0, -z]}>
      <mesh position={[0, 0.9, 0]}>
        <capsuleGeometry args={[0.18, 0.75, 5, 10]} />
        <meshStandardMaterial color="#d7dbe1" />
      </mesh>
      <mesh position={[0, 1.55, 0]}>
        <sphereGeometry args={[0.22, 12, 12]} />
        <meshStandardMaterial color="#eceff2" />
      </mesh>
    </group>
  );
}

function LaneMark({ x, z, length = 4 }: { x: number; z: number; length?: number }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[x, 0.016, -z]}>
      <planeGeometry args={[0.11, length]} />
      <meshBasicMaterial color="#f5f7fb" />
    </mesh>
  );
}

function RoadScene({ objects, lane }: Props) {
  const laneShift = THREE.MathUtils.clamp(lane.centerOffset * 2.8, -1.2, 1.2);
  const laneXs = [-5.4, -1.8, 1.8, 5.4].map((x) => x + laneShift);
  const dashes = useMemo(() => Array.from({ length: 13 }, (_, i) => 6 + i * 7), []);

  return (
    <>
      <color attach="background" args={["#737d99"]} />
      <fog attach="fog" args={["#737d99", 32, 92]} />
      <ambientLight intensity={2.1} />
      <directionalLight position={[8, 18, 8]} intensity={2.3} castShadow />

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, -40]} receiveShadow>
        <planeGeometry args={[80, 120]} />
        <meshStandardMaterial color="#515a70" roughness={0.96} />
      </mesh>

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, -39.5]}>
        <planeGeometry args={[17.5, 82]} />
        <meshStandardMaterial color="#444d61" roughness={0.9} />
      </mesh>

      {laneXs.map((x) => (
        <group key={x}>
          {dashes.map((z) => <LaneMark key={`${x}-${z}`} x={x} z={z} length={3.4} />)}
        </group>
      ))}

      {lane.visible && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[laneShift, 0.008, -32]}>
          <planeGeometry args={[3.5, 62]} />
          <meshBasicMaterial color="#4d8bff" transparent opacity={0.10 + lane.confidence * 0.08} />
        </mesh>
      )}

      <group position={[0, 0.05, 1.4]}>
        <mesh position={[0, 0.55, 0]} castShadow>
          <boxGeometry args={[1.8, 1.05, 4.15]} />
          <meshStandardMaterial color="#f5f6f8" metalness={0.22} roughness={0.32} />
        </mesh>
        <mesh position={[0, 1.12, -0.2]}>
          <boxGeometry args={[1.5, 0.55, 2.15]} />
          <meshStandardMaterial color="#8ca6c4" metalness={0.18} roughness={0.2} />
        </mesh>
      </group>

      {objects.slice(0, 24).map((object) => {
        const x = THREE.MathUtils.clamp(object.lateral * 7.2, -13, 13);
        const z = THREE.MathUtils.clamp(object.distance * 2.15 + 4, 7, 78);
        if (object.class === "person") return <Pedestrian key={object.id} x={x} z={z} />;
        return <Vehicle key={object.id} x={x} z={z} kind={object.class} />;
      })}
    </>
  );
}

export default function AdasScene(props: Props) {
  return (
    <div className="adas-scene">
      <Canvas shadows camera={{ position: [0, 8.8, 13.5], fov: 53, near: 0.1, far: 140 }} onCreated={({ camera }) => camera.lookAt(0, 0, -27)}>
        <RoadScene {...props} />
      </Canvas>
      <div className="scene-badge">3D PERCEPTION</div>
    </div>
  );
}
