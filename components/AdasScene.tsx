"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { memo, useMemo, useRef } from "react";
import * as THREE from "three";

export type SceneObject = { id: number; class: string; lateral: number; distance: number };
export type SceneLane = { centerOffset: number; confidence: number; visible: boolean; curve: number; roadDepth?: number };
type Props = { objects: SceneObject[]; lane: SceneLane };
type RoadFn = (z: number) => number;

function ribbonGeometry(depth: number, centerAt: RoadFn, halfWidthAt: (z: number) => number, y = 0, start = 2.8, segments = 40) {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const z = start + (depth - start) * t;
    const c = centerAt(z);
    const w = halfWidthAt(z);
    positions.push(c - w, y, -z, c + w, y, -z);
    if (i < segments) {
      const a = i * 2;
      indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  return geometry;
}

function boundaryGeometry(depth: number, centerAt: RoadFn, offset: number, start = 4, segments = 38) {
  const width = 0.065;
  const positions: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const z = start + (depth - start) * t;
    const x = centerAt(z) + offset;
    positions.push(x - width, 0.035, -z, x + width, 0.035, -z);
    if (i < segments) {
      const a = i * 2;
      indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  return geometry;
}

const Vehicle = memo(function Vehicle({ object, roadX, roadDepth }: { object: SceneObject; roadX: RoadFn; roadDepth: number }) {
  const group = useRef<THREE.Group>(null);
  const target = useMemo(() => new THREE.Vector3(), []);
  const current = useMemo(() => new THREE.Vector3(), []);
  const initialized = useRef(false);
  const truck = object.class === "truck" || object.class === "bus";
  const bike = object.class === "motorcycle" || object.class === "bicycle";
  const z = THREE.MathUtils.clamp(object.distance * 2.05 + 4.5, 6.5, Math.max(8, roadDepth - 2));
  const x = roadX(z) + THREE.MathUtils.clamp(object.lateral * 6.4, -10.5, 10.5);

  useFrame((_, delta) => {
    if (!group.current) return;
    target.set(x, 0, -z);
    if (!initialized.current) {
      current.copy(target);
      initialized.current = true;
    }
    const alpha = 1 - Math.exp(-11 * Math.min(delta, 0.05));
    current.lerp(target, alpha);
    group.current.position.copy(current);
  });

  if (bike) {
    return (
      <group ref={group}>
        <mesh position={[0, 0.42, 0]} scale={[0.42, 0.8, 1.15]}>
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial color="#bbc1c8" />
        </mesh>
        <mesh position={[0, 0.9, -0.1]}>
          <capsuleGeometry args={[0.18, 0.6, 2, 5]} />
          <meshBasicMaterial color="#7f8995" />
        </mesh>
      </group>
    );
  }

  const bodyScale: [number, number, number] = truck ? [2.05, 1.45, 5.0] : [1.82, 0.72, 4.05];
  const cabinScale: [number, number, number] = truck ? [1.82, 0.86, 2.35] : [1.45, 0.58, 2.05];
  return (
    <group ref={group}>
      <mesh position={[0, bodyScale[1] * 0.5 + 0.18, 0]}>
        <boxGeometry args={bodyScale} />
        <meshBasicMaterial color={truck ? "#c4c9cf" : "#d7dbe0"} />
      </mesh>
      <mesh position={[0, truck ? 1.48 : 1.02, truck ? -0.35 : -0.15]}>
        <boxGeometry args={cabinScale} />
        <meshBasicMaterial color="#8996a5" />
      </mesh>
      <mesh position={[0, truck ? 0.45 : 0.35, bodyScale[2] * 0.505]} scale={[truck ? 1.65 : 1.45, 0.12, 0.05]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial color="#eef1f4" />
      </mesh>
    </group>
  );
});

const Person = memo(function Person({ object, roadX, roadDepth }: { object: SceneObject; roadX: RoadFn; roadDepth: number }) {
  const group = useRef<THREE.Group>(null);
  const z = THREE.MathUtils.clamp(object.distance * 2.05 + 4.5, 6.5, Math.max(8, roadDepth - 2));
  const x = roadX(z) + THREE.MathUtils.clamp(object.lateral * 6.4, -10.5, 10.5);
  useFrame((_, delta) => {
    if (!group.current) return;
    const a = 1 - Math.exp(-10 * Math.min(delta, 0.05));
    group.current.position.x = THREE.MathUtils.lerp(group.current.position.x, x, a);
    group.current.position.z = THREE.MathUtils.lerp(group.current.position.z, -z, a);
  });
  return (
    <group ref={group} position={[x, 0, -z]}>
      <mesh position={[0, 0.83, 0]}>
        <capsuleGeometry args={[0.16, 0.72, 2, 6]} />
        <meshBasicMaterial color="#8f98a3" />
      </mesh>
      <mesh position={[0, 1.48, 0]}>
        <sphereGeometry args={[0.19, 7, 6]} />
        <meshBasicMaterial color="#d4c2b1" />
      </mesh>
    </group>
  );
});

function EgoVehicle() {
  return (
    <group position={[0, 0, 2.7]}>
      <mesh position={[0, 0.36, 0]}>
        <boxGeometry args={[1.9, 0.62, 4.25]} />
        <meshBasicMaterial color="#f4f5f6" />
      </mesh>
      <mesh position={[0, 0.94, -0.18]}>
        <boxGeometry args={[1.5, 0.5, 2.12]} />
        <meshBasicMaterial color="#7e93aa" />
      </mesh>
      <mesh position={[0, 0.22, -2.14]} scale={[1.5, 0.12, 0.05]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial color="#ff5d67" />
      </mesh>
    </group>
  );
}

function Scene({ objects, lane }: Props) {
  const roadDepth = THREE.MathUtils.clamp(lane.roadDepth ?? 68, 32, 88);
  const laneShift = THREE.MathUtils.clamp(lane.centerOffset * 1.8, -0.85, 0.85);
  const curveStrength = THREE.MathUtils.clamp(lane.curve, -1, 1) * 7.2;
  const roadX = useMemo<RoadFn>(() => (z: number) => {
    const t = THREE.MathUtils.clamp(z / roadDepth, 0, 1);
    return laneShift * (0.25 + t * 0.75) + curveStrength * t * t * (0.42 + 0.58 * t);
  }, [laneShift, curveStrength, roadDepth]);

  const geometries = useMemo(() => ({
    road: ribbonGeometry(roadDepth, roadX, () => 5.2, 0, 2.8, 46),
    shoulderLeft: ribbonGeometry(roadDepth, z => roadX(z) - 5.42, () => 0.18, 0.012, 2.8, 40),
    shoulderRight: ribbonGeometry(roadDepth, z => roadX(z) + 5.42, () => 0.18, 0.012, 2.8, 40),
    path: ribbonGeometry(roadDepth - 3, roadX, z => 1.42 - 0.18 * (z / roadDepth), 0.025, 3.6, 38),
    leftLane: boundaryGeometry(roadDepth, roadX, -1.78),
    rightLane: boundaryGeometry(roadDepth, roadX, 1.78)
  }), [roadDepth, roadX]);

  const materials = useMemo(() => ({
    ground: new THREE.MeshBasicMaterial({ color: "#d9dde2" }),
    road: new THREE.MeshBasicMaterial({ color: "#2e343c", side: THREE.DoubleSide }),
    shoulder: new THREE.MeshBasicMaterial({ color: "#afb6be", side: THREE.DoubleSide }),
    lane: new THREE.MeshBasicMaterial({ color: "#f5f7f8", side: THREE.DoubleSide }),
    path: new THREE.MeshBasicMaterial({ color: "#3478f6", transparent: true, opacity: 0.34, depthWrite: false, side: THREE.DoubleSide })
  }), []);

  return (
    <>
      <color attach="background" args={["#cfd4da"]} />
      <fog attach="fog" args={["#cfd4da", Math.max(30, roadDepth * 0.72), roadDepth + 12]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.06, -36]} material={materials.ground}>
        <planeGeometry args={[90, 105]} />
      </mesh>
      <mesh geometry={geometries.road} material={materials.road} />
      <mesh geometry={geometries.shoulderLeft} material={materials.shoulder} />
      <mesh geometry={geometries.shoulderRight} material={materials.shoulder} />
      {lane.visible && <>
        <mesh geometry={geometries.path} material={materials.path} />
        <mesh geometry={geometries.leftLane} material={materials.lane} />
        <mesh geometry={geometries.rightLane} material={materials.lane} />
      </>}
      {objects.slice(0, 18).map(object => object.class === "person" ? (
        <Person key={object.id} object={object} roadX={roadX} roadDepth={roadDepth} />
      ) : object.class === "traffic light" || object.class === "stop sign" ? null : (
        <Vehicle key={object.id} object={object} roadX={roadX} roadDepth={roadDepth} />
      ))}
      <EgoVehicle />
    </>
  );
}

export default function AdasScene(props: Props) {
  return (
    <div className="adas-scene">
      <Canvas
        dpr={1}
        frameloop="always"
        camera={{ position: [0, 7.1, 14.6], fov: 47, near: 0.2, far: 120 }}
        gl={{ antialias: false, alpha: false, powerPreference: "high-performance", stencil: false }}
        onCreated={({ camera, gl }) => {
          camera.lookAt(0, 0.35, -29);
          gl.outputColorSpace = THREE.SRGBColorSpace;
        }}
      >
        <Scene {...props} />
      </Canvas>
      <div className="scene-badge">3D ROAD MODEL</div>
      <div className="scene-horizon">PERSISTENT TRACKS · REAL PERSPECTIVE</div>
    </div>
  );
}
