"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";

export type SceneObject = { id: number; class: string; lateral: number; distance: number };
export type SceneLane = { centerOffset: number; confidence: number; visible: boolean; curve: number; roadDepth?: number };
type Props = { objects: SceneObject[]; lane: SceneLane };
type InstanceSpec = { id?: number; x: number; y: number; z: number; sx?: number; sy?: number; sz?: number; ry?: number };

const tempObject = new THREE.Object3D();
const tempMatrix = new THREE.Matrix4();
const tempPosition = new THREE.Vector3();
const tempQuaternion = new THREE.Quaternion();
const tempScale = new THREE.Vector3();
const targetPosition = new THREE.Vector3();
const targetQuaternion = new THREE.Quaternion();
const targetScale = new THREE.Vector3();

function makeRoadRibbon(depth: number, laneShift: number, curve: number, halfWidth = 5.4) {
  const segments = 28;
  const positions: number[] = [];
  const indices: number[] = [];
  const roadX = (z: number) => laneShift + curve * (z / depth) * (z / depth);
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const z = 3 + t * (depth - 3);
    const perspectiveWidth = halfWidth * (1 - t * 0.46);
    const cx = roadX(z);
    positions.push(cx - perspectiveWidth, 0, -z, cx + perspectiveWidth, 0, -z);
    if (i < segments) {
      const a = i * 2;
      indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function makePathRibbon(depth: number, laneShift: number, curve: number, confidence: number) {
  const segments = 22;
  const positions: number[] = [];
  const indices: number[] = [];
  const roadX = (z: number) => laneShift + curve * (z / depth) * (z / depth);
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const z = 4 + t * Math.max(10, depth - 9);
    const width = 1.45 * (1 - t * 0.38) * (0.92 + confidence * 0.08);
    const cx = roadX(z);
    positions.push(cx - width, 0.018, -z, cx + width, 0.018, -z);
    if (i < segments) {
      const a = i * 2;
      indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function StaticInstances({ items, geometry, material }: { items: InstanceSpec[]; geometry: THREE.BufferGeometry; material: THREE.Material }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      tempObject.position.set(item.x, item.y, item.z);
      tempObject.rotation.set(0, item.ry ?? 0, 0);
      tempObject.scale.set(item.sx ?? 1, item.sy ?? 1, item.sz ?? 1);
      tempObject.updateMatrix();
      mesh.setMatrixAt(i, tempObject.matrix);
    }
    mesh.count = items.length;
    mesh.instanceMatrix.needsUpdate = true;
  }, [items]);
  if (!items.length) return null;
  return <instancedMesh ref={ref} args={[geometry, material, Math.max(1, items.length)]} frustumCulled={false} />;
}

function MovingRoadInstances({ items, geometry, material, roadDepth, speed }: { items: InstanceSpec[]; geometry: THREE.BufferGeometry; material: THREE.Material; roadDepth: number; speed: number }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const offsetRef = useRef(0);
  useFrame((_, delta) => {
    const mesh = ref.current;
    if (!mesh || !items.length) return;
    offsetRef.current = (offsetRef.current + delta * speed) % 6.2;
    const wrap = Math.max(9, roadDepth - 4);
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      let depth = Math.abs(item.z) - offsetRef.current;
      while (depth < 4) depth += wrap;
      if (depth > roadDepth) depth = 4 + ((depth - 4) % wrap);
      tempObject.position.set(item.x, item.y, -depth);
      tempObject.rotation.set(0, item.ry ?? 0, 0);
      tempObject.scale.set(item.sx ?? 1, item.sy ?? 1, item.sz ?? 1);
      tempObject.updateMatrix();
      mesh.setMatrixAt(i, tempObject.matrix);
    }
    mesh.count = items.length;
    mesh.instanceMatrix.needsUpdate = true;
  });
  if (!items.length) return null;
  return <instancedMesh ref={ref} args={[geometry, material, Math.max(1, items.length)]} frustumCulled={false} />;
}

function SmoothInstances({ items, geometry, material, damping = 10 }: { items: InstanceSpec[]; geometry: THREE.BufferGeometry; material: THREE.Material; damping?: number }) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const current = useRef<THREE.Matrix4[]>([]);
  useLayoutEffect(() => {
    if (current.current.length > items.length) current.current.length = items.length;
    while (current.current.length < items.length) current.current.push(new THREE.Matrix4().makeScale(0, 0, 0));
  }, [items.length]);
  useFrame((_, delta) => {
    const mesh = ref.current;
    if (!mesh) return;
    const alpha = 1 - Math.exp(-damping * Math.min(delta, 0.05));
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const target = tempMatrix.compose(
        targetPosition.set(item.x, item.y, item.z),
        targetQuaternion.setFromEuler(new THREE.Euler(0, item.ry ?? 0, 0)),
        targetScale.set(item.sx ?? 1, item.sy ?? 1, item.sz ?? 1)
      );
      const matrix = current.current[i];
      if (matrix.determinant() === 0) matrix.copy(target);
      else {
        matrix.decompose(tempPosition, tempQuaternion, tempScale);
        tempPosition.lerp(targetPosition, alpha);
        tempQuaternion.slerp(targetQuaternion, alpha);
        tempScale.lerp(targetScale, alpha);
        matrix.compose(tempPosition, tempQuaternion, tempScale);
      }
      mesh.setMatrixAt(i, matrix);
    }
    mesh.count = items.length;
    mesh.instanceMatrix.needsUpdate = true;
  });
  if (!items.length) return null;
  return <instancedMesh ref={ref} args={[geometry, material, Math.max(1, items.length)]} frustumCulled={false} />;
}

function RoadScene({ objects, lane }: Props) {
  const laneShift = THREE.MathUtils.clamp(lane.centerOffset * 2.35, -1.05, 1.05);
  const curve = THREE.MathUtils.clamp(lane.curve, -1, 1) * 5.6;
  const roadDepth = THREE.MathUtils.clamp(lane.roadDepth ?? 70, 34, 88);
  const roadX = (z: number) => laneShift + curve * (z / roadDepth) * (z / roadDepth);
  const heading = (z: number) => Math.atan((2 * curve * z) / (roadDepth * roadDepth));

  const resources = useMemo(() => ({
    laneGeometry: new THREE.PlaneGeometry(0.09, 2.75),
    carGeometry: new THREE.BoxGeometry(1.7, 0.72, 3.7),
    roofGeometry: new THREE.BoxGeometry(1.18, 0.38, 1.68),
    wheelGeometry: new THREE.BoxGeometry(0.2, 0.26, 0.56),
    personGeometry: new THREE.CapsuleGeometry(0.14, 0.58, 2, 5),
    headGeometry: new THREE.SphereGeometry(0.18, 6, 5),
    poleGeometry: new THREE.CylinderGeometry(0.035, 0.035, 4.0, 5),
    signalGeometry: new THREE.BoxGeometry(0.36, 0.7, 0.2),
    laneMaterial: new THREE.MeshBasicMaterial({ color: "#f7f8fb" }),
    roadMaterial: new THREE.MeshBasicMaterial({ color: "#d7dade", side: THREE.DoubleSide }),
    pathMaterial: new THREE.MeshBasicMaterial({ color: "#4f86ff", transparent: true, opacity: 0.26, depthWrite: false, side: THREE.DoubleSide }),
    vehicleMaterial: new THREE.MeshBasicMaterial({ color: "#c9cdd3" }),
    roofMaterial: new THREE.MeshBasicMaterial({ color: "#8f99a7" }),
    wheelMaterial: new THREE.MeshBasicMaterial({ color: "#4b5058" }),
    personMaterial: new THREE.MeshBasicMaterial({ color: "#d8b48e" }),
    signalPoleMaterial: new THREE.MeshBasicMaterial({ color: "#9ca3ad" }),
    signalMaterial: new THREE.MeshBasicMaterial({ color: "#555b64" }),
    egoMaterial: new THREE.MeshBasicMaterial({ color: "#f2f3f5" }),
    egoGlassMaterial: new THREE.MeshBasicMaterial({ color: "#7f9ab4" }),
    groundMaterial: new THREE.MeshBasicMaterial({ color: "#eceeef" })
  }), []);

  const roadGeometry = useMemo(() => makeRoadRibbon(roadDepth, laneShift, curve), [roadDepth, laneShift, curve]);
  const pathGeometry = useMemo(() => makePathRibbon(roadDepth, laneShift, curve, lane.confidence), [roadDepth, laneShift, curve, lane.confidence]);

  const sceneData = useMemo(() => {
    const laneMarks: InstanceSpec[] = [];
    if (lane.visible) {
      const count = Math.max(5, Math.floor((roadDepth - 5) / 6.2));
      for (let i = 0; i < count; i++) {
        const z = 6 + i * 6.2;
        if (z > roadDepth) break;
        const ry = -heading(z);
        laneMarks.push({ x: roadX(z) - 1.78, y: 0.025, z: -z, ry });
        laneMarks.push({ x: roadX(z) + 1.78, y: 0.025, z: -z, ry });
      }
    }

    const vehicleBodies: InstanceSpec[] = [];
    const vehicleRoofs: InstanceSpec[] = [];
    const wheelBlocks: InstanceSpec[] = [];
    const peopleBodies: InstanceSpec[] = [];
    const peopleHeads: InstanceSpec[] = [];
    const signalPoles: InstanceSpec[] = [];
    const signalBoxes: InstanceSpec[] = [];

    for (const object of objects.slice(0, 16)) {
      const z = THREE.MathUtils.clamp(object.distance * 2.12 + 4, 7, roadDepth - 2.5);
      const x = roadX(z) + THREE.MathUtils.clamp(object.lateral * 6.9, -11.5, 11.5);
      if (object.class === "person") {
        peopleBodies.push({ id: object.id, x, y: 0.72, z: -z });
        peopleHeads.push({ id: object.id, x, y: 1.38, z: -z });
      } else if (object.class === "traffic light" || object.class === "stop sign") {
        signalPoles.push({ id: object.id, x, y: 2.0, z: -z });
        signalBoxes.push({ id: object.id, x, y: 3.87, z: -z });
      } else {
        const truck = object.class === "truck" || object.class === "bus";
        const bike = object.class === "motorcycle" || object.class === "bicycle";
        const sx = truck ? 1.12 : bike ? 0.48 : 1;
        const sy = truck ? 1.55 : bike ? 0.82 : 1;
        const sz = truck ? 1.33 : bike ? 0.6 : 1;
        vehicleBodies.push({ id: object.id, x, y: 0.42 * sy, z: -z, sx, sy, sz });
        if (!bike) {
          vehicleRoofs.push({ id: object.id, x, y: 0.91 * sy, z: -z - 0.12, sx: truck ? 1.12 : 1, sy: truck ? 1.35 : 1, sz: truck ? 1.2 : 1 });
          wheelBlocks.push({ x: x - 0.73 * sx, y: 0.19, z: -z - 0.88 * sz, sx, sy, sz });
          wheelBlocks.push({ x: x + 0.73 * sx, y: 0.19, z: -z - 0.88 * sz, sx, sy, sz });
        }
      }
    }
    return { laneMarks, vehicleBodies, vehicleRoofs, wheelBlocks, peopleBodies, peopleHeads, signalPoles, signalBoxes };
  }, [objects, lane.visible, laneShift, curve, roadDepth]);

  return <>
    <color attach="background" args={["#e9ebee"]} />
    <fog attach="fog" args={["#e9ebee", Math.max(24, roadDepth * 0.62), roadDepth + 9]} />

    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.055, -34]} geometry={new THREE.PlaneGeometry(90, 100)} material={resources.groundMaterial} />
    <mesh geometry={roadGeometry} material={resources.roadMaterial} />
    {lane.visible && <mesh geometry={pathGeometry} material={resources.pathMaterial} />}

    <MovingRoadInstances items={sceneData.laneMarks} geometry={resources.laneGeometry} material={resources.laneMaterial} roadDepth={roadDepth} speed={20} />
    <SmoothInstances items={sceneData.vehicleBodies} geometry={resources.carGeometry} material={resources.vehicleMaterial} damping={13} />
    <SmoothInstances items={sceneData.vehicleRoofs} geometry={resources.roofGeometry} material={resources.roofMaterial} damping={13} />
    <SmoothInstances items={sceneData.wheelBlocks} geometry={resources.wheelGeometry} material={resources.wheelMaterial} damping={13} />
    <SmoothInstances items={sceneData.peopleBodies} geometry={resources.personGeometry} material={resources.personMaterial} damping={11} />
    <SmoothInstances items={sceneData.peopleHeads} geometry={resources.headGeometry} material={resources.personMaterial} damping={11} />
    <SmoothInstances items={sceneData.signalPoles} geometry={resources.poleGeometry} material={resources.signalPoleMaterial} damping={10} />
    <SmoothInstances items={sceneData.signalBoxes} geometry={resources.signalGeometry} material={resources.signalMaterial} damping={10} />

    <group position={[0, 0.03, 1.45]}>
      <mesh position={[0, 0.39, 0]} geometry={resources.carGeometry} material={resources.egoMaterial} scale={[1.1, 0.9, 1.12]} />
      <mesh position={[0, 0.9, -0.18]} geometry={resources.roofGeometry} material={resources.egoGlassMaterial} scale={[1.05, 1.12, 1.16]} />
      <mesh position={[-0.77, 0.2, 0.86]} geometry={resources.wheelGeometry} material={resources.wheelMaterial} />
      <mesh position={[0.77, 0.2, 0.86]} geometry={resources.wheelGeometry} material={resources.wheelMaterial} />
    </group>
  </>;
}

export default function AdasScene(props: Props) {
  return <div className="adas-scene">
    <Canvas
      frameloop="always"
      dpr={1}
      camera={{ position: [0, 6.4, 11.2], fov: 54, near: 0.2, far: 102 }}
      gl={{ antialias: false, alpha: false, powerPreference: "high-performance", stencil: false }}
      onCreated={({ camera, gl }) => {
        camera.lookAt(0, 0.35, -25);
        gl.outputColorSpace = THREE.SRGBColorSpace;
      }}
    >
      <RoadScene {...props} />
    </Canvas>
    <div className="scene-badge">TESLA-STYLE 3D PERCEPTION</div>
    <div className="scene-horizon">LIVE ROAD + TRACKED OBJECTS</div>
  </div>;
}
