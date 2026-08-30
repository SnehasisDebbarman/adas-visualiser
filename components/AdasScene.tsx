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
    offsetRef.current = (offsetRef.current + delta * speed) % 6.1;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const baseDepth = Math.abs(item.z);
      let depth = baseDepth - offsetRef.current;
      while (depth < 4) depth += Math.max(8, roadDepth - 4);
      if (depth > roadDepth) depth = 4 + ((depth - 4) % Math.max(8, roadDepth - 4));
      const t = roadDepth > 0 ? depth / roadDepth : 0;
      const xScale = 0.88 + t * 0.12;
      tempObject.position.set(item.x * xScale, item.y, -depth);
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
    while (current.current.length < items.length) current.current.push(new THREE.Matrix4());
  }, [items.length]);

  useFrame((_, delta) => {
    const mesh = ref.current;
    if (!mesh) return;
    const alpha = 1 - Math.exp(-damping * Math.min(delta, 0.05));
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      const target = tempMatrix.compose(
        tempPosition.set(item.x, item.y, item.z),
        tempQuaternion.setFromEuler(new THREE.Euler(0, item.ry ?? 0, 0)),
        tempScale.set(item.sx ?? 1, item.sy ?? 1, item.sz ?? 1)
      );
      const matrix = current.current[i];
      if (matrix.elements[15] === 0) matrix.copy(target);
      else {
        matrix.decompose(tempPosition, tempQuaternion, tempScale);
        const targetPosition = new THREE.Vector3(item.x, item.y, item.z);
        const targetQuaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, item.ry ?? 0, 0));
        const targetScale = new THREE.Vector3(item.sx ?? 1, item.sy ?? 1, item.sz ?? 1);
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
  const laneShift = THREE.MathUtils.clamp(lane.centerOffset * 2.6, -1.2, 1.2);
  const curve = THREE.MathUtils.clamp(lane.curve, -1, 1) * 6.4;
  const roadDepth = THREE.MathUtils.clamp(lane.roadDepth ?? 72, 34, 88);
  const roadX = (z: number) => laneShift + curve * (z / roadDepth) * (z / roadDepth);
  const heading = (z: number) => Math.atan((2 * curve * z) / (roadDepth * roadDepth));

  const resources = useMemo(() => ({
    roadGeometry: new THREE.PlaneGeometry(24, 1),
    laneGeometry: new THREE.PlaneGeometry(0.11, 3.3),
    pathGeometry: new THREE.PlaneGeometry(3.1, 4.5),
    carGeometry: new THREE.BoxGeometry(1.65, 1, 3.6),
    roofGeometry: new THREE.BoxGeometry(1.25, 0.42, 1.75),
    personGeometry: new THREE.CapsuleGeometry(0.16, 0.62, 2, 5),
    headGeometry: new THREE.SphereGeometry(0.2, 6, 5),
    poleGeometry: new THREE.CylinderGeometry(0.04, 0.04, 4.2, 5),
    signalGeometry: new THREE.BoxGeometry(0.42, 0.82, 0.22),
    roadMaterial: new THREE.MeshBasicMaterial({ color: "#2f3541" }),
    laneMaterial: new THREE.MeshBasicMaterial({ color: "#f5f7fb" }),
    pathMaterial: new THREE.MeshBasicMaterial({ color: "#2d78ff", transparent: true, opacity: 0.18, depthWrite: false }),
    vehicleMaterial: new THREE.MeshBasicMaterial({ color: "#dfe5ec" }),
    roofMaterial: new THREE.MeshBasicMaterial({ color: "#8193a8" }),
    personMaterial: new THREE.MeshBasicMaterial({ color: "#f0d5b8" }),
    signalPoleMaterial: new THREE.MeshBasicMaterial({ color: "#aab2bd" }),
    signalMaterial: new THREE.MeshBasicMaterial({ color: "#252a33" }),
    egoMaterial: new THREE.MeshBasicMaterial({ color: "#f7f9fc" }),
    egoGlassMaterial: new THREE.MeshBasicMaterial({ color: "#58708a" })
  }), []);

  const sceneData = useMemo(() => {
    const laneMarks: InstanceSpec[] = [];
    const path: InstanceSpec[] = [];
    if (lane.visible) {
      const laneCount = Math.max(5, Math.floor((roadDepth - 5) / 6.1));
      for (let i = 0; i < laneCount; i++) {
        const z = 6 + i * 6.1;
        if (z > roadDepth) break;
        const ry = -heading(z);
        laneMarks.push({ x: roadX(z) - 1.8, y: 0.018, z: -z, ry });
        laneMarks.push({ x: roadX(z) + 1.8, y: 0.018, z: -z, ry });
      }
      const pathCount = Math.max(4, Math.floor((roadDepth - 4) / 6.5));
      for (let i = 0; i < pathCount; i++) {
        const z = 5 + i * 6.5;
        if (z > roadDepth) break;
        path.push({ x: roadX(z), y: 0.012, z: -z, ry: -heading(z) });
      }
    }

    const vehicleBodies: InstanceSpec[] = [];
    const vehicleRoofs: InstanceSpec[] = [];
    const peopleBodies: InstanceSpec[] = [];
    const peopleHeads: InstanceSpec[] = [];
    const signalPoles: InstanceSpec[] = [];
    const signalBoxes: InstanceSpec[] = [];

    for (const object of objects.slice(0, 14)) {
      const z = THREE.MathUtils.clamp(object.distance * 2.15 + 4, 7, roadDepth - 3);
      const x = roadX(z) + THREE.MathUtils.clamp(object.lateral * 7.5, -12, 12);
      if (object.class === "person") {
        peopleBodies.push({ id: object.id, x, y: 0.82, z: -z });
        peopleHeads.push({ id: object.id, x, y: 1.5, z: -z });
      } else if (object.class === "traffic light" || object.class === "stop sign") {
        signalPoles.push({ id: object.id, x, y: 2.1, z: -z });
        signalBoxes.push({ id: object.id, x, y: 4.05, z: -z });
      } else {
        const truck = object.class === "truck" || object.class === "bus";
        const bike = object.class === "motorcycle" || object.class === "bicycle";
        const sx = truck ? 1.16 : bike ? 0.48 : 1;
        const sy = truck ? 1.5 : bike ? 0.8 : 1;
        const sz = truck ? 1.3 : bike ? 0.58 : 1;
        vehicleBodies.push({ id: object.id, x, y: 0.55 * sy, z: -z, sx, sy, sz });
        if (!bike) vehicleRoofs.push({ id: object.id, x, y: 1.05 * sy, z: -z - 0.2, sx: truck ? 1.12 : 1, sy: truck ? 1.25 : 1, sz: truck ? 1.2 : 1 });
      }
    }

    return { laneMarks, path, vehicleBodies, vehicleRoofs, peopleBodies, peopleHeads, signalPoles, signalBoxes };
  }, [objects, lane.visible, laneShift, curve, roadDepth]);

  return <>
    <color attach="background" args={["#596274"]} />
    <fog attach="fog" args={["#596274", Math.max(24, roadDepth * 0.55), roadDepth + 12]} />

    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.04, -(roadDepth / 2) + 2]} geometry={resources.roadGeometry} material={resources.roadMaterial} scale={[1, roadDepth - 4, 1]} />

    <MovingRoadInstances items={sceneData.laneMarks} geometry={resources.laneGeometry} material={resources.laneMaterial} roadDepth={roadDepth} speed={19} />
    <MovingRoadInstances items={sceneData.path} geometry={resources.pathGeometry} material={resources.pathMaterial} roadDepth={roadDepth} speed={15} />
    <SmoothInstances items={sceneData.vehicleBodies} geometry={resources.carGeometry} material={resources.vehicleMaterial} damping={12} />
    <SmoothInstances items={sceneData.vehicleRoofs} geometry={resources.roofGeometry} material={resources.roofMaterial} damping={12} />
    <SmoothInstances items={sceneData.peopleBodies} geometry={resources.personGeometry} material={resources.personMaterial} damping={10} />
    <SmoothInstances items={sceneData.peopleHeads} geometry={resources.headGeometry} material={resources.personMaterial} damping={10} />
    <SmoothInstances items={sceneData.signalPoles} geometry={resources.poleGeometry} material={resources.signalPoleMaterial} damping={9} />
    <SmoothInstances items={sceneData.signalBoxes} geometry={resources.signalGeometry} material={resources.signalMaterial} damping={9} />

    <group position={[0, 0.05, 1.6]}>
      <mesh position={[0, 0.52, 0]} geometry={resources.carGeometry} material={resources.egoMaterial} scale={[1.08, 1.05, 1.14]} />
      <mesh position={[0, 1.08, -0.18]} geometry={resources.roofGeometry} material={resources.egoGlassMaterial} scale={[1.05, 1.1, 1.15]} />
    </group>
  </>;
}

export default function AdasScene(props: Props) {
  return <div className="adas-scene">
    <Canvas
      frameloop="always"
      dpr={1}
      camera={{ position: [0, 8.8, 14.4], fov: 50, near: 0.2, far: 105 }}
      gl={{ antialias: false, alpha: false, powerPreference: "high-performance", stencil: false }}
      onCreated={({ camera, gl }) => {
        camera.lookAt(0, 0.15, -29);
        gl.outputColorSpace = THREE.SRGBColorSpace;
      }}
    >
      <RoadScene {...props} />
    </Canvas>
    <div className="scene-badge">FAST 3D PERCEPTION</div>
    <div className="scene-horizon">LIVE ROAD DEPTH + SMOOTH MOTION</div>
  </div>;
}
