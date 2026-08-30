"use client";

import { Canvas } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";

export type SceneObject = { id: number; class: string; lateral: number; distance: number };
export type SceneLane = { centerOffset: number; confidence: number; visible: boolean; curve: number };
type Props = { objects: SceneObject[]; lane: SceneLane };

type InstanceSpec = { x: number; y: number; z: number; sx?: number; sy?: number; sz?: number; ry?: number };

const tempObject = new THREE.Object3D();

function Instances({ items, geometry, material }: { items: InstanceSpec[]; geometry: THREE.BufferGeometry; material: THREE.Material }) {
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
  return <instancedMesh ref={ref} args={[geometry, material, items.length]} frustumCulled={false} />;
}

function RoadScene({ objects, lane }: Props) {
  const laneShift = THREE.MathUtils.clamp(lane.centerOffset * 2.6, -1.2, 1.2);
  const curve = THREE.MathUtils.clamp(lane.curve, -1, 1) * 6.4;
  const roadX = (z: number) => laneShift + curve * (z / 86) * (z / 86);
  const heading = (z: number) => Math.atan((2 * curve * z) / (86 * 86));

  const resources = useMemo(() => ({
    roadGeometry: new THREE.PlaneGeometry(26, 110),
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
      for (let i = 0; i < 13; i++) {
        const z = 7 + i * 6.1;
        const ry = -heading(z);
        laneMarks.push({ x: roadX(z) - 1.8, y: 0.018, z: -z, ry });
        laneMarks.push({ x: roadX(z) + 1.8, y: 0.018, z: -z, ry });
      }
      for (let i = 0; i < 12; i++) {
        const z = 5 + i * 6.5;
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
      const z = THREE.MathUtils.clamp(object.distance * 2.15 + 4, 7, 82);
      const x = roadX(z) + THREE.MathUtils.clamp(object.lateral * 7.5, -12, 12);
      if (object.class === "person") {
        peopleBodies.push({ x, y: 0.82, z: -z });
        peopleHeads.push({ x, y: 1.5, z: -z });
      } else if (object.class === "traffic light" || object.class === "stop sign") {
        signalPoles.push({ x, y: 2.1, z: -z });
        signalBoxes.push({ x, y: 4.05, z: -z });
      } else {
        const truck = object.class === "truck" || object.class === "bus";
        const bike = object.class === "motorcycle" || object.class === "bicycle";
        const sx = truck ? 1.16 : bike ? 0.48 : 1;
        const sy = truck ? 1.5 : bike ? 0.8 : 1;
        const sz = truck ? 1.3 : bike ? 0.58 : 1;
        vehicleBodies.push({ x, y: 0.55 * sy, z: -z, sx, sy, sz });
        if (!bike) vehicleRoofs.push({ x, y: 1.05 * sy, z: -z - 0.2, sx: truck ? 1.12 : 1, sy: truck ? 1.25 : 1, sz: truck ? 1.2 : 1 });
      }
    }

    return { laneMarks, path, vehicleBodies, vehicleRoofs, peopleBodies, peopleHeads, signalPoles, signalBoxes };
  }, [objects, lane.visible, laneShift, curve]);

  return <>
    <color attach="background" args={["#596274"]} />
    <fog attach="fog" args={["#596274", 36, 94]} />

    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.04, -45]} geometry={resources.roadGeometry} material={resources.roadMaterial} />

    <Instances items={sceneData.laneMarks} geometry={resources.laneGeometry} material={resources.laneMaterial} />
    <Instances items={sceneData.path} geometry={resources.pathGeometry} material={resources.pathMaterial} />
    <Instances items={sceneData.vehicleBodies} geometry={resources.carGeometry} material={resources.vehicleMaterial} />
    <Instances items={sceneData.vehicleRoofs} geometry={resources.roofGeometry} material={resources.roofMaterial} />
    <Instances items={sceneData.peopleBodies} geometry={resources.personGeometry} material={resources.personMaterial} />
    <Instances items={sceneData.peopleHeads} geometry={resources.headGeometry} material={resources.personMaterial} />
    <Instances items={sceneData.signalPoles} geometry={resources.poleGeometry} material={resources.signalPoleMaterial} />
    <Instances items={sceneData.signalBoxes} geometry={resources.signalGeometry} material={resources.signalMaterial} />

    <group position={[0, 0.05, 1.6]}>
      <mesh position={[0, 0.52, 0]} geometry={resources.carGeometry} material={resources.egoMaterial} scale={[1.08, 1.05, 1.14]} />
      <mesh position={[0, 1.08, -0.18]} geometry={resources.roofGeometry} material={resources.egoGlassMaterial} scale={[1.05, 1.1, 1.15]} />
    </group>
  </>;
}

export default function AdasScene(props: Props) {
  return <div className="adas-scene">
    <Canvas
      frameloop="demand"
      dpr={1}
      camera={{ position: [0, 8.8, 14.4], fov: 50, near: 0.2, far: 115 }}
      gl={{ antialias: false, alpha: false, powerPreference: "high-performance", stencil: false }}
      onCreated={({ camera, gl }) => {
        camera.lookAt(0, 0.15, -29);
        gl.outputColorSpace = THREE.SRGBColorSpace;
      }}
    >
      <RoadScene {...props} />
    </Canvas>
    <div className="scene-badge">FAST 3D PERCEPTION</div>
    <div className="scene-horizon">GPU INSTANCED ROAD MODEL</div>
  </div>;
}
