// One scene, two variables. The floor is the fixed physical page pool: tile
// colour says which request owns that page, so continuous batching reads as
// several colours coexisting and recycling in a pool that never grows. The
// pillars behind it are runtime components: height says whether the component
// is attached right now, so just-in-time residency reads as volume that rises
// only for the phase that needs it. Nothing else moves, and no text lives in
// the canvas — labels are flat HTML outside it.
import * as THREE from "https://unpkg.com/three@0.169.0/build/three.module.js";

const OWNER_COLOR = {
  A: 0x87b9dc,
  B: 0xd8b274,
  C: 0xadc696,
  D: 0xd89a8e,
  E: 0x7fcbbb,
  free: 0x1d3327,
};

const PILLAR_COLOR = 0x9ec98a;
const ROWS = 3;
const COLS = 6; // 18 slots, matching the schematic pool size
const GAP = 1.06;
const PILLAR_Z = -(ROWS - 1) / 2 * GAP - 1.15;

export function mountPagePool(container) {
  if (!container) return null;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.58));
  const key = new THREE.DirectionalLight(0xffffff, 1.15);
  key.position.set(4, 8, 5);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x7fcbbb, 0.42);
  rim.position.set(-5, 3, -5);
  scene.add(rim);

  // --- floor: the fixed page pool -----------------------------------------
  const tileGeo = new THREE.BoxGeometry(0.9, 0.22, 0.58);
  const tileMat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 });
  const tiles = new THREE.InstancedMesh(tileGeo, tileMat, ROWS * COLS);
  scene.add(tiles);

  const dummy = new THREE.Object3D();
  const tilePos = [];
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const x = (c - (COLS - 1) / 2) * GAP;
      const z = (r - (ROWS - 1) / 2) * GAP;
      tilePos.push({ x, z });
    }
  }

  const tileTarget = tilePos.map(() => new THREE.Color(OWNER_COLOR.free));
  const tileCurrent = tileTarget.map((c) => c.clone());
  const liftTarget = new Array(tilePos.length).fill(0);
  const liftCurrent = new Array(tilePos.length).fill(0);
  tileCurrent.forEach((c, i) => tiles.setColorAt(i, c));

  // --- pillars: components, present only while attached --------------------
  const pillarGeo = new THREE.BoxGeometry(0.62, 1, 0.62);
  pillarGeo.translate(0, 0.5, 0); // grow upward from the floor
  const pillars = [0, 1, 2].map((i) => {
    const mesh = new THREE.Mesh(
      pillarGeo,
      new THREE.MeshStandardMaterial({ color: PILLAR_COLOR, roughness: 0.42, transparent: true })
    );
    mesh.position.set((i - 1) * 1.5, 0, PILLAR_Z);
    scene.add(mesh);
    const pad = new THREE.Mesh(
      new THREE.BoxGeometry(0.78, 0.06, 0.78),
      new THREE.MeshStandardMaterial({ color: 0x182c21, roughness: 0.8 })
    );
    pad.position.set((i - 1) * 1.5, 0.03, PILLAR_Z);
    scene.add(pad);
    return mesh;
  });
  const pillarTarget = [0, 0, 0];
  const pillarCurrent = [0, 0, 0];

  function setFrame(owners, attached) {
    owners.forEach((owner, i) => {
      tileTarget[i] = new THREE.Color(OWNER_COLOR[owner] ?? OWNER_COLOR.free);
      liftTarget[i] = owner ? 0.2 : 0;
    });
    (attached || []).forEach((on, i) => {
      pillarTarget[i] = on ? 1 : 0;
    });
  }

  function frame() {
    const ease = reduceMotion ? 1 : 0.12;
    for (let i = 0; i < tilePos.length; i++) {
      tileCurrent[i].lerp(tileTarget[i], ease);
      tiles.setColorAt(i, tileCurrent[i]);
      liftCurrent[i] += (liftTarget[i] - liftCurrent[i]) * ease;
      dummy.position.set(tilePos[i].x, liftCurrent[i], tilePos[i].z);
      dummy.updateMatrix();
      tiles.setMatrixAt(i, dummy.matrix);
    }
    tiles.instanceMatrix.needsUpdate = true;
    if (tiles.instanceColor) tiles.instanceColor.needsUpdate = true;

    pillars.forEach((mesh, i) => {
      pillarCurrent[i] += (pillarTarget[i] - pillarCurrent[i]) * (reduceMotion ? 1 : 0.1);
      const h = Math.max(pillarCurrent[i], 0.001);
      mesh.scale.y = h * 1.7;
      mesh.material.opacity = Math.min(1, 0.25 + pillarCurrent[i] * 0.85);
      mesh.visible = pillarCurrent[i] > 0.02;
    });

    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    // Pull back and raise the camera on narrow viewports so the whole pool,
    // the pillars behind it, and the gap between them stay in frame.
    const narrow = w < 560;
    camera.position.set(0, narrow ? 3.7 : 3.1, narrow ? 6.6 : 5.7);
    camera.lookAt(0, narrow ? 0.62 : 0.52, -0.8);
    camera.updateProjectionMatrix();
  }

  resize();
  window.addEventListener("resize", resize);
  requestAnimationFrame(frame);

  return { setFrame };
}
