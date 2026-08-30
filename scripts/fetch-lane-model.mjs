import fs from "node:fs/promises";
import { createWriteStream } from "node:fs";
import path from "node:path";
import os from "node:os";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { spawnSync } from "node:child_process";
import { extract } from "tar";

const MODEL_URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/324_Ultra-Fast-Lane-Detection-v2/resources.tar.gz";
const CANDIDATES = [
  "ufldv2_culane_res18_320x1600.onnx",
  "ufldv2_culane_res34_320x1600.onnx"
];
const outputDir = path.join(process.cwd(), "public", "models");
const outputPath = path.join(outputDir, "ufldv2-culane-v1.onnx");
const fp16Path = path.join(outputDir, "ufldv2-culane-v1-fp16.onnx");

async function exists(file) {
  try { await fs.access(file); return true; } catch { return false; }
}

async function validModel(file) {
  if (!(await exists(file))) return false;
  const stat = await fs.stat(file);
  return stat.size > 1_000_000;
}

async function findCandidate(root) {
  const entries = await fs.readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) {
      const nested = await findCandidate(full);
      if (nested) return nested;
    } else if (CANDIDATES.includes(entry.name)) {
      return full;
    }
  }
  return null;
}

function run(command, args, options = {}) {
  return spawnSync(command, args, { stdio: "inherit", ...options });
}

async function tryCreateFp16() {
  if (await validModel(fp16Path)) {
    const stat = await fs.stat(fp16Path);
    console.log(`FP16 UFLDv2 already present (${(stat.size / 1024 / 1024).toFixed(1)} MB).`);
    return;
  }

  const python = process.platform === "win32" ? "python" : "python3";
  const toolsDir = path.join(os.tmpdir(), "adas-onnx-tools");
  await fs.mkdir(toolsDir, { recursive: true });

  console.log("Preparing optional FP16 UFLDv2 model…");
  const install = run(python, [
    "-m", "pip", "install", "--disable-pip-version-check", "--quiet",
    "--target", toolsDir, "onnx>=1.16,<2", "onnxconverter-common>=1.14,<2"
  ]);
  if (install.error || install.status !== 0) {
    console.warn("FP16 converter dependencies could not be installed; keeping FP32 model only.");
    return;
  }

  const env = { ...process.env, PYTHONPATH: toolsDir };
  const conversion = run(python, [
    path.join(process.cwd(), "scripts", "convert-lane-model-fp16.py"),
    outputPath,
    fp16Path
  ], { env });

  if (conversion.error || conversion.status !== 0 || !(await validModel(fp16Path))) {
    console.warn("FP16 UFLDv2 conversion failed; WebGPU clients will fall back to FP32.");
    await fs.rm(fp16Path, { force: true }).catch(() => undefined);
  }
}

await fs.mkdir(outputDir, { recursive: true });

if (!(await validModel(outputPath))) {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "ufldv2-"));
  const archivePath = path.join(tempDir, "resources.tar.gz");

  try {
    console.log("Streaming UFLDv2 lane model archive…");
    const response = await fetch(MODEL_URL);
    if (!response.ok || !response.body) throw new Error(`Model download failed: ${response.status}`);

    await pipeline(Readable.fromWeb(response.body), createWriteStream(archivePath));
    console.log("Archive downloaded; extracting model…");
    await extract({ file: archivePath, cwd: tempDir });

    const modelPath = await findCandidate(tempDir);
    if (!modelPath) throw new Error("No supported CULane UFLDv2 ONNX model found in archive");
    await fs.copyFile(modelPath, outputPath);

    const stat = await fs.stat(outputPath);
    if (stat.size < 1_000_000) throw new Error("Downloaded model is unexpectedly small");
    console.log(`UFLDv2 model ready: ${path.basename(modelPath)} (${(stat.size / 1024 / 1024).toFixed(1)} MB).`);
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
} else {
  const stat = await fs.stat(outputPath);
  console.log(`UFLDv2 FP32 model already present (${(stat.size / 1024 / 1024).toFixed(1)} MB).`);
}

await tryCreateFp16();
