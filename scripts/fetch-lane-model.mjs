import fs from "node:fs/promises";
import { createWriteStream } from "node:fs";
import path from "node:path";
import os from "node:os";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { extract } from "tar";

const MODEL_URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/324_Ultra-Fast-Lane-Detection-v2/resources.tar.gz";
const CANDIDATES = [
  "ufldv2_culane_res18_320x1600.onnx",
  "ufldv2_culane_res34_320x1600.onnx"
];
const outputDir = path.join(process.cwd(), "public", "models");
const outputPath = path.join(outputDir, "ufldv2.onnx");

async function exists(file) {
  try { await fs.access(file); return true; } catch { return false; }
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

if (await exists(outputPath)) {
  const stat = await fs.stat(outputPath);
  if (stat.size > 1_000_000) {
    console.log(`UFLDv2 model already present (${(stat.size / 1024 / 1024).toFixed(1)} MB).`);
    process.exit(0);
  }
}

await fs.mkdir(outputDir, { recursive: true });
const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "ufldv2-"));
const archivePath = path.join(tempDir, "resources.tar.gz");

try {
  console.log("Streaming UFLDv2 lane model archive…");
  const response = await fetch(MODEL_URL);
  if (!response.ok || !response.body) throw new Error(`Model download failed: ${response.status}`);

  // Stream to disk instead of materialising the whole archive in memory. Vercel's
  // build container previously OOM-killed the process while using arrayBuffer().
  await pipeline(Readable.fromWeb(response.body), createWriteStream(archivePath));
  console.log("Archive downloaded; extracting model…");
  await extract({ file: archivePath, cwd: tempDir });

  const modelPath = await findCandidate(tempDir);
  if (!modelPath) throw new Error(`No supported CULane UFLDv2 ONNX model found in archive`);
  await fs.copyFile(modelPath, outputPath);

  const stat = await fs.stat(outputPath);
  if (stat.size < 1_000_000) throw new Error("Downloaded model is unexpectedly small");
  console.log(`UFLDv2 model ready: ${path.basename(modelPath)} (${(stat.size / 1024 / 1024).toFixed(1)} MB).`);
} finally {
  await fs.rm(tempDir, { recursive: true, force: true });
}
