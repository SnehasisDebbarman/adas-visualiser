import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { extract } from "tar";

const MODEL_URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/324_Ultra-Fast-Lane-Detection-v2/resources.tar.gz";
const TARGET_NAME = "ufldv2_culane_res18_320x1600.onnx";
const outputDir = path.join(process.cwd(), "public", "models");
const outputPath = path.join(outputDir, "ufldv2.onnx");

async function exists(file) {
  try { await fs.access(file); return true; } catch { return false; }
}

async function findFile(root, name) {
  const entries = await fs.readdir(root, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) {
      const nested = await findFile(full, name);
      if (nested) return nested;
    } else if (entry.name === name) {
      return full;
    }
  }
  return null;
}

if (await exists(outputPath)) {
  console.log("UFLDv2 model already present.");
  process.exit(0);
}

await fs.mkdir(outputDir, { recursive: true });
const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "ufldv2-"));
const archivePath = path.join(tempDir, "resources.tar.gz");

try {
  console.log("Downloading UFLDv2 lane model…");
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`Model download failed: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  await fs.writeFile(archivePath, bytes);
  await extract({ file: archivePath, cwd: tempDir });
  const modelPath = await findFile(tempDir, TARGET_NAME);
  if (!modelPath) throw new Error(`${TARGET_NAME} was not found in model archive`);
  await fs.copyFile(modelPath, outputPath);
  const stat = await fs.stat(outputPath);
  console.log(`UFLDv2 model ready (${(stat.size / 1024 / 1024).toFixed(1)} MB).`);
} catch (error) {
  console.warn("UFLDv2 model fetch failed; browser will use classical lane fallback.");
  console.warn(error);
} finally {
  await fs.rm(tempDir, { recursive: true, force: true });
}
